import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as func
from tqdm import tqdm
import numpy as np

class DataStore(Dataset):
    def __init__(self, data, feature_cols, target_col="RealizedVOl", seq_len=60):
      self.seq_len = seq_len
      self.xi_target = data["Xi"].values

      self.features = data[feature_cols].values
      self.target = data[target_col].shift(-1).values

      self.features = self.features[:-1]
      self.target = self.target[:-1]

    def __len__(self):
      return len(self.features) - self.seq_len

    def __getitem__(self, idx):
      x = self.features[idx: idx + self.seq_len]
      y = self.target[idx: idx + self.seq_len]

      xi_target = self.xi_target[idx + self.seq_len]
      xi_target = torch.tensor(xi_target, dtype=torch.float32).unsqueeze(0)

      x = torch.from_numpy(np.array(x, dtype=np.float32))
      y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

      return (x, y, xi_target)


class Encoder(nn.Module):
    def __init__(self, model_params: dict):
        super().__init__()
        self.seq_len = model_params["seq_len"]

        self.fc_in = nn.Linear(model_params["input_dim"], model_params["d_model"])

        self.pos_embd = nn.Parameter(
            torch.randn(1, model_params["seq_len"], model_params["d_model"])
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_params["d_model"],
            nhead=model_params["nhead"],
            dim_feedforward=4*model_params["d_model"],
            dropout=model_params["dropout"],
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=model_params["num_layers"]
        )

        self.fc_out = nn.Linear(model_params["d_model"], model_params["seq_len"])

    def forward(self, x):
        x = self.fc_in(x)
        x = x + self.pos_embd[:, :self.seq_len, :]

        x = self.transformer_encoder(x)
        z_seq = x

        return z_seq

class Decoder(nn.Module):
  def __init__(self, model_params):
    super().__init__()
    d_model = model_params["d_model"]
    output_dim = model_params["output_dim"]
    self.seq_len = model_params["seq_len"]

    self.log_alpha = nn.Parameter(torch.tensor(0.0))
    self.log_theta = nn.Parameter(torch.tensor(0.0))

    self.decoder = nn.Sequential(
        nn.Linear(d_model, 2*d_model),
        nn.GELU(),
        nn.LayerNorm(2*d_model),
        nn.Linear(2*d_model, d_model),
        nn.GELU(),
        nn.Linear(d_model, output_dim)
    )

  def forward(self, z_t):
    return self.decoder(z_t)


class NMVM(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim

    self.mu = nn.Parameter(torch.zeros(dim))
    self.beta = nn.Parameter(torch.zeros(dim))

    self.L_unconstrained = nn.Parameter(torch.eye(dim))

    self.log_alpha = nn.Parameter(torch.tensor(0.0))
    self.log_theta = nn.Parameter(torch.tensor(0.0))
    self.L = None

  def _get_L(self):
    L = torch.tril(self.L_unconstrained)
    diag = torch.diagonal(L)
    L = L - torch.diag(diag) + torch.diag(torch.exp(diag))
    return L

  @property
  def _params(self):
    return {
        "mu": self.mu.detach().cpu().numpy(),
        "beta": self.beta.detach().cpu().numpy(),
        "alpha": torch.exp(self.log_alpha).detach().cpu().numpy(),
        "theta": torch.exp(self.log_theta).detach().cpu().numpy(),
        "L": self.L.detach().cpu().numpy()
    }
    if len(self.L) > 0:
      return [self.mu, self.beta, self.log_alpha, self.log_theta, self.L]
    else:
      return [self.mu, self.beta, self.log_alpha, self.log_theta]


  def sample(self, n):
    device = self.mu.device
    alpha = torch.exp(self.log_alpha)
    theta = torch.exp(self.log_theta)

    self.L = self._get_L()
    gamma = dist.Gamma(alpha, 1/theta)
    W = gamma.sample((n,)).to(device)

    eps = torch.randn(n, self.dim, device=device)

    X = self.mu + self.beta*W.unsqueeze(-1) + torch.sqrt(W).unsqueeze(-1)*(eps @ self.L.T)

    return X

class CompositeLoss(nn.Module):
  def __init__(self):
    super().__init__()

    losses = ["rec", "smooth", "EVT", "conservative", "metric", "prior"]
    self.num_losses = len(losses)

    self.log_vars = nn.Parameter(torch.zeros(self.num_losses))

  def rec_loss(self, y_pred, y_true):
    return func.mse_loss(y_pred.squeeze(-1), y_true.squeeze(-1))

  def smoothness_loss(self, z):
    return torch.mean((z[:, 1:] - z[:, :-1])**2)

  def evt_loss(self, xi_pred, xi_target):
    return func.mse_loss(xi_pred, xi_target)

  def metric_loss(self, vol_pred, vol_true):
    dz = vol_pred[:, 1:] - vol_pred[:, :-1]
    dvol = vol_true[:, 1:] - vol_true[:, :-1]
    return func.mse_loss(dz, dvol)

  def conservative_loss(self, vol_pred, vol_target):
    under = torch.clamp(vol_target - vol_pred, min=0)
    return torch.mean(under**2)

  def prior_loss(self, z, mu, sigma):
    sigma_safe = torch.clamp(torch.diag(sigma), min=1e-6)
    loss = (z - mu)**2 / sigma_safe
    return torch.mean(loss)

  def forward(self, y_true, y_pred, z_sim, nmvm_params, xi_target):
    L = nmvm_params.get("L")
    mu = nmvm_params.get("mu")
    alpha = nmvm_params.get("alpha")
    cov = L @ L.T
    xi_pred = torch.trace(cov) / (2 * alpha + 1e-8)
    xi_pred = xi_pred.expand_as(xi_target)
    total_loss = 0
    losses = []

    losses.append(self.rec_loss(y_pred, y_true))
    losses.append(self.smoothness_loss(z_sim))
    losses.append(self.evt_loss(xi_pred, xi_target))
    losses.append(self.metric_loss(y_pred, y_true))
    losses.append(self.conservative_loss(y_pred, y_true))
    losses.append(self.prior_loss(y_pred, mu, L))

    if any(i.isnan() for i in losses):
      print(losses)

    for i, loss in enumerate(losses):
      precision = torch.exp(-self.log_vars[i])
      total_loss = total_loss + precision * loss + self.log_vars[i]

    return total_loss
      
class LatentSpaceModel(LatentSpaceVol):
  def __init__(self, feature_params, model_params):
    super().__init__(
      feature_params=feature_params
    )
    self.load_or_generate_features()
    model_params["model_params"]["input_dim"] = self.data.shape[1]

    self.encoder_params = model_params["model_params"]
    self.decoder_params = model_params["model_params"]
    self.train_settings = model_params["train_settings"]
    self.seq_len = model_params["model_params"]["seq_len"]


  def load_or_generate_features(self):
    filename = f"features_{self.params.ticker}.parquet"
    if os.path.exists(filename):
      self.data = pd.read_parquet(filename)

    else:
      self.get_price_data()
      self.generate_features()
      self.data.to_parquet(filename)

  def sample(self):
    return


  def fit(self):
    epochs = self.train_settings["epochs"]
    device = self.train_settings["device"]
    lr = self.train_settings["lr"]
    batch_size = self.train_settings["batch_size"]

    encoder = Encoder(self.encoder_params).to(device)
    decoder = Decoder(self.decoder_params).to(device)
    nmvm = NMVM(self.decoder_params["latent_dim"]).to(device)
    criterion = CompositeLoss().to(device)

    optimizer = torch.optim.Adam(
        (
            list(encoder.parameters())
            +list(decoder.parameters())
            +list(criterion.parameters())
            +list(nmvm.parameters())
        ),
        lr=lr
    )
    self.data = self.data.dropna()

    dataset = DataStore(self.data, feature_cols=self.data.columns)
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
        )

    epoch_loss = []

    for epoch in range(epochs):
        for x, y, xi in tqdm(train_loader):

          x = x.to(device)
          y = y.to(device)
          xi = xi.to(device)

          optimizer.zero_grad()

          z_real = encoder(x)
          y_pred = decoder(z_real)
          z_sim = nmvm.sample(z_real.shape[0])
          nmvm_params = nmvm._params

          loss = criterion(
              y_true=y,
              y_pred=y_pred,
              xi_target=xi,
              nmvm_params=nmvm_params,
              z_sim=z_sim
          )

          loss.backward()
          optimizer.step()
          epoch_loss.append(loss.item())

        print(f"Epoch: {epoch+1}, Train Loss: {sum(epoch_loss)/len(epoch_loss)}")
    return nmvm


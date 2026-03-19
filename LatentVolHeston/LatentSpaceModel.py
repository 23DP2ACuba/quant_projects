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

        self.fc_out = nn.Linear(model_params["d_model"], model_params["latent_dim"])

    def forward(self, x):
        x = self.fc_in(x)
        x = x + self.pos_embd[:, :self.seq_len, :]

        x = self.transformer_encoder(x)
        z_seq = self.fc_out(x)

        return z_seq

class Decoder(nn.Module):
  def __init__(self, model_params):
    super().__init__()
    latent_dim = model_params["latent_dim"]
    d_model = model_params["d_model"]
    output_dim = model_params["output_dim"]
    self.seq_len = model_params["seq_len"]

    self.log_alpha = nn.Parameter(torch.tensor(0.0))
    self.log_theta = nn.Parameter(torch.tensor(0.0))

    self.decoder = nn.Sequential(
        nn.Linear(latent_dim, 2*d_model),
        nn.GELU(),
        nn.LayerNorm(2*d_model),
        nn.Linear(2*d_model, d_model),
        nn.GELU(),
        nn.Linear(d_model, output_dim)
    )

  def forward(self, z_t):
    return nn.functional.softplus(self.decoder(z_t))


class NMVM(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim

    self.mu = nn.Parameter(torch.zeros(dim))
    self.beta = nn.Parameter(torch.zeros(dim))

    self.L_unconstrained = nn.Parameter(torch.eye(dim))

    self.log_alpha = nn.Parameter(torch.tensor(0.0))
    self.log_theta = nn.Parameter(torch.tensor(0.0))

  def _get_L(self):
    L = torch.tril(self.L_unconstrained)
    diag = torch.diagonal(L)
    L = L - torch.diag(diag) + torch.diag(torch.exp(diag))
    return L

  @property
  def _params(self):
    return {
        "mu": self.mu,
        "beta": self.beta,
        "alpha": torch.exp(self.log_alpha),
        "theta": torch.exp(self.log_theta),
        "L": self.L.detach(),
        "dof": 2*torch.exp(self.log_alpha)
    }

  def sample(self, shape):
    device = self.mu.device
    alpha = torch.exp(self.log_alpha)
    theta = torch.exp(self.log_theta)

    self.L = self._get_L()

    gamma = dist.InverseGamma(alpha, 1/theta)
    W = gamma.sample(shape).to(device)

    eps = torch.randn(*shape, self.dim, device=device)

    mu = self.mu.view(1, 1, -1)
    beta = self.beta.view(1, 1, -1)

    X = (
          self.mu
          + self.beta*W.unsqueeze(-1)
          + torch.sqrt(W).unsqueeze(-1)*(eps @ self.L.T)
        )

    return X

class CompositeLoss(nn.Module):
  def __init__(self):
    super().__init__()

    losses = ["rec", "smooth", "EVT", "conservative", "metric", "prior", "latent_match"]
    self.num_losses = len(losses)

    self.log_vars = nn.Parameter(torch.ones(self.num_losses) * 0.1)

  def rec_loss(self, y_pred, y_true):
    return func.mse_loss(y_pred.squeeze(-1), y_true.squeeze(-1))

  def smoothness_loss(self, z):
    return torch.mean((z[:, 1:] - z[:, :-1])**2)

  def evt_loss(self, xi_pred, xi_target):
    xi_pred = xi_pred.expand_as(xi_target)
    return func.mse_loss(xi_pred, xi_target)

  def metric_loss(self, vol_pred, vol_true):
    vol_true = vol_true.reshape(vol_true.shape[0], -1)
    dz = torch.norm(vol_pred[:, 1:] - vol_pred[:, :-1], dim=-1)
    dvol = torch.abs(vol_true[:, 1:] - vol_true[:, :-1])
    return func.mse_loss(dz, dvol)

  def conservative_loss(self, vol_pred, vol_target):
    under = torch.clamp(vol_target - vol_pred, min=0)
    return torch.mean(under**2)

  def latent_match_loss(self, z_sim, z_real):
    return func.mse_loss(z_sim.mean(dim=0), z_real.mean(dim=0))

  def prior_loss(self, z, mu, L):
    B, T, D = z.shape
    z_flat = z.reshape(-1, D)
    diff = (z_flat-mu).T
    solved = torch.linalg.solve_triangular(L, diff, upper=False)
    loss = torch.mean(torch.sum(solved**2, dim=0))
    return loss

  def forward(self, y_true, y_pred, z_real, nmvm_params, xi_target, z_sim):
    L = nmvm_params.get("L")
    mu = nmvm_params.get("mu")
    nu = nmvm_params.get("dof")

    xi_pred = 1/nu

    total_loss = 0
    losses = []

    losses.append(self.rec_loss(y_pred, y_true))
    losses.append(self.smoothness_loss(z_real))
    losses.append(self.evt_loss(xi_pred, xi_target))
    losses.append(self.metric_loss(z_real, y_true))
    losses.append(self.conservative_loss(y_pred, y_true))
    losses.append(self.prior_loss(z_real, mu, L))
    losses.append(self.latent_match_loss(z_sim, z_real))

    if any(i.isnan() for i in losses):
      print(losses)

    log_vars = torch.clamp(self.log_vars, -5, 5)
    weights = torch.softmax(log_vars, dim=0)
    total_loss = torch.sum(weights * torch.stack(losses))
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

  def sample(self, num_paths, num_steps, to_numpy=True):
    shape = (num_paths, num_steps)
    z_sim = self.nmvm.sample(shape)

    B, T, D = z_sim.shape

    z_flat = z_sim.reshape(-1, D)

    y_flat = self.decoder(z_flat)

    y_sim = y_flat.view(B, T)

    return y_sim.detach().cpu().numpy() if to_numpy else y_sim

  def fit(self):
    epochs = self.train_settings["epochs"]
    device = self.train_settings["device"]
    lr = self.train_settings["lr"]
    batch_size = self.train_settings["batch_size"]

    encoder = Encoder(self.encoder_params).to(device)
    self.decoder = Decoder(self.decoder_params).to(device)
    self.nmvm = NMVM(self.decoder_params["latent_dim"]).to(device)
    criterion = CompositeLoss().to(device)

    optimizer = torch.optim.Adam(
        (
            list(encoder.parameters())
            +list(self.decoder.parameters())
            +list(criterion.parameters())
            +list(self.nmvm.parameters())
        ),
        lr=lr
    )
    self.data = self.data.dropna()

    dataset = DataStore(self.data, feature_cols=self.data.columns)
    train_loader = DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=4,
      pin_memory=True
    )

    for epoch in range(epochs):
        epoch_loss = []

        for x, y, xi in tqdm(train_loader):

          x = x.to(device)
          y = y.to(device)
          xi = xi.to(device)

          optimizer.zero_grad()

          z_real = encoder(x)
          self.n = z_real.shape[0]
          y_pred = self.decoder(z_real)
          z_sim = self.nmvm.sample((self.n, ))
          nmvm_params = self.nmvm._params


          loss = criterion(
              y_true=y,
              y_pred=y_pred,
              xi_target=xi,
              nmvm_params=nmvm_params,
              z_real=z_real,
              z_sim=z_sim
          )

          loss.backward()
          optimizer.step()
          epoch_loss.append(loss.item())

        print(f"Epoch: {epoch+1}, Train Loss: {sum(epoch_loss)/len(epoch_loss)}")

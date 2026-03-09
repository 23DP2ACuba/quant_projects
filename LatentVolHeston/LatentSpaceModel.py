import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as func

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

      x = torch.tensor(x, dtype=torch.float32)
      y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

      return (x, y, xi_target)

class Encoder(nn.Module):
    def __init__(self, model_params: dict):
        super().__init__()


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
        seq_len, _ = x.shape

        x = self.fc_in(x)
        x = x + self.pos_embd[:, :seq_len, :]

        x = self.transformer_encoder(x)
        z_seq = x

        return z_seq

class Decoder(nn.Module):
  def __init__(self, model_params):
    super().__init__()
    d_model = model_params["d_model"]
    param_dim = model_params["latent_dim"]
    self.seq_len = model_params["seq_len"]

    self.log_alpha = nn.Parameter(torch.tensor(0.0))
    self.log_theta = nn.Parameter(torch.tensor(0.0))

    self.decoder = nn.Sequential(
        nn.Linear(d_model, 2*d_model),
        nn.GELU(),
        nn.LayerNorm(2*d_model),
        nn.Linear(2*d_model, d_model),
        nn.GELU(),
        nn.Linear(d_model, param_dim)
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

    losses = ["rec", "smooth", "EVT", "conservative", "metric"]
    self.num_losses = len(losses)

    self.log_vars = nn.Parameter(torch.zeros(self.num_losses))

  def rec_loss(self, y_pred, y_true):
    return func.mse_loss(y_pred.squeeze(-1), y_true.squeeze(-1))

  def smoothness_loss(self, z):
    return torch.mean((z[:, 1:] - z[:, :-1])**2)

  def evt_loss(self, xi_pred, xi_target):
    return func.mse_loss(xi_pred, xi_target)

  def metric_loss(self, z_seq, vol_seq):
    dz = torch.norm(z_seq[:, 1] - z_seq[:, :-1])
    dvol = torch.abs(vol_seq[:, 1:] - vol_seq[:, :-1])
    return func.mse_loss(dz, dvol)

  def conservative_loss(self, vol_pred, vol_target):
    under = torch.clamp(vol_target - vol_pred, min=0)
    return torch.mean(under**2)

  def prior_loss(self, z, mu, sigma):
    sigma_safe = torch.clamp(sigma, min=1e-6)
    loss = (z - mu)**2 / sigma_safe
    return torch.mean(loss)

  def forward(self, y_true, y_pred, z_sim, nmvm_params, xi_target):
    L = nmvm_params.get("L", 0)
    alpha = nmvm_params.get("alpha", 0) 
    xi_pred = L**2/(2*alpha) if alpha == 0 else torch.zeros_like(xi_target)
    total_loss = 0
    losses = []

    losses.append(self.rec_loss(y_pred, y_true[:, -1]))
    losses.append(self.smoothness_loss(z_sim))
    losses.append(self.evt_loss(xi_pred, xi_target))
    losses.append(self.metric_loss(z_sim, y_true))
    vol_pred = torch.norm(z_sim, dim=-1)
    losses.append(self.conservative_loss(vol_pred, y_true))

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
    print(device)
    lr = self.train_settings["lr"]
    batch_size = self.train_settings["batch_size"]

    encoder = Encoder(self.encoder_params)
    decoder = Decoder(self.decoder_params)
    nmvm = NMVM(self.decoder_params["latent_dim"])
    criterion = CompositeLoss()

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

    train_loader = DataStore(self.data, feature_cols=self.data.columns)
    epoch_loss = []

    for epoch in range(epochs):
        for x, y, xi in train_loader:
          x.to(device)
          y.to(device)
          xi.to(device)

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


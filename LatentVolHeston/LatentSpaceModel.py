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
      
      xi_target = np.abs(np.log(self.xi_target[idx + self.seq_len - 1]) - np.log(self.xi_target[idx + self.seq_len - 2]))
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

        y_pred = self.fc_out(x)

        return y_pred, z_seq

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
  
  def sample(self, nmvm_params):
    alpha = torch.exp(self.log_alpha)
    theta = torch.exp(self.log_theta)
    nmvm_params = nmvm_params.transpose(1, 2)
    mu = nmvm_params[:, 0]
    beta = nmvm_params[:, 1]
    sigma = torch.nn.functional.softplus(nmvm_params[:, 2]) + 1e-6

    gamma_dist = dist.Gamma(alpha, 1/theta)
    W = gamma_dist.sample((self.seq_len,))

    eps = torch.randn(self.seq_len)

    return mu + beta * W + sigma * torch.sqrt(W) * eps


  def forward(self, z_t):
    nmvm_params = self.decoder(z_t)
    X = self.sample(nmvm_params)
    
    return nmvm_params, X


class NMVMDistribution:
    def __init__(self, latent_dim, W_dist='gamma'):
        self.latent_dim = latent_dim
        self.mu = torch.zeros(latent_dim)
        self.beta = torch.zeros(latent_dim)
        self.Sigma = torch.eye(latent_dim)
        self.W_dist = W_dist

    def sample_W(self, n_samples):
        """
        Sample W from chosen mixing distribution.
        """
        if self.W_dist == 'gamma':
            shape, scale = 2.0, 1.0  # example parameters
            W = torch.distributions.Gamma(shape, scale).sample((n_samples,))
        elif self.W_dist == 'inverse_gaussian':
            # Placeholder: implement Inverse Gaussian
            W = torch.ones(n_samples)
        else:
            raise ValueError("Unsupported W distribution")

        return W

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

  def forward(self, y_true, y_pred, nmvm_params, xi_target, v_t):
    xi_pred = nmvm_params[:, 5]
    total_loss = 0
    losses = []

    losses.append(self.rec_loss(y_pred, y_true[:, -1]))
    losses.append(self.smoothness_loss(nmvm_params))
    losses.append(self.evt_loss(xi_pred, xi_target))
    losses.append(self.metric_loss(nmvm_params, y_true))
    losses.append(self.conservative_loss(v_t, y_true))

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
    pass


  def fit(self):
    epochs = self.train_settings["epochs"]
    device = self.train_settings["device"]
    lr = self.train_settings["lr"]
    batch_size = self.train_settings["batch_size"]

    encoder = Encoder(self.encoder_params)
    decoder = Decoder(self.decoder_params)
    criterion = CompositeLoss()

    optimizer = torch.optim.Adam(
        list(encoder.parameters())+list(decoder.parameters())+list(criterion.parameters()), 
        lr=lr
    )

    train_loader = DataStore(self.data, feature_cols=self.data.columns)

    for epoch in range(epochs):
        for x, y, xi_target_i in train_loader:
          x.to(device)
          y.to(device)
          xi_target_i.to(device)

          optimizer.zero_grad()

          y_pred, z = encoder(x)
          nmvm_params, v_t = decoder(z)

          loss = criterion(
              y_true=y,
              y_pred=y_pred,
              nmvm_params=nmvm_params,
              xi_target=xi_target_i,
              v_t=v_t
          )

          loss.backward()
          optimizer.step()

        print(f"Epoch: {epoch+1}, Train Loss: {sum(total_loss)/len(total_loss)}")
    return model

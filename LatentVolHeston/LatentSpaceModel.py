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
      y = self.target[idx + self.seq_len - 1]
      xi_target = np.abs(np.log(self.xi_target[idx + self.seq_len - 1]) - np.log(self.xi_target[idx + self.seq_len - 2]))

      x = torch.tensor(x, dtype=torch.float32)
      y = torch.tensor(y, dtype=torch.float32)

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

        self.fc_out = nn.Linear(model_params["d_model"], 1)

    def forward(self, x):
        seq_len, _ = x.shape
        print(x.shape)

        x = self.fc_in(x)
        x = x + self.pos_embd[:, :seq_len, :]

        x = self.transformer_encoder(x)

        z_t = x[:, -1, :]

        y_pred = self.fc_out(z_t)

        return y_pred, z_t

class Decoder(nn.Module):
  def __init__(self, model_params):
    super().__init__()
    d_model = model_params["d_model"]
    param_dim = model_params["latent_dim"]
    self.k_min = 2.0
    self.theta_min = 0.2
    self.decoder = nn.Sequential(
        nn.Linear(d_model, 2*d_model),
        nn.GELU(),
        nn.LayerNorm(2*d_model),
        nn.Linear(2*d_model, d_model),
        nn.GELU(),
        nn.Linear(d_model, param_dim)
    )

    self.k_layer = nn.Linear(d_model, 1)
    self.theta_layer = nn.Linear(d_model, 1)

  def forward(self, z_t):
    nmvm_params = self.decoder(z_t)
    k_pred = func.softplus(self.k_layer(z_t)) + self.k_min
    theta_pred = func.softplus(self.theta_layer(z_t)) + self.theta_min
    return nmvm_params,  k_pred, theta_pred


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

class CompositeLoss:
  def __init__(self, lambdas):
    self.l = lambdas

  def rec_loss(self, y_pred, y_true):
    return func.mse_loss(y_pred, y_true)

  def smoothness_loss(self, z):
    return torch.mean((z[:, 1:] - z[:, :-1])**2)

  def evt_loss(self, xi_pred, xi_target):
    return func.mse_loss(xi_pred, xi_target)

  def metric_loss(self, z, vol_target):
    dz = torch.norm(z[:, 1:] - z[:, :-1], dim=-1)
    dvol = torch.abs(vol_target[:, 1:] - vol_target[:, :-1])
    return func.mse_loss(dz, dvol)

  def conservative_loss(self, vol_pred, vol_target):
    under = torch.clamp(vol_target - vol_pred, min=0)
    return torch.mean(under**2)

  def directional_loss(self, z):
    B, T, d = z.shape
    z_flat = z.reshape(B*T, d)
    cov = torch.cov(z_flat.T)
    off_diag = cov - torch.diag(torch.diag(cov))
    return torch.mean(off_diag**2)

  def prior_loss(self, z, mu, sigma):
    return torch.mean((z-mu)**2 / sigma)


  def __call__(self, y_true, y_pred, z_t, nmvm_params, xi_target, W):
    mu = nmvm_params[:, 0]
    beta = nmvm_params[:, 1]
    sigma = nmvm_params[:, 2]
    alpha = nmvm_params[:, 3]
    delta = nmvm_params[:, 4]
    xi_pred = nmvm_params[:, 5]
    v_t = mu + beta*W + torch.sqrt(W)*np.sqrt(sigma) @ eps
    vol_target = torch.mean(y_true)
    eps = torch.randn_like(z_t)

    total_loss = 0
    losses = {
        "rec": self.rec_loss(y_pred, y_true),
        "smooth": self.smoothness_loss(z),
        "EVT": self.evt_loss(xi_pred, xi_target),
        "metric": self.metric_loss(z_t, vol_target),
        "conservative": self.conservative_loss(v_t, vol_target),
        "directional": self.directional_loss(z_t),
        "prior": self.prior_loss(z_t, mu, sigma)
    }

    for k in losses:
      total_loss += self.l[k] * losses[k]

    return total_loss, losses

class LatentSpaceModel(LatentSpaceVol):
  def __init__(self, feature_params, model_params, lambdas):
    super().__init__(
      feature_params=feature_params
    )
    self.encoder_params = model_params["model_params"]
    self.decoder_params = model_params["model_params"]
    self.lambdas = list(lambdas.values())
    self.lambda_params = torch.nn.Parameter(
      torch.tensor(self.lambdas, dtype=torch.float32)
    )
    self.train_settings = model_params["train_settings"]
    self.load_or_generate_features()

  def load_or_generate_features(self):
    filename = f"features_{self.params.ticker}.parquet"
    if os.path.exists(filename):
      self.data = pd.read_parquet(filename)
    else:
      self.get_price_data()
      self.generate_features()
      self.data.to_parquet(filename)


  def fit(self):
    epochs = self.train_settings["epochs"]
    device = self.train_settings["device"]
    lr = self.train_settings["lr"]
    batch_size = self.train_settings["batch_size"]

    encoder = Encoder(self.encoder_params)
    decoder = Decoder(self.decoder_params)
    criterion = CompositeLoss(self.lambdas)
    optimizer_model = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=lr)
    optimizer_lambda = torch.optim.Adam([self.lambda_params], lr=lr)
    print(self.data)
    train_loader = DataStore(self.data, feature_cols=self.data.columns)

    for epoch in range(epochs):
        for xi, yi, xi_target_i in train_loader:
          print(xi.shape)
          xi.to(device)
          yi.to(device)

          y, z = encoder(xi)
          nmvm_params, k, theta = decoder(z)
          optimizer.zero_grad()
          W = k*theta

          loss = criterion(
              y_true=y,
              y_pred=yi,
              z_t=z_t,
              nmvm_params=nmvm_params,
              xi_target=xi_target_i,
              W=W
          )
          loss.backward(retain_graph=True)
          optimizer.step()

          optimizer_lambda.zero_grad()
          loss.backward()
          optimizer_lambda.step()

          criterion.l = lambda_params.detach().numpy()

        print(f"Epoch: {epoch+1}, Train Loss: {sum(total_loss)/len(total_loss)}")
    return model

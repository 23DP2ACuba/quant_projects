import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as func

class DataStore(Dataset):
    def __init__(self, data, feature_cols, target_col="RealizedVol", seq_len=60):

        self.seq_len = seq_len

        self.features = data[feature_cols].values

        self.target = data[target_col].shift(-1).values

        self.features = self.features[:-1]
        self.target = self.target[:-1]

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.seq_len]
        y = self.target[idx + self.seq_len - 1]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y

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
        batch_size, seq_len, _ = x.shape

        x = self.fc_in(x)
        x = x + self.pos_embd[:, :seq_len, :]

        x = self.transformer_encoder(x)

        z_t = x[:, -1, :]

        y_pred = self.fc_out(z_t)

        return y_pred, z_t

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        
    def forward(self, x):
        pass



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


  def __call__(self, inputs):
    y_true, y_pred, z, vol_pred, vol_target, xi_pred, xi_target, mu, sigma = inputs
    total_loss = 0
    losses = {
        "rec": self.rec_loss(y_pred, y_true),
        "smooth": self.smoothness_loss(z),
        "EVT": self.evt_loss(xi_pred, xi_target),
        "metric": self.metric_loss(z, vol_target),
        "conservative": self.conservative_loss(vol_pred, vol_target),
        "directional": self.directional_loss(z),
        "prior": self.prior_loss(z, mu, sigma)
    }

    for k in losses:
      total_loss += self.l[k] * losses[k]

    return total_loss, losses

class LatentSpaceModel(LatentSpaceVol):
  def __init__(self, feature_params, model_params, lambdas):
    super().__init__(
        feature_params=feature_params
      )
    self.encoder_params = model_params["encoder_params"]
    self.lambdas = lambdas
    self.train_settings = model_params["train_settings"]

  def generate_data(self):
    self.get_price_data()
    self.generate_features()
    print(self.data.head())

  def train(self):
    model = Encoder(self.params)
    criterion = CompositeLoss(self.lambdas)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader()
    for epoch in range(epochs):
        model.train()
        total_loss=[]
        for xi, yi in train_loader:
          xi.to(device)
          yi.to(device)

          optimizer.zero_grad()

          out = model(xi)
          loss = criterion(out, yi)
          loss.backward()
          optimizer.step()
          total_loss.append(loss.item())

        print(f"Epoch: {epoch+1}, Train Loss: {sum(total_loss)/len(total_loss)}")
        if val_loader:
          eval_loss=[]
          model.eval()
          with torch.no_grad():
            for xi, yi in val_loader:
              xi.to(device)
              yi.to(device)

              out = model(xi)
              loss = criterion(out, yi)
              train_losses.append(loss.item())
            print(f"Eval Loss: {sum(total_loss)/len(total_loss)}")
    return model
    


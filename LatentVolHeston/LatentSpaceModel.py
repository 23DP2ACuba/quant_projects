import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

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
    def __init__(self, params: dict):
        super().__init__()

        self.fc_in = nn.Linear(params["input_dim"], params["d_model"])

        self.pos_embd = nn.Parameter(
            torch.randn(1, params["seq_len"], params["d_model"])
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=params["d_model"],
            nhead=params["nhead"],
            dim_feedforward=4*params["d_model"],
            dropout=params["dropout"],
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=params["num_layers"]
        )

        self.fc_out = nn.Linear(params["d_model"], 1)

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

class LatentSpaceModel(DataStore, Encoder, Decoder):
  def __init__(self, data, params):
    Encoder.__init__(self, params["encoder_params"])
    # Decoder.__init__(self, params["decoder_params"])
    self.losses = ["rec", "smooth", "EVT", "conserv", "metric", "directional", "prior"]
    self.data = data

    self.lams = [1] + [0.01]*(len(self.losses)-1)

  def reconstruction_loss(self, y_pred, y_true):
    return torch.mean((y_pred - y_true)**2)

  def smoothness_loss(self):
    return np.sum((self.z_t[1:] - self.z_t[:-1])**2)

  def velocity_loss(self):
    return np.sum((self.z_t.shift(-1) - self.z_t)-(self.z_t - self.z_t.shift(1))**2)

  def conservative_loss(self):
    pass

  def composite_loss(y_true, y_pred, z_t):
    self.z_t = z_t
    self.y_pred, self.y_true = y_pred, y_true
    eps = 1e-5
    losses = 0
    loss_funcs = [self.reconstruction_loss, self.velocity_loss, self.conservative_loss]
    for i in range(len(self.losses)):
      losses += self.lams[i] * loss_funcs[i]()

    return losses


  def train(self):
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


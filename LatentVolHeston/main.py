import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -----------------------------
# Step 1: Transformer Encoder
# -----------------------------
class Encoder(nn.Module):
    """
    Transformer-based encoder for extracting latent volatility states from features.
    Input: scaled features X_t of shape (batch, seq_len, num_features)
    Output: latent vector z_t of shape (batch, latent_dim)
    """
    def __init__(self, num_features, latent_dim, n_heads=4, n_layers=2, dropout=0.1):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_proj = nn.Linear(num_features, latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(latent_dim, latent_dim)  # final latent vector

    def forward(self, x):
        """
        x: (batch, seq_len, num_features)
        """
        x = self.input_proj(x)
        x = self.transformer(x)
        # Take last time step as representation
        z = self.output_proj(x[:, -1, :])
        return z


# -----------------------------
# Step 2: NMVM Latent Distribution
# -----------------------------
class NMVMDistribution:
    """
    Normal Mean-Variance Mixture distribution for latent space.
    z = mu + beta * W + sqrt(W) * Sigma^(1/2) * epsilon
    """
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

    def sample(self, n_samples):
        """
        Sample latent vectors from NMVM.
        """
        W = self.sample_W(n_samples).unsqueeze(-1)  # (n_samples,1)
        epsilon = torch.randn(n_samples, self.latent_dim)
        # Cholesky decomposition of Sigma
        L = torch.linalg.cholesky(self.Sigma)
        z = self.mu + self.beta * W + torch.sqrt(W) * (epsilon @ L.T)
        return z


# -----------------------------
# Step 3: Decoder to Volatility
# -----------------------------
class Decoder(nn.Module):
    """
    Maps latent vector z_t to volatility v_t.
    """
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),  # scalar volatility
            nn.Softplus()  # ensure positivity
        )

    def forward(self, z):
        v = self.fc(z)
        return v.squeeze(-1)


# -----------------------------
# Step 4: Latent NMVM Heston Simulator
# -----------------------------
class LatentVolHeston:
    """
    Modified Heston model using latent volatility sampled from NMVM.
    """
    def __init__(self, encoder, decoder, nmvm_dist, mu=0.0, dt=1/252):
        self.encoder = encoder
        self.decoder = decoder
        self.nmvm_dist = nmvm_dist
        self.mu = mu  # drift of S_t
        self.dt = dt  # time step

    def simulate(self, X_seq, S0=100.0, n_steps=252):
        """
        Simulate asset path using latent-volatility-enhanced Heston model.
        X_seq: input feature sequence for encoder
        S0: initial asset price
        n_steps: number of simulation steps
        """
        device = next(self.encoder.parameters()).device
        S = torch.zeros(n_steps)
        S[0] = S0
        with torch.no_grad():
            z_t = self.encoder(X_seq.unsqueeze(0))  # (1, latent_dim)
            for t in range(1, n_steps):
                # Sample latent vector from NMVM
                z_sample = self.nmvm_dist.sample(1)
                # Map to volatility
                v_t = self.decoder(z_sample)
                # Euler-Maruyama step
                dW = torch.randn(1) * np.sqrt(self.dt)
                S[t] = S[t-1] + self.mu*S[t-1]*self.dt + torch.sqrt(v_t)*S[t-1]*dW
        return S.numpy()

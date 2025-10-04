import math
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------- Config ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SYMBOL = "BTC-USD"
START = "2019-01-01"
END = "2025-08-01"
PERIOD = "1d"
WINDOW_SIZE = 20
LOOKBACK = 8
MA_PERIOD = 20
N_DAYS = 5
BATCH_SIZE = 32
DROPOUT = 0.4
HMM_COMPONENTS = 1
EPOCHS = 15
LR = 3e-4
RSI_WINDOW = 14
USE_PCA = False
TEST_SIZE = 0.2 
# ---------------------------------------

from scipy.stats import zscore

def train_hmm_on_data(X_train_for_hmm, n_components=HMM_COMPONENTS, use_pca=False):
    """
    Train HMM on training rows only. Return fitted hmm_model and scaler (fitted on train).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_for_hmm.reshape(-1, 1) if X_train_for_hmm.ndim == 1 else X_train_for_hmm)

    if use_pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        X_scaled = pca.fit_transform(X_scaled)
    else:
        pca = None

    hmm_model = GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=400,
        tol=1e-4,
        verbose=False,
        init_params="stmc"
    )
    hmm_model.fit(X_scaled)

    return hmm_model, scaler, pca

def apply_hmm_to_full_df(df, hmm_model, scaler, pca=None, feature_col='Log_Return'):
    X_all = df[[feature_col]].values if isinstance(feature_col, str) else df[feature_col].values
    X_all_scaled = scaler.transform(X_all.reshape(-1, 1) if X_all.ndim == 1 else X_all)
    if pca is not None:
        X_all_scaled = pca.transform(X_all_scaled)

    regime_probs = hmm_model.predict_proba(X_all_scaled)
    most_likely = np.argmax(regime_probs, axis=1)
    confidence = regime_probs[np.arange(len(most_likely)), most_likely]

    df = df.copy()
    df.loc[:, "Most_Likely_Regime"] = most_likely
    df.loc[:, "Regime_Prob"] = confidence
    return df

def create_features_no_hmm(raw_df):
    df = raw_df.copy()
    df['Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    df['Volatility'] = df['Return'].rolling(5, min_periods=1).std().shift(1)
    df['Momentum'] = df['Close'].shift(1) - df['Close'].shift(6)

    df['Log_Volume'] = np.log(df['Volume'].shift(1).clip(lower=1e-6))
    df["Ma"] = df['Close'].rolling(MA_PERIOD).mean().shift(1)
    df['Cl_to_Ma_pct'] = (df['Close'] - df['Ma']) / df['Close'] * 100

    df["Z-Score"] = (df['Return'] - df['Return'].rolling(WINDOW_SIZE).mean().shift(1)) / (df['Return'].rolling(WINDOW_SIZE).std().shift(1) + 1e-6)
    df["Delta"] = df['Close'].diff().shift(1)

    df = df.dropna().copy()
    return df

def create_features_with_target_and_hmm(raw_df, hmm_model=None, scaler=None, pca=None):

    df = create_features_no_hmm(raw_df)

    if hmm_model is not None and scaler is not None:
        df = apply_hmm_to_full_df(df, hmm_model, scaler, pca=pca, feature_col='Log_Return')
    else:
        df['Most_Likely_Regime'] = np.nan
        df['Regime_Prob'] = np.nan

    df['Target'] = np.log(df['Close'] / df['Close'].shift(-N_DAYS))*100
    df = df.dropna().copy() 
    return df

# ---------------- Download raw data ----------------
raw_data = yf.Ticker(SYMBOL).history(interval=PERIOD, start=START, end=END)[["Open", "High", "Low", "Close", "Volume"]]
raw_data = raw_data.dropna().copy()

# -------------- Build deterministic features first (no HMM) --------------
features_df = create_features_no_hmm(raw_data)

# -------------- Train/test split (time-ordered, no shuffle) --------------
n = len(features_df)
train_size = int((1 - TEST_SIZE) * n)
train_indices = list(range(0, train_size))
test_indices = list(range(train_size, n))

# -------------- Train HMM on training portion only --------------
hmm_train_X = features_df.iloc[train_indices]['Log_Return'].values.reshape(-1, 1)
hmm_model, hmm_scaler, hmm_pca = train_hmm_on_data(hmm_train_X, n_components=HMM_COMPONENTS, use_pca=USE_PCA)

# -------------- Build final dataset with HMM applied and targets --------------
data = create_features_with_target_and_hmm(raw_data, hmm_model=hmm_model, scaler=hmm_scaler, pca=hmm_pca)

common_index = features_df.index.intersection(data.index)

n_common = len(common_index)
train_size_common = int((1 - TEST_SIZE) * n_common)
train_index_common = common_index[:train_size_common]
test_index_common = common_index[train_size_common:]

features = data.drop(columns=["Target"])
targets = data["Target"]

if features.isna().any().any() or targets.isna().any():
    raise ValueError("NaN values detected in features or targets after final construction - check shifting/rolling logic.")

if np.isinf(features.to_numpy()).any() or np.isinf(targets.to_numpy()).any():
    raise ValueError("Infinite values detected in features or targets after final construction.")

X_train = features.loc[train_index_common]
X_test = features.loc[test_index_common]
y_train = targets.loc[train_index_common]
y_test = targets.loc[test_index_common]

y_mean, y_std = y_train.mean(), y_train.std()
y_train = (y_train - y_mean) / y_std
y_test = (y_test - y_mean) / y_std

# -------------- Scale features (fit scaler on training only) --------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------- Create DataLoaders --------------
train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                              torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32),
                             torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------- Model definition (same VAE-like encoder) --------------
class Encoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20, out_dim=1):
        super().__init__()
        self.data_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2log_sigma = nn.Linear(h_dim, z_dim)  
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2out = nn.Linear(h_dim, out_dim)
        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.data_2hid(x))
        mu = self.hid_2mu(h)
        log_sigma = self.hid_2log_sigma(h)
        return mu, log_sigma

    def regressor(self, z):
        h = self.relu(self.z_2hid(z))
        return self.hid_2out(h) 

    def forward(self, x):
        mu, log_sigma = self.encode(x)
        sigma = torch.exp(log_sigma)
        epsilon = torch.randn_like(sigma)
        z_new = mu + sigma * epsilon
        x_pred = self.regressor(z_new)
        return x_pred, mu, log_sigma

INPUT_DIM = X_train.shape[1]
model = Encoder(INPUT_DIM, h_dim=200, z_dim=20, out_dim=1).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------------- Training loop (unchanged but with consistent shapes and early stopping) --------------
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs, device):
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_mse = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs, mu, log_sigma = model(batch_X)

            loss = criterion(outputs, batch_y)
            kl_div = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - torch.exp(log_sigma)) / batch_X.size(0)
            total_loss = loss + 0.1 * kl_div

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item() * batch_X.size(0)
            train_mse += loss.item() * batch_X.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        train_mse = train_mse / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_mse = 0.0

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs, mu, log_sigma = model(batch_X)

                loss = criterion(outputs, batch_y)
                kl_div = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - torch.exp(log_sigma)) / batch_X.size(0)
                total_loss = loss + 0.1 * kl_div

                val_loss += total_loss.item() * batch_X.size(0)
                val_mse += loss.item() * batch_X.size(0)

        val_loss = val_loss / len(test_loader.dataset)
        val_mse = val_mse / len(test_loader.dataset)

        print(f'Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.6f}, Train MSE: {train_mse:.6f} | Val Loss: {val_loss:.6f}, Val MSE: {val_mse:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

train_model(model, train_loader, test_loader, criterion, optimizer, EPOCHS, DEVICE)

def test_model(model, X_test_scaled, y_test, device, checkpoint_path="best_model.pt", n_show=20):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), 512): 
            batch = X_tensor[i:i+512]
            out, _, _ = model(batch)
            preds.append(out.cpu().numpy())
    preds = np.vstack(preds).flatten()

    actuals = y_test.values if hasattr(y_test, "values") else y_test
    preds_sign = np.sign(preds)
    actuals_sign = np.sign(actuals)

    preds_sign[preds_sign == 0] = 1
    actuals_sign[actuals_sign == 0] = 1

    direction_acc = np.mean(preds_sign == actuals_sign)
    print(f"Directional Accuracy on Test Set: {direction_acc:.2%}")

    compare_df = pd.DataFrame({
        "Actual": actuals[:n_show],
        "Predicted": preds[:n_show],
        "Actual_Sign": actuals_sign[:n_show],
        "Pred_Sign": preds_sign[:n_show]
    })
    print("\nSample Predictions vs Actuals:")
    print(compare_df.to_string(index=False))
    
    return preds, actuals, direction_acc
preds, actuals, dir_acc = test_model(model, X_test_scaled, y_test, DEVICE)

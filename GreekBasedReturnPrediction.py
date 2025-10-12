import numpy as np
import scipy.stats as si
import yfinance as yf
from datetime import datetime
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ----------------------------
# Config
# ----------------------------
ticker = "TSLA"
market_ticker = "^GSPC"
vix_ticker = "^VIX"
start = "2019-01-01"
end = "2025-01-01"

config = {
    'window': 20,
    'w_norm': 63,
    'w_vol': 21,
    'w_beta': 60,
    'beta_span': 10,
    'lags': {
        "Norm_Returns": 3,
        "Sigma_hat": 3,
        "Beta_vol": 3,
        "market_returns": 3,
        "Beta_simple": 2,
        "Z-Score_i": 2
    }
}

# ----------------------------
# Download data
# ----------------------------

def download(ticker, start, end=datetime.now()):
  return yf.Ticker(ticker).history(start=start, end=end)[["Close"]]

data = download(ticker=ticker, start=start, end=end)
market = download(ticker=market_ticker, start=start, end=end)
#market = download(ticker=vix_ticker, start=start, end=end)

# ----------------------------
# Feature helpers
# ----------------------------

def get_estimation(x, y, window):
    """Compute rolling OLS estimates for beta, gamma, vanna."""
    x = x.dropna()
    betas, gammas, vannas, index = [], [], [], []

    for i in range(window, len(y)):
        x_window = x.iloc[i-window:i]
        y_window = y.iloc[i-window:i]

        model = sm.OLS(y_window, sm.add_constant(x_window)).fit()
        betas.append(model.params['mkt'])
        gammas.append(2 * model.params['mkt_sq'])
        vannas.append(model.params['mkt_interaction'])
        index.append(y.index[i-1])

    return (
        pd.Series(betas, index=index),
        pd.Series(gammas, index=index),
        pd.Series(vannas, index=index)
    )

def get_z_score(market_returns, w_norm):
    return (market_returns - market_returns.rolling(w_norm).mean()) / market_returns.rolling(w_norm).std()

def get_sigma_hat(r_s, window):
    return r_s.ewm(span=window).std() * np.sqrt(252)

def add_lags(df, lags_dict):
    df = df.copy()
    for feat, n_lags in lags_dict.items():
        for i in range(1, n_lags + 1):
            df[f"{feat}_{i}"] = df[feat].shift(i)
    return df.dropna()

def get_beta_gamma_vanna(market_returns, log_returns, sigma, window):
    sigma_diff = sigma.diff().fillna(0)
    X = pd.DataFrame({
        'mkt': market_returns,
        'mkt_sq': market_returns**2,
        'mkt_interaction': market_returns * sigma_diff
    })
    y = log_returns
    return get_estimation(window=window, x=X, y=y)

# ----------------------------
# Feature computation
# ----------------------------
def compute_features(df, market_returns, config):
    log_returns = np.log(df["Close"] / df["Close"].shift())
    df = pd.DataFrame({"log_returns": log_returns, "market_returns": market_returns}).dropna()

    df["Target"] = df["log_returns"].rolling(5).mean().shift(-1)

    sigma_hat = get_sigma_hat(df["log_returns"], window=config['w_vol'])
    z_score = get_z_score(df["market_returns"], w_norm=config['w_norm'])

    beta_hat, gamma, vanna = get_beta_gamma_vanna(
        df["market_returns"], df["log_returns"], sigma_hat, window=config['w_beta']
    )

    beta_simple = df["log_returns"].rolling(config['window']).cov(df["market_returns"]) / \
                  df["market_returns"].rolling(config['window']).var()

    df["Beta_simple"] = beta_simple
    df["Beta_hat_ewma"] = beta_hat.ewm(span=config['beta_span']).mean()
    df["Beta_vol"] = beta_hat.rolling(window=config['w_beta']).std()
    df["Gamma_i"] = gamma
    df["Vanna_i"] = vanna
    df["Charm"] = beta_hat.diff()
    df["Z-Score_i"] = z_score
    df["Sigma_hat"] = sigma_hat
    df["Epsilon"] = df["log_returns"] - beta_hat * df["market_returns"]
    df["Norm_Returns"] = df["log_returns"] / sigma_hat

    df = add_lags(df, lags_dict=config['lags'])
    df = df.drop(["log_returns"], axis=1)
    df = df.dropna()
    return df

# ----------------------------
# Preprocessing
# ----------------------------
def preprocessing(df, test_size):
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()

    X_train = train_df.drop(columns=["Target"])
    y_train = train_df["Target"]
    X_test  = test_df.drop(columns=["Target"])
    y_test  = test_df["Target"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

# ----------------------------
# Regression metrics
# ----------------------------
def regression_metrics(y_true, y_pred, dataset_name=""):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    direction_acc = np.mean(np.sign(y_pred) == np.sign(y_true))
    print(f"{dataset_name} Metrics:")
    print(f"  R²: {r2:.4f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Directional Accuracy: {direction_acc:.4f}\n")
    return r2, rmse, mae, direction_acc

# ----------------------------
# Compute returns and features
# ----------------------------
log_returns = np.log(data["Close"] / data["Close"].shift())
market_returns = np.log(market["Close"] / market["Close"].shift())
df = pd.DataFrame({"log_returns": log_returns, "market_returns": market_returns}).dropna()

z_score = get_z_score(market_returns=df["market_returns"], w_norm=w_norm)

sigma_hat = get_sigma_hat(r_s=df["log_returns"], window=w_vol)

beta_hat, gamma, vanna = get_beta_gamma_vanna(df["market_returns"], df["log_returns"], sigma_hat, window=w_beta)

beta_simple = df["log_returns"].rolling(window).cov(df["market_returns"]) / df["market_returns"].rolling(window).var()

# ----------------------------
# Assemble final features
# ----------------------------
df_features = compute_features(data, market_returns=np.log(market["Close"] / market["Close"].shift()), config=config)

x_train, x_test, y_train, y_test = preprocessing(df_features, test_size=0.2)

model = XGBRegressor(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1
)
model.fit(x_train, y_train)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

train_metrics = regression_metrics(y_train, y_train_pred, dataset_name="Train")
test_metrics  = regression_metrics(y_test, y_test_pred, dataset_name="Test")

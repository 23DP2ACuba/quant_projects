import numpy as np
import scipy.stats as si
import yfinance as yf
from datetime import datetime
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
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

test_size = 0.2
window = 20
w_norm = 63
w_vol = 21
w_beta = 60
beta_span = 10
n = 3
lags = {
    "Norm_Returns": 3,
    "Sigma_hat": 3,
    "Beta_vol": 3,
    "market_returns": 3,
    "Beta_simple": 2,
    "Z-Score_i": 2

}

# ----------------------------
# Download data
# ----------------------------

def download(ticker, start, end=datetime.now()):
  return yf.Ticker(ticker).history(start=start, end=end)[["Close"]]

data = download(ticker=ticker, start=start, end=end)
market = download(ticker=market_ticker, start=start, end=end)
vix = download(ticker=vix_ticker, start=start, end=end)

# ----------------------------
# Feature helpers
# ----------------------------

def get_estimation(x, y, window):
  x = x.dropna()
  betas = []
  gammas = []
  vannas = []
  index = []
 
  for i in range(window, len(y)):
      x_window = x.iloc[i-window:i]
      y_window = y.iloc[i-window:i]

      model = sm.OLS(y_window, sm.add_constant(x_window)).fit()

      betas.append(model.params['mkt'])
      gammas.append(2 * model.params['mkt_sq'])
      vannas.append(model.params['mkt_interaction'])

      index.append(y.index[i-1])

  store = []

  for greek in [betas, gammas, vannas]:
    store.append(pd.Series(greek, index=index))

  return store


def get_z_score(market_returns, w_norm):
    return ((market_returns 
            - market_returns.rolling(w_norm).mean()) 
            / market_returns.rolling(w_norm).std())


def get_sigma_hat(window, r_s):
  return r_s.ewm(span=window).std() * np.sqrt(252)

def add_lags(df, lags_dict):
  df = df.copy()
  for feat, vals in lags_dict.items():
    for i in range(1, vals+1):
      df[f"{feat}_{i}"] = df[f"{feat}"].shift(i)

  return df.dropna()

def get_beta_gamma_vanna(market_returns, log_returns, sigma, window):
  sigma_diff = sigma.diff().fillna(0)

  X = pd.DataFrame({
    'mkt': market_returns,
    'mkt_sq': market_returns**2,
    'mkt_interaction': market_returns*sigma_diff
  })
  y = log_returns

  return get_estimation(window=window, x=X, y=y)

def preprocessing(df, test_size):
  shuffle=False
  assert shuffle == False

  x = df[df.columns.difference(["Target"])]
  assert "Target" not in x.columns

  y = df["Target"]
  
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=shuffle)

  scaler = StandardScaler()
  x_train = scaler.fit_transform(x_train)
  x_test = scaler.fit_transform(x_test)

  return x_train, x_test, y_train, y_test

  
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
df["Target"] = df["log_returns"].shift(-n)

df["Beta_simple"] = beta_simple
df["Beta_hat_ewma"] = beta_hat.ewm(span=beta_span).mean()
df["Beta_vol"] = beta_hat.rolling(window=w_beta).std()
df["Gamma_i"] = gamma
df["Vanna_i"] = vanna
df["Charm"] = beta_hat.diff()
df["Z-Score_i"] = z_score
df["Sigma_hat"] = sigma_hat
df["Epsilon"] = df["log_returns"] - beta_hat * df["market_returns"]
df["Norm_Returns"] = df["log_returns"] / sigma_hat

df = add_lags(df=df, lags_dict=lags)

df = df.drop(["log_returns"], axis=1)


x_train, x_test, y_train, y_test = preprocessing(df=df, test_size=test_size)


model = RandomForestRegressor()
model.fit(x_train, y_train)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

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

train_metrics = regression_metrics(y_train, y_train_pred, dataset_name="Train")
test_metrics  = regression_metrics(y_test, y_test_pred, dataset_name="Test")

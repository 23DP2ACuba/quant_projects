import numpy as np
import scipy.stats as si
from statsmodels.tsa.seasonal import STL
import yfinance as yf
from datetime import datetime
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score, 
                             mean_squared_error, 
                             mean_absolute_error)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy


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
        "Z-Score_i": 2,
        "Trend": 3,
        "Seasonal": 3,
        "Residuals": 3,
    }
}

model_config = {
    "n_estimators": 500,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1
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

def get_decomposition_result(data, period=21, robust=True):
  data = data.copy()
  stl = STL(data, period=period, robust=robust)
  res = stl.fit()

  return res.trend, res.seasonal, res.resid

def get_z_score(returns, w_norm):
    return (returns - returns.rolling(w_norm).mean()) / \
        returns.rolling(w_norm).std()

def get_sigma_hat(r_s, window):
    return r_s.ewm(span=window).std() * np.sqrt(252)

def add_lags(df, lags_dict):
    df = df.copy()

    for feat, n_lags in lags_dict.items():
        for i in range(1, n_lags + 1):
            df[f"{feat}_{i}"] = df[feat].shift(i)
            
    return df.dropna()

def get_beta_gamma_vanna(returns, log_returns, sigma, window):
    sigma_diff = sigma.diff().fillna(0)
    X = pd.DataFrame({
        'mkt': returns,
        'mkt_sq': returns**2,
        'mkt_interaction': returns * sigma_diff
    })
    y = log_returns

    return get_estimation(window=window, x=X, y=y)

# ----------------------------
# Feature computation
# ----------------------------
def compute_features(df, config, target = "log_ret"):
  sigma_hat = get_sigma_hat(df["log_returns"], window=config['w_vol'])
  z_score = get_z_score(df["market_returns"], w_norm=config['w_norm'])

  beta_hat, gamma, vanna = get_beta_gamma_vanna(
      df["market_returns"], df["log_returns"], 
      sigma_hat, window=config['w_beta']
  )

  beta_simple = df["log_returns"].rolling(config['window'])\
    .cov(df["market_returns"]) / df["market_returns"]\
        .rolling(config['window']).var()

  (dec_trend, dec_seasonal, dec_residuals) = get_decomposition_result(
        data, 
        period=21, 
        robust=True
      )

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
  df["Trend"] = dec_trend
  df["Seasonal"] = dec_seasonal
  df["Residuals"] = dec_residuals

  if target == "alpha_component":
    future_returns = np.log(data["Close"].shift(-1) / data["Close"])
    df["Target"] = future_returns - df["Beta_hat_ewma"] * \
        df["market_returns"].shift(-1)
  else:
    df["Target"] = df["log_returns"].rolling(5).mean().shift(-1)

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
# Compute returns
# ----------------------------
window = config.get("window")
log_returns = np.log(data["Close"] / data["Close"].shift())
market_returns = np.log(market["Close"] / market["Close"].shift())
df = pd.DataFrame({"log_returns": log_returns, 
                   "market_returns": market_returns}).dropna()

# ----------------------------
# Assemble final features
# ----------------------------
df_features = compute_features(df, config=config)

x_train, x_test, y_train, y_test = preprocessing(df_features, test_size=0.2)  

print(df_features.columns)

model = XGBRegressor(
    n_estimators=model_config["n_estimators"],
    max_depth=model_config["max_depth"],
    learning_rate=model_config["learning_rate"],
    subsample=model_config["subsample"],
    colsample_bytree=model_config["colsample_bytree"],
    reg_alpha=model_config["reg_alpha"],
    reg_lambda=model_config["reg_lambda"]
)
model.fit(x_train, y_train)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

train_metrics = regression_metrics(y_train, y_train_pred, 
                                   dataset_name="Train")
test_metrics  = regression_metrics(y_test, y_test_pred, 
                                   dataset_name="Test")

def walk_forward_plot(model, X_train, y_train, X_test, y_test, 
                      retrain=False, scaler=None, 
                      title="Walk-Forward Prediction"):

    y_pred_walk = []
    y_true_walk = []
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    test_start_idx = len(X_train)
    n_test = len(X_test)

    model_ = copy.deepcopy(model)

    for i in range(n_test):
        x_next = X_test[i].reshape(1, -1)
        y_pred = model_.predict(x_next)[0]
        y_pred_walk.append(y_pred)
        y_true_walk.append(y_test.iloc[i] if hasattr(y_test, "iloc") else y_test[i])

        if retrain:
            X_new_train = X_all[: test_start_idx + i + 1]
            y_new_train = y_all[: test_start_idx + i + 1]
            if scaler:
                X_new_train = scaler.fit_transform(X_new_train)
            model_.fit(X_new_train, y_new_train)

    df_res = pd.DataFrame({
        "True": y_true_walk,
        "Pred": y_pred_walk
    })

    correlation = df_res["True"].corr(df_res["Pred"])
    covariance = df_res["True"].cov(df_res["Pred"])

    plt.figure(figsize=(10, 4))
    plt.plot(df_res["True"].values, label="True", lw=2)
    plt.plot(df_res["Pred"].values, label="Predicted", lw=2, linestyle="--")
    plt.title(f"{title}\nCorr={correlation:.3f} | Cov={covariance:.6f}")
    plt.xlabel("Step")
    plt.ylabel("Target value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Correlation (Pred vs True): {correlation:.4f}")
    print(f"Covariance (Pred vs True): {covariance:.6f}")

    metrics = {"correlation": correlation, "covariance": covariance}

    return np.array(y_pred_walk), metrics


y_walk_preds, metrics = walk_forward_plot(
    model=model,
    X_train=x_train,
    y_train=y_train,
    X_test=x_test,
    y_test=y_test,
    retrain=False,
    title="Walk-Forward Prediction (XGB)"
)

# ----------------------------
# Cash allocation simulation
# ----------------------------

def simulate_cash_allocation(initial_cash, model, X_train, 
                             y_train, X_test, y_test, retrain=False, 
                             scaler=None, allocation_factor=1.0):
    """
    Simulate trading cash allocation over the test period based on predicted returns.
    - initial_cash: starting portfolio value
    - allocation_factor: multiplier that determines position size sensitivity to predicted return
    """

    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    test_start_idx = len(X_train)
    n_test = len(X_test)
    model_ = copy.deepcopy(model)

    portfolio = [initial_cash]
    cash = initial_cash

    for i in range(n_test):
        x_next = X_test[i].reshape(1, -1)
        y_pred = model_.predict(x_next)[0]
        y_true = y_test.iloc[i] if hasattr(y_test, "iloc") else y_test[i]


        position = np.tanh(allocation_factor * y_pred)
        cash = cash * (1 + position * y_true)
        portfolio.append(cash)

        if retrain:
            X_new_train = X_all[: test_start_idx + i + 1]
            y_new_train = y_all[: test_start_idx + i + 1]
            if scaler:
                X_new_train = scaler.fit_transform(X_new_train)
            model_.fit(X_new_train, y_new_train)

    df_portfolio = pd.DataFrame({
        "Step": np.arange(n_test + 1),
        "PortfolioValue": portfolio
    })
    df_portfolio["Return"] = df_portfolio["PortfolioValue"].pct_change()

    plt.figure(figsize=(10, 4))
    plt.plot(df_portfolio["PortfolioValue"], lw=2, label="Simulated Portfolio Value")
    plt.title(f"Simulated Cash Allocation | Final Value = {portfolio[-1]:.2f}")
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    total_return = (portfolio[-1] / initial_cash) - 1
    avg_daily_return = df_portfolio["Return"].mean()
    volatility = df_portfolio["Return"].std()
    sharpe_ratio = (avg_daily_return / volatility) * np.sqrt(252) if volatility > 0 else 0

    print(f"Total Return: {total_return*100:.2f}%")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")

    return df_portfolio

# ----------------------------
# Run the cash allocation simulation
# ----------------------------

portfolio_df = simulate_cash_allocation(
    initial_cash=1000,
    model=model,
    X_train=x_train,
    y_train=y_train,
    X_test=x_test,
    y_test=y_test,
    retrain=False,
    allocation_factor=10.0
)

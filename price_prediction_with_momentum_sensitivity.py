import pandas as pd
import yfinance as yf
from datetime import datetime
from pykalman import KalmanFilter
from statsmodels.tsa.ar_model import AutoReg
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

ticker = "TSLA"
start = "2019-01-01"
end = "2025-01-01"
LAGS = 4
WINDOW = 30

def download(ticker, start, end=datetime.now()):
  return yf.Ticker(ticker).history(start=start, end=end)[["Close"]]

data = download(ticker=ticker, start=start, end=end)

def apply_kalman_filter(
    data,
    transition_mtx=[1],
    observation_mtx=[1],
    initial_state_cov=1,
    transition_cov=0.01,
    observation_covariance=1,
    lookback=3
):
    close = data['Close'].copy()

    kf = KalmanFilter(
        transition_matrices=transition_mtx,
        observation_matrices=observation_mtx,
        initial_state_mean=close.iloc[0],
        initial_state_covariance=initial_state_cov,
        observation_covariance=observation_covariance,
        transition_covariance=transition_cov,
    )


    state_means, _ = kf.filter(close)

    data["filtered"] = state_means

    for i in range(1, lookback + 1):
        data[f"filtered_{i}"] = data["filtered"].shift(i - 1)


    data["Target"] = data["filtered"].shift(-1)
    data = data.drop(columns=["filtered"])

    return data


def get_beta(ds, lags, trend='c'):
    model = AutoReg(ds, lags=lags, trend=trend)
    results = model.fit()

    betas = results.params.filter(like='L').values
    n = len(betas)

    S1 = np.mean(betas)
    w = np.arange(1, n + 1)[::-1] / np.sum(np.arange(1, n + 1))
    S2 = np.sum(w * betas)
    S3 = np.linalg.norm(betas)

    return np.array([S1, S2, S3])


def apply_features(data, window, lags):
    close = data['Close']
    results = np.full((len(close), 3), np.nan)

    for i in range(window - 1, len(close)):
        window_data = close.iloc[i - window + 1:i + 1]
        results[i, :] = get_beta(window_data, lags)

    feature_df = pd.DataFrame(results, columns=['S1', 'S2', 'S3'], index=data.index)
    data["S1"] = feature_df["S1"]
    data["S2"] = feature_df["S2"]
    data["S3"] = feature_df["S3"]
    return data


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


data = apply_features(data=data, window=WINDOW, lags=LAGS)
data = apply_kalman_filter(data)
data = data.dropna()


x_train, x_test, y_train, y_test = preprocessing(data, test_size=0.2)

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

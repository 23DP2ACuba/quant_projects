'''Stock Pricing model with GBRM risk and volatility modeling'''

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans

# ==========================
# Config
# ==========================
n = 8
ma_len = 10


# ==========================
# Model
# ==========================
class LRPricingModel:
    def __init__(self, n):
        self.n = n

    def create_features(self, df, ma_len=None):
        self.df = df
        self.ma_len = ma_len if ma_len is not None else self.ma_len

        self.df["Return"] = (self.df["Close"] - self.df["Open"]) / self.df["Open"]
        self.df["Volatility"] = self.df["Return"].rolling(window=8).std()
        self.df["HL_pct"] = ((self.df["High"] - self.df["Low"]) / self.df["High"]) * 100
        self.df["TP"] = ((self.df["High"] + self.df["Low"] + self.df["Close"]) / 3) * self.df["Volume"]
        self.df["MA"] = self.df["Close"].rolling(window=self.ma_len).mean()
        self.df["lag1"] = self.df["Return"].shift(1)
        self.df["lag3"] = self.df["Return"].shift(3)
        self.df["lag7"] = self.df["Return"].shift(7)

        return df

    def create_target(self):
        self.df["Target"] = self.df["MA"].shift(-self.n)
        self.df.dropna(inplace=True)

        x = self.df.drop(columns=["Target"])
        y = self.df["Target"]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.2, shuffle=False
        )

        self.cluster_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KMeans(n_clusters=3, random_state=42))
        ])
        self.cluster_pipeline.fit(self.x_train)

        self.x_train = self.x_train.assign(market_state=self.cluster_pipeline.predict(self.x_train))
        self.x_test = self.x_test.assign(market_state=self.cluster_pipeline.predict(self.x_test))
        print("Train features:", self.x_train.columns.tolist())

    def train(self):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LinearRegression())
        ])

        param_grid = {
            'lr__fit_intercept': [True, False]
        }

        tscv = TimeSeriesSplit(n_splits=5)
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring='r2',
            cv=tscv,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(self.x_train, self.y_train)
        self.pipeline = grid_search.best_estimator_
        self.y_pred = self.pipeline.predict(self.x_test)

        r2 = r2_score(self.y_test, self.y_pred)

        print("Best params:", grid_search.best_params_)
        print("Best CV R²:", grid_search.best_score_)
        print("Test R²:", r2)

    def predict(self, x, n):
      x = self.create_features(x, self.ma_len).dropna()
      x = x.head(1)
      print(x)
      x = x.assign(market_state=self.cluster_pipeline.predict(x))

      return self.pipeline.predict(x), x["Close"]

    def debug_test(self):
        for i in range(len(self.y_pred)):
            print(f"{self.y_pred[i]:.2f}, {self.y_test.iloc[i]:.2f}")

def model_training(df, n, ma_len):
    pricing_model = LRPricingModel(n)
    pricing_model.create_features(df, ma_len)
    pricing_model.create_target()
    pricing_model.train()

    return pricing_model
  
def brownian_bridge(S0, Sn, n_days=30, n_sim=1000, sigma=0.2, mu=0.3, dt=1/252):
    paths = np.zeros((n_days + 1, n_sim))
    paths[0] = S0
    paths[-1] = Sn  

    for t in range(1, n_days):
        tau = t / n_days
        bridge_var = sigma**2 * dt * (1 - tau)
        Z = np.random.normal(0, 1, n_sim)
        paths[t] = S0 + tau * (Sn - S0) + Z * np.sqrt(bridge_var) * S0

    return paths

# ==========================
# Run
# ==========================
data = yf.Ticker("TSLA").history(start="2020-01-01", end="2025-09-17")
data = data[["Open", "High", "Low", "Close", "Volume"]]
print(data.columns)
model = model_training(data.copy(), n, ma_len)
x = data.tail(n+ma_len)
Sn, x = model.predict(x, n)
print(Sn)
print(brownian_bridge(S0=x, Sn=Sn[0], n_days=n, n_sim=1000))

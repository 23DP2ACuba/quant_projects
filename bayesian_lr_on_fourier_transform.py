import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans


n = 5
ma_len = 3
cutoff = 0.02
symbol = "TSLA"
start = "2020-01-01"
end = "2025-09-17"


class LRPricingModel:
    def __init__(self, n, cutoff, ma_len):
        self.n = n
        self.ma_len = ma_len
        self.cutoff = cutoff
        self.df = None

    def apply_fourier_denoise(self):
        prices = self.df["Close"].values
        denoised = []

        for i in range(1, len(prices) + 1):
            window_data = prices[:i]
            t = len(window_data)

            fft_data = np.fft.fft(window_data)
            freq = np.fft.fftfreq(t, d=1)

            fft_filtered = fft_data.copy()
            fft_filtered[np.abs(freq) > self.cutoff] = 0

            smoothed = np.fft.ifft(fft_filtered).real
            denoised.append(smoothed[-1])

        self.df.loc[:, "DenoisedClose"] = denoised

    def create_features(self, df):
        self.df = df.copy()

        self.df.loc[:, "Return"] = (self.df["Close"] - self.df["Open"]) / self.df["Open"]
        self.df.loc[:, "Volatility"] = self.df["Return"].rolling(window=8).std()
        self.df.loc[:, "HL_pct"] = ((self.df["High"] - self.df["Low"]) / self.df["High"]) * 100
        self.df.loc[:, "TP"] = ((self.df["High"] + self.df["Low"] + self.df["Close"]) / 3) * self.df["Volume"]
        self.df.loc[:, "MA"] = self.df["Close"].rolling(window=self.ma_len).mean()
        self.df.loc[:, "lag1"] = self.df["Return"].shift(1)
        self.df.loc[:, "lag3"] = self.df["Return"].shift(3)
        self.df.loc[:, "lag7"] = self.df["Return"].shift(7)
        self.apply_fourier_denoise()

        return self.df

    def apply_clustering(self):
        self.cluster_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KMeans(n_clusters=3, random_state=42))
        ])
        self.cluster_pipeline.fit(self.x_train)

        self.x_train = self.x_train.assign(market_state=self.cluster_pipeline.predict(self.x_train))
        self.x_test = self.x_test.assign(market_state=self.cluster_pipeline.predict(self.x_test))

    def create_target(self, test_size):
        self.df.loc[:, "Target"] = self.df["DenoisedClose"].shift(-self.n)
        self.df.dropna(inplace=True)

        x = self.df.drop(columns=["Target"])
        y = self.df["Target"]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=test_size, shuffle=False
        )
        self.apply_clustering()

    def train(self):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('bayesridge', BayesianRidge())
        ])

        param_grid = {
            "bayesridge__max_iter": [300, 500, 1000], 
            "bayesridge__tol": [1e-3, 1e-4],
            "bayesridge__alpha_1": [1e-6, 1e-5, 1e-4],
            "bayesridge__alpha_2": [1e-6, 1e-5, 1e-4],
            "bayesridge__lambda_1": [1e-6, 1e-5, 1e-4],
            "bayesridge__lambda_2": [1e-6, 1e-5, 1e-4],
            "bayesridge__fit_intercept": [True, False],
            "bayesridge__compute_score": [True, False],
        }

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )


        grid_search.fit(self.x_train, self.y_train)
        self.pipeline = grid_search.best_estimator_
        self.y_pred = self.pipeline.predict(self.x_test)

        r2 = r2_score(self.y_test, self.y_pred)

        print("Best params:", grid_search.best_params_)
        print("Best CV R²:", grid_search.best_score_)
        print("Test R²:", r2)


def model_training(df, n, cutoff, ma_len, test_size=0.2, debug=False):
  pricing_model = LRPricingModel(n, cutoff, ma_len)
  pricing_model.create_features(df)
  pricing_model.create_target(test_size)
  pricing_model.train()
  if debug:
    pricing_model.debug_test()

  return pricing_model

data = yf.Ticker(symbol).history(start=start, end=end)
data = data[["Open", "High", "Low", "Close", "Volume"]]

model = model_training(data.copy(), n, cutoff, ma_len, debug=False)

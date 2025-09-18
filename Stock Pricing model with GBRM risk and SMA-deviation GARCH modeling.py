'''Stock Pricing model with GBRM risk and SMA-deviation GARCH modeling'''

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from arch import arch_model

pd.options.mode.chained_assignment = None

# ==========================
# Config
# ==========================
n = 5
ma_len = 3
cutoff = 0.06
n_sim = 200
p, q = 2, 0

# ==========================
# Model
# ==========================
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
        self.df.loc[:, "Target"] = self.df["MA"].shift(-self.n)
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
        x = self.create_features(x).dropna()
        x = x.head(1)
        x = x.assign(market_state=self.cluster_pipeline.predict(x))

        return self.pipeline.predict(x)[0], x["Close"].iloc[0]

    def debug_test(self):
        idx = self.x_test.index
        for i in range(len(self.y_pred)):
            print(f"{idx[i]}: {self.y_pred[i]:.2f}, {self.y_test.iloc[i]:.2f}")


def model_training(df, n, cutoff, ma_len, test_size=0.2):
    pricing_model = LRPricingModel(n, cutoff, ma_len)
    pricing_model.create_features(df)
    pricing_model.create_target(test_size)
    pricing_model.train()

    return pricing_model


# ==========================
# Brownian Bridge
# ==========================
def brownian_bridge(df, S0, Sn, pred_dev, n_days=30, n_sim=1000, dt=1/252):

    def generate_bridge(Sn):
        paths = np.zeros((n_days + 1, n_sim))
        paths[0] = S0
        paths[-1] = Sn

        for t in range(1, n_days):
            tau = t / n_days
            bridge_var = pred_dev**2 * dt * (1 - tau)
            Z = np.random.normal(0, 1, n_sim)
            paths[t] = S0 + tau * (Sn - S0) + Z * np.sqrt(bridge_var)

        return paths

    plt.plot(df["Close"], label="Historical Close")

    print(f"S_n + deviation: {Sn + pred_dev}")
    print(f"S_n - deviation: {Sn - pred_dev}")

    paths1 = generate_bridge(Sn + pred_dev)
    paths2 = generate_bridge(Sn - pred_dev)

    plt.figure(figsize=(10, 6))
    for i in range(n_sim):
        plt.plot(paths1[:, i], lw=1.0, alpha=0.6, color="blue")
    for i in range(n_sim):
        plt.plot(paths2[:, i], lw=1.0, alpha=0.6, color="red")

    plt.title("Brownian Bridge Stock Price Paths")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.show()


# ==========================
# GARCH on SMA Deviations
# ==========================
class GARCH:
    def __init__(self, series_or_df, p, q, n, ma_len=10):
        if isinstance(series_or_df, pd.DataFrame):
            self.close = series_or_df["Close"].copy()
        else:
            self.close = series_or_df.copy()
        self.p = p
        self.q = q
        self.n = n
        self.ma_len = ma_len

    def get_deviation(self):
        sma = self.close.rolling(self.ma_len).mean()
        deviation = (self.close - sma).dropna()

        model = arch_model(deviation, p=self.p, q=self.q, vol="Garch", dist="normal")
        model_fit = model.fit(disp="off")

        pred = model_fit.forecast(horizon=1)
        dev_std = np.sqrt(pred.variance.values[-1, :][0])

        return dev_std


# ==========================
# Run
# ==========================
data = yf.Ticker("TSLA").history(start="2020-01-01", end="2025-09-17")
data = data[["Open", "High", "Low", "Close", "Volume"]]

model = model_training(data.copy(), n, cutoff, ma_len)
x = data.tail(n + ma_len)
Sn, x = model.predict(x, n)

garch = GARCH(data, p, q, n, ma_len=ma_len)
pred_dev = garch.get_deviation()

brownian_bridge(df=data, S0=x, Sn=Sn, n_days=n, n_sim=n_sim, pred_dev=pred_dev)

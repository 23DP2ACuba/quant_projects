import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from scipy.stats import genpareto
import math
import numpy as np
from dataclasses import dataclass

@dataclass
class Params:
  ticker: str
  start: str
  end: str
  vol_window: int = 20
  T: float = 1/252
  garch_window: int = 60
  tcm_window: int = 150
  q: float = 0.90
  pct: float = 0.95
  bounds = [(-1, 2), (1e-6, None)]


class LatentSpaceVol(Params):
  def get_data(self):
    self.data = yf.Ticker(self.ticker).history(start=self.start, end=self.end, auto_adjust=False)[["Close"]]

  def get_tail_data(self, returns):
    L = np.abs(returns[returns < 0])
    if len(L) < 10:
      return np.nan, np.nan, np.nan, np.nan, np.nan

    u = np.percentile(L, 100 * self.pct)
    P_u = np.mean(L > u)
    Y = L[L > u] - u

    xi, loc, beta = genpareto.fit(Y, floc=0)

    VaR_q = u + beta / xi * (((1 - self.q) / P_u) ** (-xi) - 1)
    CTE_q = (VaR_q + (beta - xi * u)) / (1 - xi)

    moments = self.get_tail_moments_gpd((xi, beta))
    tv, skewness, kurtosis = moments[1:]

    return xi, CTE_q, tv, skewness, kurtosis

  def get_tail_moments_gpd(self, params):
    xi, beta = params
    moments = []
    order = 4
    eps = 1e-6

    for k in range(1, order+1):
      if xi >= 1 / k:
        moments.append(eps)
        continue

      moment = np.exp(
        math.lgamma(k + 1) + k * np.log(beta) - sum(np.log(1 - j * xi) for j in range(1, k + 1))
      )
      moments.append(round(moment, 8))

    return moments

  def gpd_neg_log_likelihood(self, params: tuple, Y: np.ndarray):
    xi, beta = params
    eps = 1e-6

    if beta <= 0:
      return eps
    if abs(xi) < eps:
      nll = np.sum(np.log(beta) + Y/beta)

    else:
      t = 1 + xi * Y / beta
      if np.any(t <= 0):
        return eps
      nll = np.sum(np.log(beta) + (1/self.xi+ 1) * np.log(t))

    return nll


  def get_features(self):
    r = self.data["Close"].pct_change()
    r2 = r**2
    self.data["RealizedVOl"] = np.sqrt(r2.rolling(window=self.vol_window).mean())

    self.data["StdDev"] = r.rolling(window=self.vol_window).std()

    self.data["MAD"] = r.rolling(window=self.vol_window).apply(lambda x: np.mean(np.abs(x-np.mean(x))))
    
    tcm_cols = ["xi", "CTE", "TV", "Skew", "Kurt"]
    tcm = pd.DataFrame(index=self.data.index, columns=tcm_cols)

    print("caculating tcm:")
    step = np.floor(len(r)/15)

    for i in range(self.tcm_window, len(r)):
      window = r.iloc[i - self.tcm_window:i].dropna()
      tcm.iloc[i] = self.get_tail_data(window)
      if i % step == 0:
        print("|", end="")


    self.data = pd.concat([self.data, tcm], axis=1)

    self.data.dropna(inplace=True)


import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_random_walk(end):
  x = np.random.normal(size=end)
  pd.plotting.autocorrelation_plot(x, label="random walk autocor", color="red")
  plt.show()

def get_data(symbol, start, end):
  global data
  data = yf.Ticker(symbol).history(start=start, end=end)[["Close"]]
  data["diff"] = data["Close"].diff()
  ds = data["diff"].dropna()
  pd.plotting.autocorrelation_plot(ds, label="price autocor")
  plt.show()

get_random_walk(2000)
get_data("SPY", "2020-01-01", "2025-01-01")



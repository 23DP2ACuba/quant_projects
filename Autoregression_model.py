import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import yfinance as yf

np.random.seed(42)

def get_data(symbol, start, end):
    global data
    data = yf.Ticker(symbol).history(start=start, end=end)[["Close"]]
    data["diff"] = np.log(data["Close"]).diff().dropna() 
    return data["diff"].dropna()  

ds = get_data("IBM", "2020-01-01", "2025-01-01")

plt.figure(figsize=(10, 6))
plt.plot(ds, label="Log-Differenced Returns")
plt.title("Log-Differenced Returns of IBM")
plt.xlabel("Date")
plt.ylabel("Log Returns")
plt.legend()
plt.grid(True)
plt.show()

max_lags = 10
bic_values = []
lag_range = range(1, max_lags + 1)
models = []

for lags in lag_range:
    try:
        model = AutoReg(ds, lags=lags, trend='c') 
        results = model.fit()
        bic_values.append(results.bic)
        models.append(results)
    except Exception as e:
        print(f"Error fitting model with {lags} lags: {e}")
        bic_values.append(np.inf)
        models.append(None)

optimal_lags = lag_range[np.argmin(bic_values)]
optimal_model = models[np.argmin(bic_values)]

if optimal_model is None:
    print("No valid model found.")
    exit()

print(f"Selected AR order (using BIC): {optimal_lags}")
print(f"BIC value: {optimal_model.bic:.4f}")
print(optimal_model.summary())
print(f"AR coefficients: {optimal_model.params}")

fitted = optimal_model.fittedvalues
plt.figure(figsize=(10, 6))
plt.plot(ds.index[optimal_lags:], ds[optimal_lags:], label="Log-Differenced Returns")
plt.plot(ds.index[optimal_lags:], fitted, label="Fitted Values", linestyle="--")
plt.title("Log-Differenced Returns vs. Fitted Values")
plt.xlabel("Date")
plt.ylabel("Log Returns")
plt.legend()
plt.grid(True)
plt.show()

residuals = optimal_model.resid
print(f"Residuals mean: {residuals.mean():.4f}")

plt.figure(figsize=(10, 6))
plt.plot(lag_range, bic_values, marker="o", color="#1f77b4")
plt.xlabel("Number of Lags")
plt.ylabel("BIC")
plt.title("BIC vs. Number of Lags")
plt.grid(True)
plt.show()

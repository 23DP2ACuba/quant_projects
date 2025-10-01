from arch import arch_model
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
# %%
data = yf.Ticker("TSLA").history(start="2020-01-01", end="2025-01-01")[["Close"]]
log_return = np.log(data["Close"]/data["Close"].shift(1)).dropna()

plt.figure(figsize=(10,4))
plt.plot(log_return, label="Simulated GARCH(1,1)")
plt.legend()
plt.show()

plot_acf(log_return, lags=40)
plt.show()

model = arch_model(log_return, p=1, q=1, vol="GARCH", dist="normal")
results = model.fit()

print(results)
print(results.conf_int())
# %%
alpha0 = 0.1
alpha1 = 0.4
beta1 = 0.2

w = np.random.normal(size=2000)
x= np.zeros(2000)
sigma2 = np.zeros(2000)

for t in range(2, 2000):
    sigma2[t] = alpha0+alpha1*x[t-1]**2+beta1*sigma2[t-1]
    x[t] = np.sqrt(sigma2[t]) * w[t]

plt.figure(figsize=(10,4))
plt.plot(x, label="Simulated GARCH(1,1)")
plt.legend()
plt.show()

plot_acf(x, lags=40)
plt.show()

model = arch_model(x, p=1, q=1, vol="GARCH", dist="normal")
results = model.fit()

print(results.conf_int())

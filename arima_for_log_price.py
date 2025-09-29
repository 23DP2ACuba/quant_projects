import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.arima.model import ARIMA

data = yf.Ticker("TSLA").history(start="2020-01-01", end="2025-01-01")[["Close"]]

lnprice = np.log(data["Close"])

acf_1 = acf(lnprice)[1:20]
test_df = pd.DataFrame([acf_1]).T
test_df.columns=["Autocorelation"]
test_df.index+=1
test_df.plot(kind="bar")

pacf_1 = pacf(lnprice)[1:20]
test_df = pd.DataFrame([pacf_1]).T
test_df.columns=["PartialAutocorelation"]
test_df.index+=1
test_df.plot(kind="bar")

result = ts.adfuller(lnprice, 1)
print(result)
diff = lnprice.diff().dropna()
acf_1_diff = acf(diff)[1:20]
test_df = pd.DataFrame([acf_1_diff]).T
test_df.columns = ["first_order_autocor"]
test_df.index += 1
test_df.plot(kind="bar")
plt.show()

price_mtx = lnprice.values
model = ARIMA(price_mtx, order=(0, 1, 0))

model_fit = model.fit()

print(model_fit.summary())
preds = model_fit.predict(122, 127, typ = "levels")
preds_adj = np.exp(preds)
plt.plot(preds_adj)
plt.show()



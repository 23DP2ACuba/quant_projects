import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sn
import statsmodels.tsa.stattools import ts, adfuller


def get_data(start:str, tickers: list[str]) -> pd.DataFrame:
    df = yf.download(tickers=tickers, start=start)["Close"]
    return df

tickers = ["GOOG", "AMZN", "AMD", "BRK-B","SPY", "NFLX", "MSFT", "AAPL", "NVDA", "PLTR", "TSLA"]
data = get_data(start="2020-01-01", tickers=tickers)
print(data.tail(5))

corr_mtx = data.corr()

plt.figure(figsize=(8, 6), dpi=200)
sn.heatmap(corr_mtx, annot=True)
corr_stocks = [data["BRK-B"], data["MSFT"]]

result = ts.coint(corr_stocks[0], corr_stocks[1])

coint_stats = result[0]
p_val = result[1]
cvt = result[2]

adf_stock1, adf_stock2 = [adfuller(stock) for stock in corr_stocks]

spread_adf = adfuller(corr_stocks[0] - corr_stocks[1])



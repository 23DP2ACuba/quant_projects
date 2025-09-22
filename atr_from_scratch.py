import pandas as pd 
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

def atr(data):
    hl = data["High"] - data["Low"]
    hc = np.abs(data["High"] - data["Close"].shift(1))
    lc = np.abs(data["Low"] - data["Close"])
    
    ranges = pd.concat([hl, hc, lc], axis=1)
    tr = np.max(ranges, axis=1)
    atr_ = tr.rolling(14).mean()
    return atr_
    

if __name__ == "__main__":
    start = "2020-01-01"
    end = "2025-01-01"
    
    data = yf.download("XOM", start, end)
    atr_ = atr(data)
    plt.plot(atr_, label="atr")
    plt.show()

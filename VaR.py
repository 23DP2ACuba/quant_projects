import numpy as np
import yfinance as yf
from scipy.stats import norm
import pandas as pd
import datetime

class VaRMC:
    def __init__(self, S, mu, sigma, c, n, i):
        self.S = S
        self.mu = mu
        self.c = c
        self.n = n
        self.sigma = sigma
        self.i = i
    
    def sim(self):
        rand = np.random.normal(0, 1, [1, self.i])
        sp = self.S * np.exp(self.n * (self.mu - 0.5*self.sigma**2) + self.sigma * np.sqrt(self.n)*rand)

        sorted_sp = np.sort(sp)

        percentile = np.percentile(sorted_sp, (1-self.c)*100)

        return self.S - percentile


def download_data(stock, start_date, end_date):
    data = {}
    ticker = yf.download(stock, start=start_date, end=end_date)
    return ticker["Close"]

def calculate_var(pos, c, mu, std): # (n=1)
    var = pos * (mu - std * norm.ppf(1-c))
    return var
def calculate_var_n(pos, c, mu, std, n):
    var = pos * (mu * n - std * norm.ppf(1-c) * np.sqrt(n))
    return var

def calculate_var_mc(S, mu, sigma, c, n, i):
    model = VaRMC(S, mu, sigma, c, n, i)
    return model.sim()

if __name__ == "__main__":
    start = datetime.datetime(2014, 1, 1)
    end = datetime.datetime(2017, 10, 15)
    stock = "C"

    S = 1e6
    c = 0.99
    n = 1
    i = 10000

    data = download_data(stock, start, end)

    data["returns"] = np.log(data[stock] / data[stock].shift(1))

    mu = np.mean(data["returns"])
    std = np.std(data["returns"])



    data = data.dropna()

    print(f"VaR tomorrow: {calculate_var(S, c, mu, std)}")
    print(f"VaR over n days: {calculate_var_n(S, c, mu, std, n)}")
    print(f"MC VaR over n days: {calculate_var_mc(S, mu, std, c, n, i)}")

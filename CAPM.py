import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

class CAPM:
    def __init__(self, stocks, start, end, rfr=None):
        self.stocks = stocks
        self.start = start
        self.end = end
        self.data = None
        self.rfr = rfr

    def download_data(self):
        tickers = yf.download(self.stocks, self.start, self.end)["Close"]
        return tickers
    
    def initialize(self):
        stock_data = self.download_data()
        stock_data = stock_data.resample("M").last()
        self.data = pd.DataFrame({"s_close": stock_data[self.stocks[0]],
                                  "m_close": stock_data[self.stocks[1]],
                                  })
        self.data[["s_returns", "m_returns"]] = np.log(self.data[["s_close", "m_close"]] / self.data[["s_close", "m_close"]].shift(1))
        self.data.dropna(inplace=True)

    def get_beta(self):
        cov_matrix = np.cov(self.data["s_returns"], self.data["m_returns"])
        print(cov_matrix)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        print("beta from formula", beta)

    def CAPM(self):
        beta, alpha = np.polyfit(self.data["m_returns"], self.data["s_returns"], deg = 1)
        print(f"Beta: {beta}, Alpha: {alpha}")

        ex_return = self.rfr + beta * (self.data["m_returns"].mean()*12 - self.rfr)
        print(ex_return)
        self.plot_model(alpha, beta)

    def plot_model(self, alpha, beta):
        fig, axis = plt.subplots(1, figsize=(20, 10))
        axis.scatter(self.data["m_returns"], self.data["s_returns"], label="Data Points")
        axis.plot(self.data["m_returns"], beta * self.data["m_returns"] + alpha, color = "red", label = "CAPM line")
        plt.xlabel("market return $R_m")
        plt.ylabel("asset return $R_a")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    capm = CAPM(["IBM", "^GSPC"], "2010-01-01", "2017-01-01", rfr=0.05)
    capm.initialize()
    capm.get_beta()
    capm.CAPM()



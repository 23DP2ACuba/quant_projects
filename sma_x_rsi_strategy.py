import numpy as np 
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd

class MARSIStrategy:
    def __init__(self, capital, stock, start, end, short_period, long_period):
        self.data = None
        self.is_long=False
        self.short_period = short_period
        self.long_period = long_period
        self.capital = capital
        self.equity = [capital]
        self.stock = stock
        self.start = start
        self.end = end
        
    def construct_signals(self):
        self.data["short_ma"] = self.data["Close"].ewm(span=self.short_period).mean()
        self.data["long_ma"] = self.data["Close"].ewm(span=self.long_period).mean()
        
        self.data["move"] = self.data["Close"] - self.data["Close"].shift(1)
        self.data["up"] = np.where(self.data["move"] > 0, self.data["move"], 0)
        self.data["down"] = np.where(self.data["move"] < 0, self.data["move"], 0)
        
        self.data["average_gain"] = self.data["up"].rolling(14).mean()
        self.data["average_loss"] = self.data["down"].abs().rolling(14).mean()
        
        self.data["RS"] = self.data["average_gain"] / self.data["average_loss"]
        self.data["RSI"] = 100 - (100/(1+self.data["RS"]))

        self.data.dropna(inplace=True)        

    def plot_signals(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.data["Close"], label="Stock price")
        plt.plot(self.data["short_ma"], label="short ma")
        plt.plot(self.data["long_ma"], label="long ma")
        plt.title("moving average x rsi strategy")
        plt.plot(self.data["RSI"])
        plt.show()
        
    def plot_equity(self):
        plt.plot(self.equity, label="Portfolio value")
        plt.show()
        
    def show_stats(self, rfr = 0):
        print(f"profit: {(float(self.equity[-1])-float(self.equity[0])) / float(self.equity[0]) * 100}")
        print(f"Portfolio value: {self.equity[-1]}")
        
        returns = (self.data["Close"] - self.data["Close"].shift(1)) / self.data["Close"].shift(1)
        ratio = returns.mean() / returns.std() * np.sqrt(252)
        
        print(f"Annualized Sharpe ratio: {ratio[0]}")
        
    def simmulate(self):
        price_when_buy = 0
        for index, row in self.data.iterrows():
            print(row["short_ma"][0])
            if row["short_ma"][0] < row["long_ma"][0]:
                self.equity.append(row["Close"][0] * self.capital / price_when_buy)
                self.is_long = False
                
            elif row["short_ma"][0] > row["long_ma"][0] and \
                            not self.is_long and \
                            row["RSI"][0] < 30:
                    
                self.is_long = True
                price_when_buy = row["Close"][0]
                self.equity.append(self.equity[-1])
            else:
               self.equity.append(self.equity[-1]) 
                               
        
    def download(self):
        self.data = yf.download(self.stock, self.start, self.end)

if __name__ == "__main__":
   start = "2020-01-01"
   end = "2025-01-01"
   model = MARSIStrategy(100, "IBM", start, end, 30, 100)
   model.download()
   model.construct_signals()
   #model.plot_signals()
   model.simmulate()
   model.show_stats()

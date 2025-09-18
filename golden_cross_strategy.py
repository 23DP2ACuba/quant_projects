import yfinance  as yf 
import matplotlib.pyplot as plt
import pandas as pd

class MovingAverageCrossover:
    def __init__(self, capital, stocks, start, end, short_p, long_p):
        self.capital = capital
        self.is_long = False
        self.data = None
        self.short_period = short_p
        self.long_period = long_p
        self.equity = [capital]
        self.stock = stocks
        self.start = start
        self.end = end
        
    def download_data(self):
        self.data = yf.download(tickers=self.stock, start=self.start, end=self.end)[["Close"]]
        
    def construct_signals(self):
        self.data["short_ma"] = self.data["Close"].ewm(span=self.short_period).mean()
        self.data["long_ma"] = self.data["Close"].ewm(span=self.long_period).mean()
    
    def plot(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.Close, label="Close Price")
        plt.plot(self.data.short_ma, label="long moving average")
        plt.plot(self.data.long_ma, label="short moving average")
        plt.title("MA crossover strategy")
        plt.show()
        
    def simmulate(self):
        price_when_buy = 0
        for index, row in self.data.iterrows():
            if row.short_ma.values < row.long_ma.values and self.is_long:
                self.equity.append(self.capital*row.Close.values[0] / price_when_buy)
                self.is_long = False
                #print("SELL")
                
            elif row.short_ma.values > row.long_ma.values and not self.is_long:
                self.is_long = True
                price_when_buy = row.Close.values
                #print("BUY")
                
    def plot_equity(self):
        print(self.equity)
        print(f"Plot of the trading strategy: {float(self.equity[-1] - float(self.equity[0])) / float(self.equity[0])*100}")
        print(f"Actual_capital{self.equity[-1]}")
        plt.figure(figsize=(12, 6))
        plt.title("equity curve")
        plt.plot(self.equity, label="Equity")
        plt.show()
 
if __name__ == "__main__":
    start = "2020-01-01"
    end = "2025-01-01"
    strategy = MovingAverageCrossover(100, "IBM", start, end, 30, 50)
    strategy.download_data()
    strategy.construct_signals()
    strategy.simmulate()
    strategy.plot_equity()

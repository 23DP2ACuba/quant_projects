import backtrader as bt
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm  

class CrossSectionalMR(bt.Strategy):
    def __init__(self):
        self.data = self.datas
        self.logs = []
        self.total_bars = len(self.datas[0])
        self.bar_counter = 0
        self.pbar = None

    def log(self, txt, dt=None, dname=""):
        dt = dt or self.datas[0].datetime.date(0)
        self.logs.append(f"{dt.isoformat()} [{dname}] {txt}")

    def prenext(self):
        self.next()

    def start(self):
        self.total_bars = len(self.datas[0])
        self.bar_counter = 0
        self.pbar = tqdm(total=self.total_bars, desc="Backtesting Progress")

    def next(self):
        stock_returns = np.zeros(len(self.data))
        for index, stock in enumerate(self.data):
            stock_returns[index] = (stock.close[0] - stock.close[-1]) / stock.close[-1]

        market_return = np.mean(stock_returns)
        wi = -(stock_returns - market_return)
        wi = wi / np.sum(np.abs(wi))

        for index, stock in enumerate(self.data):
            self.order_target_percent(stock, wi[index])

        self.bar_counter += 1
        if self.pbar:
            self.pbar.update(1)

    def stop(self):
        if self.pbar:
            self.pbar.close()
        with open("log.txt", "w") as f:
            f.write("\n".join(self.logs))


if __name__ == "__main__":
    START = "2020-01-01"
    tickers = []
    cerebro = bt.Cerebro()
    
    with open("companies_cross_sectional.txt") as file_in:
        for line in file_in:
            ticker = line.strip()
            if ticker:
                tickers.append(ticker)

    if len(tickers) > 0:
        data = yf.download(tickers=tickers, start=START, group_by="ticker")
        print(f"\nTotal Assets: {len(data.columns)} \ncolumn length: {len(data[data.columns[0]])}")

        for ticker in tickers:
            df = data[ticker].dropna()
            if len(df) > 100:
                feed = bt.feeds.PandasData(dataname=df, plot=False)
                cerebro.adddata(feed, name=ticker)

        cerebro.broker.set_cash(10000)
        print(f"Initial Capital: {cerebro.broker.getvalue():.2f}")

        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

        cerebro.addstrategy(CrossSectionalMR)
        results = cerebro.run()
        strat = results[0]

        print(f"Final Capital: {cerebro.broker.getvalue():.2f}\n")
        print("Sharpe Ratio:", strat.analyzers.sharpe.get_analysis())
        print("Returns:", strat.analyzers.returns.get_analysis())
        print("DrawDown:", strat.analyzers.drawdown.get_analysis())
        print("TradeAnalyzer:", strat.analyzers.trades.get_analysis())

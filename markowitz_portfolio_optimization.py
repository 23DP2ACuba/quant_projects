import numpy as np 
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optim

stocks = ["AAPL", "WHT", "TSLA", "AMZN", "DB", "GE"]

start = "2010-01-01"
end = "2025-01-01"


NUM_DAYS = 252
NUM_PORTFOLIOS = 10_000

def download(stocks, start, end):
    return yf.download(stocks, start, end)["Close"]

def show(data):
    data.plot()
    plt.show()

def show_stats(returns):
    print(returns.mean()*NUM_DAYS)
    print(returns.cov()*NUM_DAYS)


def Exreturn(data):
    log_return = np.log(data/data.shift(1))
    show_stats(log_return)
    return log_return[1:]

def stats(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * NUM_DAYS
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, 
               np.dot(returns.cov()* NUM_DAYS, weights)
            )
        )
    
    sharpe = portfolio_return / portfolio_volatility

    return np.array([portfolio_return, 
                     portfolio_volatility,
                     sharpe])


def min_function_sharpe(weights, returns):
    return -stats(weights, returns)[2]

def generate_portfolio(returns):
    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []

    for i in range(NUM_PORTFOLIOS):
        w = np.random.random(len(stocks))
        w /= np.sum(w)
        print(i)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean() * w) * NUM_DAYS)
        portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(returns.cov()
                                                          * NUM_DAYS, w))))



    return  np.array(portfolio_weights), \
            np.array(portfolio_means), \
            np.array(portfolio_risks)

def optimize_portfolio(weights, returns):
    cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0,1) for _ in range(len(stocks)))
    return optim.minimize(fun=min_function_sharpe, 
                          x0=weights[0], 
                          args=returns, 
                          method="SLSQP", 
                          bounds=bounds, 
                          constraints=cons)



def print_optimal_portfolio(optimum, returns):
    optimum = optimum["x"].round(3)
    print(f"optimal: {optimum}, \
          stats: {stats(optimum, returns)}")

def show_optimal_portfolio(opt, rets, preturns, pvol):
    plt.scatter(pvol, preturns, c=preturns/pvol, marker = "o")
    plt.grid(True)
    plt.xlabel("Expected volatility")
    plt.xlabel("Expected returns")
    plt.colorbar(label="Sharpe ratio")
    plt.plot(stats(opt["x"], rets)[1], 
             stats(opt["x"], rets)[0], 
             "g*", markersize=20)
    plt.show()

if __name__ == "__main__":
    data = download(stocks, start, end)
    log_return = Exreturn(data)
    pweights, means, risks = generate_portfolio(log_return)

    optimum = optimize_portfolio(pweights, log_return)
    print_optimal_portfolio(optimum, log_return)
    show_optimal_portfolio(optimum, log_return, means, risks)

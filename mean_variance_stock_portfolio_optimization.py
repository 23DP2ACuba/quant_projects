import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pypfopt as pft
from pypfopt import plotting


import yfinance as yf

assets = ["KO", "MSFT", "META", "AMZN", "AAPL", "LMT", "XOM", "PLTR", "DIS", "F"]
data = yf.download(assets, start="2020-01-01", end="2025-01-01")
sp_index = yf.download("^GSPC", start="2020-01-01", end="2025-01-01")

portfolio_returns = data["Close"].pct_change().dropna()
return_cumprod = portfolio_returns.add(1).cumprod().sub(1) * 100
correlation = portfolio_returns.corr()
sns.heatmap(correlation)

train = portfolio_returns[:"2024-01-01"]
test = portfolio_returns["2024-01-01":]

mu = pft.expected_returns.ema_historical_return(train, returns_data=True, span=500)
sigma = pft.risk_models.exp_cov(train, returns_data=True, span=180)

ret_ef = np.arange(0, max(mu), 0.01)
vol_ef = []

for i in ret_ef:
    ef = pft.efficient_frontier.EfficientFrontier(mu, sigma)
    ef.efficient_return(i)
    vol_ef.append(ef.portfolio_performance()[1])

ef = pft.efficient_frontier.EfficientFrontier(mu, sigma)
ef.min_volatility()
min_volatility_return, min_volatility_vol = ef.portfolio_performance()[:2]

ef = pft.efficient_frontier.EfficientFrontier(mu, sigma)
ef.max_sharpe(risk_free_rate=0.009)
min_sharpe_return, min_sharpe_vol = ef.portfolio_performance()[:2]

sns.set()
fig, ax = plt.subplots(figsize=[15, 10])
sns.lineplot(x=vol_ef, y=ret_ef, label="Efficient Frontier")

sns.scatterplot(x=[min_volatility_vol], y=[min_volatility_return],
                label="Min Variance Portfolio", color="green", s=100)

sns.scatterplot(x=[min_sharpe_vol], y=[min_sharpe_return],
                label="Max Sharpe Portfolio", color="orange", s=100)

sharpe_ratio = (min_sharpe_return - 0.009) / min_sharpe_vol
x_vals = np.linspace(0, max(vol_ef), 100)
y_vals = 0.009 + sharpe_ratio * x_vals
sns.lineplot(x=x_vals, y=y_vals, label="Capital Market Line", color="red")

ax.set(xlim=[0, 0.5], ylim=[0, 1])
ax.set_xlabel("Volatility")
ax.set_ylabel("Mean Return")
plt.title("Efficient Frontier")
plt.show()



ef = pft.efficient_frontier.EfficientFrontier(mu, sigma)
raw_weights_minvar_exp = ef.min_volatility()
plotting.plot_weights(raw_weights_minvar_exp)
ef.portfolio_performance(verbose=True, risk_free_rate=0.009)

ef = pft.efficient_frontier.EfficientFrontier(mu, sigma)
raw_weights_sharpe_exp = ef.max_sharpe()
plotting.plot_weights(raw_weights_sharpe_exp)
ef.portfolio_performance(verbose=True, risk_free_rate=0.009)



weights_minvar_exp = list(raw_weights_minvar_exp.values())
weights_maxsharpe_exp = list(raw_weights_sharpe_exp.values())

return_1 = test.dot(weights_minvar_exp).add(1).cumprod().subtract(1).multiply(100)
return_2 = test.dot(weights_maxsharpe_exp).add(1).cumprod().subtract(1).multiply(100)

index_return = sp_index["2024-01-01":]["Close"].pct_change().add(1).cumprod().subtract(1).multiply(100)

backtest = pd.DataFrame({"MinVar":return_1, "MaxSharpe":return_2})
backtest = pd.concat([backtest, index_return], join = "outer", axis = 1)

backtest.interpolate(method="linear", inplace=True)

fig = px.line(backtest, x=backtest.index, y=backtest.columns, title = "Portfolio vs SP500 performance")
fig.update_xaxes(title_text="Date")
fig.update_yaxes(title_text="Return")



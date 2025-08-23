import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# --- Simulate trades with Z-score strategy ---
def simulate_zscore_trades(df, entry_col="Trades", sl_pct=0.02, tp_pct=0.05, n_days=10):
    df = df.copy()
    new_target = []

    for i in range(len(df) - n_days):
        signal = df.iloc[i][entry_col]

        if signal == 0:
            new_target.append(np.nan)
            continue

        entry_price = df.iloc[i + 1]["Open"]
        high_window = df["High"].iloc[i + 1:i + 1 + n_days].values
        low_window = df["Low"].iloc[i + 1:i + 1 + n_days].values

        if signal == -1:
            hit_tp = np.any((high_window - entry_price) / entry_price >= tp_pct)
            hit_sl = np.any((entry_price - low_window) / entry_price >= sl_pct)
        else:
            hit_tp = np.any((entry_price - low_window) / entry_price >= tp_pct)
            hit_sl = np.any((high_window - entry_price) / entry_price >= sl_pct)

        if hit_tp and not hit_sl:
            new_target.append(1)
        elif hit_sl and not hit_tp:
            new_target.append(-1)
        elif hit_tp and hit_sl:
            tp_index = np.argmax(hit_tp)
            sl_index = np.argmax(hit_sl)
            new_target.append(1 if tp_index <= sl_index else -1)
        else:
            new_target.append(-1)

    df["Target"] = new_target + [np.nan] * n_days
    return df.dropna()

data["MA"] = data["Close"].rolling(window=30).mean()
data["std"] = data["Close"].rolling(window=30).std()
data["Z-Score"] = (data["Close"] - data["MA"]) / data["std"]

upper_threshold = data["Z-Score"].quantile(0.85)
lower_threshold = data["Z-Score"].quantile(0.15)

data["Trades"] = 0
data.loc[data['Z-Score'] > upper_threshold, "Trades"] = 1
data.loc[data['Z-Score'] < lower_threshold, "Trades"] = -1

data_trades = data.loc[data["Trades"] != 0].copy()

split_index = int(len(data_trades) * 0.8)
train_df = data_trades.iloc[:split_index]
test_df = data_trades.iloc[split_index:]

x_train = train_df[train_df.columns.difference(["Target"])]
y_train = train_df["Target"]

x_test = test_df[test_df.columns.difference(["Target"])]
y_test = test_df["Target"]

pl_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=256, random_state=42))
])
pl_pipeline.fit(x_train, y_train)

x_test = x_test.copy()
x_test["Predicted"] = pl_pipeline.predict(x_test)
x_test["Open"] = test_df["Open"]
x_test["High"] = test_df["High"]
x_test["Low"] = test_df["Low"]
x_test["Trades"] = test_df["Trades"]

tp_pct = 0.05
sl_pct = 0.02
n_days = 10
strategy_returns = []

for i in range(len(x_test) - n_days):
    pred = x_test.iloc[i]["Predicted"]
    if pred == 0:
        strategy_returns.append(0)
        continue

    entry_price = x_test.iloc[i + 1]["Open"]
    high_window = x_test["High"].iloc[i + 1:i + 1 + n_days].values
    low_window = x_test["Low"].iloc[i + 1:i + 1 + n_days].values

    if pred == 1:  
        hit_tp = np.any((high_window - entry_price)/entry_price >= tp_pct)
        hit_sl = np.any((entry_price - low_window)/entry_price >= sl_pct)
    else: 
        hit_tp = np.any((entry_price - low_window)/entry_price >= tp_pct)
        hit_sl = np.any((high_window - entry_price)/entry_price >= sl_pct)

    if hit_tp and not hit_sl:
        strategy_returns.append(tp_pct)
    elif hit_sl and not hit_tp:
        strategy_returns.append(-sl_pct)
    elif hit_tp and hit_sl:
        tp_index = np.argmax((high_window - entry_price)/entry_price >= tp_pct if pred==1 else (entry_price - low_window)/entry_price >= tp_pct)
        sl_index = np.argmax((entry_price - low_window)/entry_price >= sl_pct if pred==1 else (high_window - entry_price)/entry_price >= sl_pct)
        strategy_returns.append(tp_pct if tp_index <= sl_index else -sl_pct)
    else:
        strategy_returns.append(0)  

strategy_returns += [0]*n_days
x_test["StrategyReturn"] = strategy_returns

# --- Cumulative returns ---
x_test["CumulativeReturn"] = (1 + x_test["StrategyReturn"]).cumprod()
print("Final strategy return:", x_test["CumulativeReturn"].iloc[-1] - 1)

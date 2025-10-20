import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = yf.Ticker("TSLA").history(start="2020-01-01", end="2025-01-01")

def create_features(data, target_window: int = 2, 
                    momentum_n: int = 10, 
                    vwap_len: int = 14):
                    
  data["LogReturn"] = np.log(data["Close"]/data["Close"].shift())
  data["Momentum"] = data["Close"] - data["Close"].shift(momentum_n)
  data["Volatility"] = data.LogReturn.rolling(window=14).std()
  data["HL_pct"] = ((data.High - data.Low) / data.High) * 100
  tp = ((data.High + data.Low + data.Close) / 3) * data.Volume
  data["VWAP"] = tp.rolling(window=vwap_len).sum() \
  / data["Volume"].rolling(window=vwap_len).sum()

  data["Target"] = data["Close"].rolling(window=target_window).mean().shift(-1)
  data = data.dropna()
  return data

data = create_features(data)

plt.plot(data["Close"], color="green", label="close")
plt.plot(data["Target"], color="red", label="smoothed close")
plt.show()

x = data[data.columns.difference(["Target"])]
print(x.columns)
y = data["Target"]
input_size = len(x.columns)
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2, 
                                                    shuffle=False)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = np.expand_dims(x_train, axis=1)
x_test = np.expand_dims(x_test, axis=1)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(16, activation="relu", return_sequences=True),
    tf.keras.layers.LSTM(32, activation="relu", return_sequences=False),
    tf.keras.layers.Dense(1, activation="linear")
])

@tf.keras.utils.register_keras_serializable()
def r2_score(y_true, y_pred):
  ss_res = tf.reduce_sum(tf.square(y_true-y_pred))
  ss_tot = tf.reduce_sum(tf.square(y_true-tf.reduce_mean(y_true)))
  return 1 - ss_res/(ss_tot + tf.keras.backend.epsilon())

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.03),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=["mae", "mse", r2_score]
)

model.fit(x_train, y_train, batch_size=16, epochs=20, verbose=1)
model.evaluate(x_test, y_test, verbose = 1)

import pandas as pd
to_plot = pd.DataFrame({"y_test": y_test, "y_pred": y_pred}, index=y_test.index)
to_plot.plot()
plt.show()
y_pred = model.predict(x_test)
verbode = False
y_pred = y_pred.squeeze()
y_actual = y_test


t, p = [], []
print("Predicted vs Actual Prices:")

for i in range(1, 101):
  pred = y_pred[i]
  actual = y_actual[i]
  prev = y_actual[i-1]
  if verbode:
    print(f"Predicted: {pred:.2f}, Actual: {actual:.2f}")
  
  t.append((actual - y_actual[i-1]) > 0)
  p.append((pred - y_pred[i-1]) > 0)

directional_acc = 1 - (abs(sum(t) - sum(p))) / len(t)
print(f"directional acccuracy: %s" %(directional_acc))

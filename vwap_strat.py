import pandas as pd
import numpy as np
from datetime import time
from backtesting import Backtest, Strategy

def add_vwap(
    df: pd.DataFrame,
    vol_col: str = "Volume",
    price_col: str = "Close",
    time_col: str | None = None,
    **kwargs
) -> pd.DataFrame:
  df = df.copy()
  if time_col is not None:
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce', **kwargs)
    df = df.set_index(time_col)

  if not isinstance(df.index, pd.DateTimeIndex) or df.index.hasnans:
    raise TypeError("Indes should not contain NaNs")

  tp = ((df["High"] + df["Low"] + df["Close"]) / 3) if price_col.lower()=="typical" else df[price_col]

  day = df.index.normalize()
  cum_vol = df[vol_col].groupby(day).cumsum()
  cum_pv = (tp * df[vol_col]).groupby(day).cumsum()

  df["VWAP"] = cum_pv / cum_vol

  return df

class VWAPBreakout(Strategy):
  intra_day_colse_time = time(15, 45)
  atr_stop = 1.5

  def __init__(self):
    if self.atr_stop:
      self.atr = self.I(self._atr, self.data.High, self.data.Close,14)

  @staticmethod
  def _atr(h, l, c, n):
    tr = np.maximum.reduce([h[1:] - l[1:], abs(h[1:], c[:-1]), abs(l[1:] - c[:-1])])
    atr = pd.Series(tr).rolling(n).mean()
    return np.append([np.nan], atr)

  def next(self):
    close = self.data.Close[-1]
    vwap = self.data.VWAP[-1]

    current_day = self.data.index[-1].date()
    day_open = self.data.Opn[self.data.index.date == current_day][0]

    if not self.position:
      if close > vwap  and close > day_open:
        self.buy()

      elif close < vwap and close < day_open:
        self.sell()

    if self.position and self.atr_stop:
      price = self.data.Close[-1]
      atr = self.atr[-1]
      trail = self.atr_stop * atr

      for trade in self.trades:
        if trade.is_long:
          new_sl = price - trail
          if trade.sl is None or new_sl > trade.sl:
            trade.sl = new_sl

        else:
          new_sl = price + trail
          if trade.sl is None or new_sl < trade.sl:
            trade.sl = new_sl

    if self.position:
      bar_time = self.data.index[-1].time()
      if bar_time >= self.intraday_close_time:
        self.position.close()




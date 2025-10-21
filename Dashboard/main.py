import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

class IBApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False
        self.hist_data = {}
        
    def error(self, reqid, errorCode, errorString, *args):
        if errorCode == 2176 and "fractional share" in errorString.lower():
            return
        
        print(f"ReqID: {reqid} | Error {errorCode} | MSG: {errorString}")
        if args:
            print(f"Additional information: {args}")
            
    def next_valid_id(self, orderId):
        self.connected = True
        print("Connected to IB")
        
    def historical_data(self, reqId, bar):
        if reqId not in self.hist_data:
            self.hist_data[reqId] = []
            
        self.hist_data[reqId].append({
            "date": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume
        })
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
        
    def historical_data_end_fn(self, reqId, start, end):
        print(f"Historical data received for reqId{reqId}")
        
        
class EarningsDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Earnings Trading Dashboard - IV Curush Analysis")
        self.root.geometry("1600x1000")
        
        self.stock_data = None
        self.vix_data = None
        self.earninfs_data = None
        self.ticker = None
        self.iv_data = None
        
        self.ib_app = IBApp()
        self.conected_flag = False
        
        self.risk_free_rate = 0.05
        
        self.ax1_twin = None
        
        self.setup_ui()
        
    def create_equity_contract(self, symbol):
        contract = Contract()
        contract.symbol = symbol.upper()
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"   
        
        return contract
    
    def create_vix_contract(self):
        contract = Contract()
        contract.symbol = "VIX"
        contract.secType = "IDX"
        contract.exchange = "CBOE"
        contract.currency = "USD"   
        
        return contract
        
    def setup_ui(self):
        main_frame = ttk.frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weiht=1)
        self.root.rowconfigure(0, weight=1)
        
        main_frame.columnconfigure(0, weiht=1)
        main_frame.rowconfigure(0, weight=1)
        
        conn_frame = ttk.LabelFrame(main_frame, text="Interactive Brokers Connection", padding="5")
        conn_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(conn_frame, text="Host: ")
        

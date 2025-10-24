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
        
        ttk.Label(conn_frame, text="Host:").grid(row=0, column=0, padx=(0, 5))
        self.host_var = tk.StringVar(value="127.0.0.1")
        ttk.Entry(conn_frame, textvariable=self.host_var, width=15).grid(row=0, column=1, padx=(0, 10))
        
        ttk.Label(conn_frame, text="Port:").grid(row=0, column=2, padx=(0, 5))
        self.port_var = tk.StringVar(value="7497")
        ttk.Entry(conn_frame, textvariable=self.port_var, width=15).grid(row=0, column=3, padx=(0, 10))
        
        self.connect_btm = ttk.Button(conn_frame, text='Connect', command=self.connect_ib)
        self.connect_btm.grid(row=0, column=4, padx=(0, 10))
        
        self.disconnect_btm = ttk.Button(conn_frame, text='Disconnect', command=self.disconnect_ib)
        self.disconnect_btm.grid(row=0, column=5, padx=(0, 10), state="disabled")
        
        earnings_frame = ttk.LabelFrame(main_frame, text="Earnings Analysis setup", padding = "5")
        earnings_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(earnings_frame, text="Ticker:").grid(row=0, column=0, padx=(0, 5))
        self.ticker_var = tk.StringVar(value="TSLA")
        ttk.Entry(earnings_frame, textvariable=self.ticker_var, width=10).grid(row=0, column=1, padx=(0, 10))
        
        ttk.Label(earnings_frame, text="Earnings Date:").grid(row=0, column=2, padx=(0, 5))
        self.date_var = tk.StringVar(value="2025-10-22")
        ttk.Entry(earnings_frame, textvariable=self.date_var, width=12).grid(row=0, column=3, padx=(0, 10))
        
        ttk.Label(earnings_frame, text="Days to Expiration:").grid(row=0, column=4, padx=(0, 5))
        self.days_to_exp_var = tk.StringVar(value="30")
        ttk.Entry(earnings_frame, textvariable=self.days_to_exp_var, width=12).grid(row=0, column=5, padx=(0, 10))
        
        self.analyze_btn = ttk.Button(earnings_frame, text="Analyze IV Crush", command=self.analyze_iv_crush, state="disabled")
        self.analyze_btn.grid(row=0, column=6)
        
        metrics_frame = ttk.LabelFrame(main_frame, text="Current Metrics", padding="5")
        metrics_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Label(metrics_frame, text="Stock Price").grid(row=0, column=0, padx=(0, 5))
        self.stock_price_label = ttk.Label(metrics_frame, text="N/A", font=("Arial", 10, "bold"))
        self.stock_price_label.grid(row=0, column=1, padx=(0, 20))
        
        ttk.Label(metrics_frame, text="VIX Level").grid(row=0, column=2, padx=(0, 5))
        self.vix_level_label = ttk.Label(metrics_frame, text="N/A", font=("Arial", 10, "bold"))
        self.vix_level_label.grid(row=0, column=3, padx=(0, 20))
        
        ttk.Label(metrics_frame, text="Current IV").grid(row=0, column=4, padx=(0, 5))
        self.curr_iv_label = ttk.Label(metrics_frame, text="N/A", font=("Arial", 10, "bold"))
        self.curr_iv_label.grid(row=0, column=5, padx=(0, 20))
        
        crush_frame = ttk.LabelFrame(main_frame, text="IV Crush Analysis", padding="5")
        crush_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        ttk.Label(crush_frame, text="Pre Event IV").grid(row=0, column=0, padx=(0, 5))
        self.stock_price_label = ttk.Label(crush_frame, text="N/A", font=("Arial", 10, "bold"))
        self.stock_price_label.grid(row=0, column=1, padx=(0, 20))
        
        ttk.Label(crush_frame, text="IV Crush %").grid(row=0, column=2, padx=(0, 5))
        self.vix_level_label = ttk.Label(crush_frame, text="N/A", font=("Arial", 10, "bold"))
        self.vix_level_label.grid(row=0, column=3, padx=(0, 20))
        
        ttk.Label(crush_frame, text="Current IV").grid(row=0, column=4, padx=(0, 5))
        self.curr_iv_label = ttk.Label(crush_frame, text="N/A", font=("Arial", 10, "bold"))
        self.curr_iv_label.grid(row=0, column=5, padx=(0, 20))
        
        

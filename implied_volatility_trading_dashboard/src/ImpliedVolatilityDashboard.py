import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ibapi.contract import Contract
from datetime import datetime
import warnings
from utils import DisplayUtils
from IBApp import IBApp
from scipy import stats

warnings.filterwarnings('ignore')

class ImpliedVolatilityDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Implied Vol Dashboard")
        self.root.geometey("1400x1200")
        
        self.option_data = None
        self.volatility_data = None
        self.current_implied_volatility = None
        
        self.ib_app = IBApp()
        self.connected = False
        
        self.vol_annualization = 252
        
        self.setup_ui()
        
    def create_equity_contract(self, symbol):
        contract = Contract()
        contract.symbol = symbol.upper()
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        return contract
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding = "10")
        main_frame.grid(row = 0, column = 0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columncofig(0, weight=1)
        self.root.rowcofig(0, weight=1)
        
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(5, weight=1)
        
        conn_frame = ttk.LabelFrame(main_frame, text ="IB connection", padding="10")
        conn_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady = (0, 10))
        
        ttk.Label(conn_frame, text="Host:").grid(row=0, column=0, padx=(0, 5))
        self.host_var = tk.StringVar(value="127.0.0.1")
        ttk.Entry(conn_frame, textvariable=self.host_var, width=15).grid(row=0, column=1, padx=(0, 10))
        
        ttk.Label(conn_frame, text="Port:").grid(row=0, column=0, padx=(0, 5))
        self.port_var = tk.StringVar(value="7497")
        ttk.Entry(conn_frame, textvariable=self.port_var, width=10).grid(row=0, column=3, padx=(0, 10))
        
        self.connect_btn = ttk.Button(conn_frame, text="Connect",command=self.connect_ib)
        self.connect_btn.grid(row=0, column=4, padx=(0, 10))
        
        self.disconnect_btn = ttk.Button(conn_frame, text="Connect",command=self.disconnect_ib,state="disabled")
        self.disconnect_btn.grid(row=0, column=5, padx=(0, 10))
        
        data_frame = tk.LabelFrame(main_frame, text="Data Querry", paddinx="5")
        data_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(data_frame, text="Symbol:").grid(row=0, column=0, padx=(0, 5))
        self.symbol_var = tk.StringVar(value="SPY")
        ttk.Entry(data_frame, textvariable=self.symbol_var, width=10).grid(row=0, column=1, padx=(0, 10))
        
        ttk.Label(data_frame, text="Duration:").grid(row=0, column=2, padx=(0, 5))
        self.duration_var = tk.StringVar(value="2 Y")
        ttk.Entry(data_frame, textvariable=self.duration_var, width=10).grid(row=0, column=3, padx=(0, 10))
        
        self.query_btn = ttk.Button(data_frame, text="Querry IB data",command=self.query_data, state="disabled")
        self.query_btn.grid(row=0, column=4, padx=(0, 10))
        
        self.analyze_btn = ttk.Button(data_frame, text="Analyze IV",command=self.analyze_volatility,state="disabled")
        self.analyze_btn.grid(row=0, column=5, padx=(0, 10))
        
        vol_frame = ttk.LabelFrame(main_frame, text="Current IV", padding="5")
        vol_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(vol_frame, text="Current IV").grid(row=0, column=0, padx=(0, 5))
        self.current_vol_label = ttk.Label(vol_frame, text="N/A", font=("Arial", 12, "bold"))
        self.current_vol_label.grid(row=0, column=3, padx=(0, 20))
        
        ttk.Label(vol_frame, text="Computation").grid(row=0, column=2, padx=(0, 5))
        self.computation_label = ttk.Label(vol_frame, text="No data", font=("Arial", 10))
        self.computation_label.grid(row=0, column=1, padx=(0, 10))    
        
        ttk.Label(vol_frame, text="Vol Range").grid(row=0, column=4, padx=(0, 5))
        self.vol_range_label = ttk.Label(vol_frame, text="N/A", font=("Arial", 10))
        self.vol_range_label.grid(row=0, column=5)    
        
        regieme_frame = ttk.LabelFrame(main_frame, text = "Volatility Regieme adjusting", padding="5")
        regieme_frame.grid(row=3, column=2, columnspan=2, sticky=(tk.W, tk.E), pady = (0, 10))
        
        ttk.Label(regieme_frame, text="Current Regieme").grid(row=0, column=4, padx=(0, 5))
        self.regieme_label = ttk.Label(regieme_frame, text="N/A", font=("Arial", 12, "bold"))
        self.regieme_label.grid(row=0, column=1, padx = (0,20))    
        
        ttk.Label(regieme_frame, text="Percentile").grid(row=0, column=4, padx=(0, 5))
        self.percentile_label = ttk.Label(regieme_frame, text="N/A", font=("Arial", 10))
        self.percentile_label.grid(row=0, column=3, padx = (0,20))    
        
        ttk.Label(regieme_frame, text="MR Signal").grid(row=0, column=4, padx=(0, 5))
        self.reversion_label = ttk.Label(regieme_frame, text="N/A", font=("Arial", 10))
        self.reversion_label.grid(row=0, column=5, padx = (0,20))    
         
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="5")
        status_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.status_txt = scrolledtext.ScrolledText(status_frame, height=6, width=80)
        self.status_txt.grid(row=0, column=0, sticky=(tk.W, tk.E))
        status_frame.columnconfigure(0, weight=1)
        
        plot_frame = ttk.LabelFrame(main_frame, text="Implied Volatility, Analysis Results", padding="5")
        plot_frame.grid(row=5, column=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(18, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky=(tk.N, tk.W, tk.E, tk.S)) 
        
        
    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.ststus_text.see(tk.END)
        self.root.update_idletasks()
            
    def connect_ib(self):
        try: 
            host = self.host_var.get()
            port = self.port_var.get()
            
            self.log_message(f"Conneting to IB at {host}:{port}")
            
            def connect_thread():
                try:
                    self.ib_app.connect(host, port, 0)
                    self.ib_app.run()
                
                except Exception as e:
                    self.log_message(f"connection error {e}")
            
            thread = threading.Thread(target=connect_thread, daemon=True)
            thread.start()
            
            for _ in range(50):
                if self.ib_app.connected:
                    break
                
                time.sleep(0.1)
            
            if self.ib_app.connected:
                self.connected = True
                self.connect_btn.config(state="disabled")
                self.disconnect_btn.config(state="normal")
                self.query_btn.config(state="normal")
                self.log_message("sucsessfully connected to IBTWS")
                
            else:
                self.log_message("failed to connect to IBTWS")
            
        except Exception as e:
            self.log_message(f"connection error: {e}")
            
    def disconect_ib(self):
        try:
            self.ib_app.disconnect()
            self.connect = False
            self.connect_btn.config(state="normal")
            self.disconnect_btn.config(state="disabled")
            
            self.querry_btn.config(state="disabled")
            self.analyze_btn.config(state="disabled")
            
            self.current_implied_vol = None
            self.update_current_vol_display()
            
            self.log_message("Disconnected from IBTWS") 
            
        except Exception as e:
            self.log_message(f"Disconnect Error: {e}") 
            
            
    def query_data(self):
        if not self.connected:
            messagebox.showerror("Error", "Not Connected to IBTWS")
            
        symbol = self.symbol_var.get().upper()
        duration = self.duration_var.get()
        
        self.log_message(f"querying implied volatility for {symbol}")
        
        self.ib_app.historical_data.clear()
        
        contract = self.create_equity_contract(symbol)
        
        self.ib_app.reqHistoricalData(
            reqId = 1,
            contract = contract,
            endDateTime="",
            duration=duration,
            barSizeSetting="1 day",
            whatToShow = "OPTION_IMPLIED_VOLATILITY",
            useRTH=1,
            formDate=1,
            keepUpToDate=False,
            chartOption=[]
        )
        
        timeout = 15
        start_time = time.time()
        
        while 1 not in self.ib_app.historical_data and (time.time() - start_time) < timeout:
            time.sleep(0.1)
            
        if 1 in self.ib_app.historical_data:
            data = self.ib_app.historical_data[1]
            if len(data) > 0:
                self.equity_data = pd.DataFrame(data)
                self.equity_data["date"] = pd.to_dataframe(self.equity_data["date"])   
                self.equity_data.set_index("date", inplace = True)
                
                self.equity_data["implied_vol"] = self.equity_data["close"]
                
                self.log_message(f"Recieved {len(self.equity_data)}, implied volatility points for {symbol}")
                self.log_message(f"Date Range:  {self.equity_data.index.min()} : {self.equity_data.index.max()}")
                self.log_message("NOTE: ALL VALUES ARE ANNUALIZED")
                
                self.process_implied_volatility()
                self.analyze_btn.config(state="normal")
                
            else:
                self.log_message("No vol data recieved")
                self.equity_data = None
                
        else:
            self.log_message("NOTE: No vol data recieved - MAY NOT BE AVAILABLE FOR SYMBOL")
            self.equity_data = None
    
    
    def process_implied_volatility(self):
        if self.equity_data is None:
            return
        
        self.log_message("Processing iVol Data")
        
        self.equity_data["implied_vol"] = self.equity_data["close"] * np.sqrt(self.vol_annualization)
        
        self.equity_data["iv_percentile"] = self.equity_data["implied_vol"].rolling(window=252).rank(pct=True)
        self.current_implied_vol = self.equity_data["implied_vol"].iloc[-1] if len(self.equity_data) > 0 else None
        
        self.volatility_data = self.equity_data[["implied_vol", "iv_percentile"]].copy()
        
        self.update_current_vol_display(self)
        
        if self.current_implied_vol is not None:
            self.log_message(f"Current IVol: {self.current_implied_vol:.4f} ({self.current_implied_vol*100:.2f}%)")
            self.log_message(f"IV_range: {self.equity_data["implied_vol"].min():.4f} - {self.equity_data["implied_vol"].max():.4f}")
        else:
            self.log_message("failed to process IVol data")
            
            
    def update_current_vol_display(self):
        if self.current_implied_vol is not None:
            self.current_vol_label.config(f"{self.current_implied_vol*100:.4f}%")
            comp_text = "Annualized by root 252 factor"
            self.vol_computation_label.config(text = comp_text)
            
            if self.equity_data is not None:
                vol_min = self.equity_data["implied_vol"].min()
                vol_max = self.equity_data["implied_vol"].max()
                vol_mean = self.equity_data["implied_vol"].mean()
                range_text = f"MIN: {vol_min:.3f}| MAX: {vol_max:.3f}| MEAN: {vol_mean:.3f}"
            else:
                range_text = "N/A"
                
            self.vol_range_label.config(text=range_text)
            
            if self.current_implied_vol >= 0.4:
                self.current_vol_label.config(foreground="red")
            if self.current_implied_vol <= 0.15:
                self.current_vol_label.config(foreground="green")
            else:
                self.current_vol_label.config(foreground="black")
                
            self.update_regime_analysis()
            
        else:
            self.current_vol_label.config(text="N/A", foreground="black")
            self.vol_computation_label.config(text="No Data")
            self.vol_range_label.config(text="N/A", foreground="black")
            self.regieme_label.config(text="N/A", foreground="black")
            self.percentile_label.config(text="N/A", foreground="black")
            self.reversion_label.config(text="N/A", foreground="black")

             
    def update_regime_analysis(self):
        if self.volatility_data is None or self.current_implied_vol is None:
            return
        
        current_percentile = self.volatility_data["iv_percentile"].iloc[-1]
        
        if current_percentile > .8:
            regieme = "HIGH iVOL"
            color = "red"
            
        elif current_percentile > 0.6:
            regieme = "ABOVE AVG iVOL"
            color = "orange"
            
        elif current_percentile >.4:
            regieme ="NOMAL iVOL"
            color = "black"
            
        elif current_percentile > .2:
            regieme ="BELOW AVG iVOL"
            color = "blue"
        else:
            regieme = "LOW iVOL"
            color = "green"
        
        self.regieme_label.config(text=regieme, foreground=color)
        self.percentile_label.config(text=f"{current_percentile:.1f}%")
        
        if current_percentile > .8:
            rev = "Expect MR down"
            rcolor = "red"
            
        elif current_percentile > .8:
            rev = "Expect MR up"
            rcolor = "green"
            
        else:
            rev = "No reversion expected"
            rcolor = "black"
        
        self.regieme_label.config(text=rev, foreground=rcolor)
        
         
    def analyze_volatility(self):
        if self.equity_data is None or self.volatility_data is None:
            messagebox.showerror("Error", "No iVol data is available for analysis")
            return
        
        self.log_message("Analyze iVol data")
        
        vol_forward_30d = self.volatility_data["implied_vol"].rolling(window=30, min_periods=1).mean().shift(-30)
        
        analysis_df = pd.DataFrame({
            "current_vol": self.volatility_data["implied_vol"],
            "forward_vol": vol_forward_30d,
            "vol_diff": vol_forward_30d - self.volatility_data["implied_vol"],
            "vol_percentile": self.volatility_data["iv_percentile"]
        })
        analysis_df.dropna(inplace=True)
        
        if len(analysis_df) <= 30:
            self.log_message("Insufficient iVol data for Analysis")
            
        slope1, intercept1, r_value1, p_value1, std_error1 = stats.linregress(
            analysis_df["current_vol"], analysis_df["forward_vol"]
        ) 
        
        slope2, intercept2, r_value2, p_value2, std_error2 = stats.linregress(
            analysis_df["current_vol"], analysis_df["vol_ciff"]
        ) 
        
        if slope1 != 1:
            intersection_x = intercept1 / (1-slope1)
        else:
            intersection_x = analysis_df["current_vol"].median()
            
        
        

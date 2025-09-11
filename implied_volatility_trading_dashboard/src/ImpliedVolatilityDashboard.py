import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
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

class ImpliedVolatilityDashboard(DisplayUtils):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.root.title("Implied Vol Dashboard")
        self.root.geometry("1200x800")
        
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
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(5, weight=1)
        
        conn_frame = ttk.LabelFrame(main_frame,
                                    text ="IB connection",
                                    padding="10")
        conn_frame.grid(row=0, column=0, columnspan=2,
                        sticky=(tk.W, tk.E), pady = (0, 10))
        
        ttk.Label(conn_frame, text="Host:").grid(row=0, column=0, padx=(0, 5))
        self.host_var = tk.StringVar(value="127.0.0.1")
        ttk.Entry(conn_frame, textvariable=self.host_var, width=15)\
            .grid(row=0, column=1, padx=(0, 10))
        
        ttk.Label(conn_frame, text="Port:").grid(row=0, column=0, padx=(0, 5))
        self.port_var = tk.StringVar(value="7497")
        ttk.Entry(conn_frame, textvariable=self.port_var, width=10) \
            .grid(row=0, column=3, padx=(0, 10))
        
        self.connect_btn = ttk.Button(conn_frame, text="Connect",
                                      command=self.connect_ib)
        self.connect_btn.grid(row=0, column=4, padx=(0, 10))
        
        self.disconnect_btn = ttk.Button(conn_frame, text="Disconnect",
                                         command=self.disconnect_ib,state="disabled")
        self.disconnect_btn.grid(row=0, column=5, padx=(0, 10))
        
        data_frame = ttk.LabelFrame(main_frame, text="Data Querry", padding="5")
        data_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(data_frame, text="Symbol:").grid(row=0, column=0, padx=(0, 5))
        self.symbol_var = tk.StringVar(value="SPY")
        ttk.Entry(data_frame, textvariable=self.symbol_var, width=10) \
            .grid(row=0, column=1, padx=(0, 10))
        
        ttk.Label(data_frame, text="Duration:").grid(row=0, column=2, padx=(0, 5))
        self.duration_var = tk.StringVar(value="2 Y")
        ttk.Entry(data_frame, textvariable=self.duration_var, width=10) \
            .grid(row=0, column=3, padx=(0, 10))
        
        self.query_btn = ttk.Button(data_frame, text="Querry IB data",
                                    command=self.query_data, state="disabled")
        self.query_btn.grid(row=0, column=4, padx=(0, 10))
        
        self.analyze_btn = ttk.Button(data_frame, text="Analyze IV",
                                      command=self.analyze_volatility,state="disabled")
        self.analyze_btn.grid(row=0, column=5, padx=(0, 10))
        
        vol_frame = ttk.LabelFrame(main_frame, text="Current IV", padding="5")
        vol_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(vol_frame, text="Current IV").grid(row=0, column=0, padx=(0, 5))
        self.current_vol_label = ttk.Label(vol_frame, text="N/A",
                                           font=("Arial", 12, "bold"))
        self.current_vol_label.grid(row=0, column=3, padx=(0, 20))
        
        ttk.Label(vol_frame, text="Computation").grid(row=0, column=2, padx=(0, 5))
        self.computation_label = ttk.Label(vol_frame, text="No data", font=("Arial", 10))
        self.computation_label.grid(row=0, column=1, padx=(0, 10))    
        
        ttk.Label(vol_frame, text="Vol Range").grid(row=0, column=4, padx=(0, 5))
        self.vol_range_label = ttk.Label(vol_frame, text="N/A", font=("Arial", 10))
        self.vol_range_label.grid(row=0, column=5)    
        
        regieme_frame = ttk.LabelFrame(main_frame,
                                       text = "Volatility Regieme adjusting",
                                       padding="5")
        regieme_frame.grid(row=3, column=2, columnspan=2, sticky=(tk.W, tk.E), pady = (0, 10))
        
        ttk.Label(regieme_frame, text="Current Regieme").grid(row=0, column=4, padx=(0, 5))
        self.regieme_label = ttk.Label(regieme_frame,
                                       text="N/A",
                                       font=("Arial", 12, "bold"))
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
        
        plot_frame = ttk.LabelFrame(main_frame, 
                                    text="Implied Volatility, Analysis Results", 
                                    padding="5")
        plot_frame.grid(row=5, column=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(18, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, 
                                        sticky=(tk.N, tk.W, tk.E, tk.S)) 
            
    
    
    def process_implied_volatility(self):
        if self.equity_data is None:
            return
        
        self.log_message("Processing iVol Data")
        
        self.equity_data["implied_vol"] = self.equity_data["close"] \
        * np.sqrt(self.vol_annualization)
        
        self.equity_data["iv_percentile"] = self.equity_data["implied_vol"] \
            .rolling(window=252).rank(pct=True)
        self.current_implied_vol = self.equity_data["implied_vol"] \
        .iloc[-1] if len(self.equity_data) > 0 else None
        
        self.volatility_data = self.equity_data[["implied_vol", "iv_percentile"]].copy()
        
        self.update_current_vol_display(self)
        
        if self.current_implied_vol is not None:
            self.log_message(f"Current IVol: {self.current_implied_vol:.4f} \
                ({self.current_implied_vol*100:.2f}%)")
            self.log_message(f"IV_range: {self.equity_data["implied_vol"].min():.4f} \
                             - {self.equity_data["implied_vol"].max():.4f}")
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
                range_text = f"MIN: {vol_min:.3f}| \
                    MAX: {vol_max:.3f}| \
                    MEAN: {vol_mean:.3f}"
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
            messagebox.showerror("Error", 
                                 "No iVol data is available for analysis")
            return
        
        self.log_message("Analyze iVol data")
        
        vol_forward_30d = self.volatility_data["implied_vol"] \
        .rolling(window=30, min_periods=1).mean().shift(-30)
        
        analysis_df = pd.DataFrame({
            "current_vol": self.volatility_data["implied_vol"],
            "forward_vol": vol_forward_30d,
            "vol_diff": vol_forward_30d - self.volatility_data["implied_vol"],
            "vol_percentile": self.volatility_data["iv_percentile"]
        })
        analysis_df.dropna(inplace=True)
        
        if len(analysis_df) <= 30:
            self.log_message("Insufficient iVol data for Analysis")
            
        self.slope1, self.intercept1, self.r_value1, self.p_value1, self.std_error1 = stats.linregress(
            analysis_df["current_vol"], analysis_df["forward_vol"]
        ) 
        
        self.slope2, self.intercept2, self.r_value2, self.p_value2, self.std_error2 = stats.linregress(
            analysis_df["current_vol"], analysis_df["vol_ciff"]
        ) 
        
        if self.slope1 != 1:
            intersection_x = self.intercept1 / (1-self.slope1)
        else:
            intersection_x = analysis_df["current_vol"].median()
            
        self.high_vol_regieme = analysis_df["current_vol"] > intersection_x
        self.low_vol_regieme = analysis_df["current_vol"] <= intersection_x
        
        if self.high_vol_regieme.sum() > 10:
            self.slope_high, self.intercept_high, self.r_value_high, self.p_value_high, self.std_error_high = stats.linregress(
                analysis_df.loc[self.high_vol_regieme, "current_vol"], analysis_df.loc[self.high_vol_regieme, "vol_diff"]
            )   
        else:
            self.slope_high = self.intercept_high = self.r_value_high = self.p_value_high = self.td_error_high = None
            
        
        if self.low_vol_regieme.sum() > 10:
            self.slope_low, self.intercept_low, self.r_value_low, self.p_value_low, self.std_error_low = stats.linregress(
                analysis_df.loc[self.low_vol_regieme, "current_vol"], analysis_df.loc[self.low_vol_regieme, "vol_diff"]
            ) 
        else:
            self.slope_low, self.intercept_low, self.r_value_low, self.p_value_low, self.std_error_low = (None,)*5
            
        self.plot_analysis(analysis_df)
            
    def plot_analysis(self,analysis_df): 
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        self.ax1.scatter(analysis_df['current_vol'], analysis_df['forward_vol'], alpha=.6, s=20)
        
        x_range = np.linspace(analysis_df['current_vol'].min(), analysis_df['current_vol'].max(), 100)
        y_pred1 = self.slope1 * x_range + self.intercept1
        
        self.ax1.plot(x_range, y_pred1, "r-", linewidth=2, label=f"Regression R**2 = {self.r_value1**2:.3f}")
        
        min_val = min(analysis_df['current_vol'].min(), analysis_df['forward_vol'].min())
        max_val = max(analysis_df['current_vol'].max(), analysis_df['forward_vol'].max())
        
        self.ax1.plot([min_val, max_val], "k--", "y=x")
        
        self.ax1.set_xlabel("current implied vol")
        self.ax1.set_ylabel("30-days forward average iVol")
        self.ax1.title(f'forward iVol vs current iVOl \n \
            y = {self.slope1:.3f}x + {self.intercept1}, R^2 = {self.r_value1**2:.3f} ')
        self.ax1.legend()
        self.ax1.grid(True, alpha=.3)
        
        self.ax2.scatter(analysis_df.loc[self.high_vol_regieme, "current_vol"], analysis_df.loc[self.high_vol_regieme, "vol_diff"],
                         alpha=.6, color="red", s=20, label="High vol regieme")
        self.ax2.scatter(analysis_df.loc[self.low_vol_regieme, "current_vol"], analysis_df.loc[self.low_vol_regieme, "vol_diff"], 
                         alpha=.6, color="green", s=20, label="Low vol regieme")
        
        if self.slope_high is not None:
            x_high = analysis_df.loc[self.high_vol_regieme, "current_vol"]
            if len(x_high) > 0:
                x_range_high = np.linspace(x_high.min(), x_high.max, 100)
                y_pred_high = self.slope_high * x_range_high + self.intercept_high
                
                self.ax2.plot(x_range_high, y_pred_high, "r-", linewidth=2,
                              label=f"High col R^2 = {self.r_value_high**2:.3f}")
                
        if self.slope_low is not None:
            x_low = analysis_df.loc[self.low_vol_regieme, "current_vol"]
            if len(x_low) > 0:
                x_range_low = np.linspace(x_low.min(), x_low.max, 100)
                y_pred_low = self.slope_low * x_range_low + self.intercept_low
                
                self.ax2.plot(x_range_low, y_pred_low, "r-", linewidth=2,
                              label=f"High col R^2 = {self.r_value_low**2:.3f}")
                
        self.ax2.axhline(y=0, color="k", linestile="--", linewidt = 1, alpha=.7, label="No Change y=0")  
        self.ax2.axvline(x=self.intersection_x, color="g", linestile=":", linewidth=1, alpha=.7, 
                         label=f"regieme split (VOL={self.intersection_x:.3f})")
        
        self.ax2.set_xlabel("current implied volatility")
        self.ax2.set_ylabel("Vol difference (Forward - Current)")
        self.ax2.title("Vol Diff vs Curr Vol")        
        self.ax2.legend()
        self.ax2.grid(True, alpha =.3)

        self.ax3.plot(self.volatility_data.index, self.volatility_data["implied_vol"],
                      label="implied volatility", linewidth = 1)
        
        vol_75th = self.volatility_data["implied_vol"].quantile(0.75)
        vol_25th = self.volatility_data["implied_vol"].quantile(0.25)
        
        self.ax3.axhline(y=vol_75th, color="red",
                         linestile="--",alpha=.7,
                         label="75th Percentile")
        
        self.ax3.axhline(y=vol_25th, color="green",
                         linestile="--", alpha=.7,
                         label="25th Percentile")
        
        self.ax3.axhline(y=self.volatility_data["implied_vol"].mean(),
                         color="gray",linestile="--", alpha=.7,label="mean")
        
        if self.current_implied_vol is not None:
            self.ax3.scatter(self.current_implied_vol[-1],
                             self.current_implied_vol,
                             color="red", s=100, zorder=5, label="Current iVOl")
        self.ax3.set_xlabel("Date")
        self.ax3.set_ylabel("Implied VOlatility")   
        self.ax3.set_title("Time Series with regime bands")
        
        self.ax3.legend()
        self.ax3.grid(True, alpha=.3)
        self.ax3.tick_params(axis="x", rotation=45)
        
        self.fig.tight_layout()
        
        self.canvas.draw()
        
        
        
        

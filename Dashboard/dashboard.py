from tkinter import ttk, messagebox, scrolledtext
import threading
from ibapi.contract import Contract
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta
from scipy.stats import norm
import tkinter as tk
from ibapp import IBApp


class EarningsDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Earnings Trading Dashboard - IV Curush Analysis")
        self.root.geometry("1600x1000")
        
        self.stock_data = None
        self.vix_data = None
        self.earnings_date = None
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
        crush_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), paddy=(0, 10))
        
        ttk.Label(crush_frame, text="Pre-Earnings IV").grid(row=0, column=0, padx=(0, 5))
        self.pre_iv_label = ttk.Label(crush_frame, text="N/A", font=("Arial", 10, "bold"))
        self.pre_iv_label.grid(row=0, column=1, padx=(0, 20))
        
        ttk.Label(crush_frame, text="Pos-Earnings IV").grid(row=0, column=2, padx=(0, 5))
        self.post_iv_label = ttk.Label(crush_frame, text="N/A", font=("Arial", 10, "bold"))
        self.post_iv_label.grid(row=0, column=3, padx=(0, 20))
        
        ttk.Label(crush_frame, text="IV Crush %").grid(row=0, column=4, padx=(0, 5))
        self.iv_crush_label = ttk.Label(crush_frame, text="N/A", font=("Arial", 10, "bold"), foreground = "red")
        self.iv_crush_label.grid(row=0, column=5, padx=(0, 20))
        
        spot_strike_frame = ttk.LabelFrame(main_frame, text='Spot vs Strike Analysis', padding="5")
        spot_strike_frame.grid(row=4,column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(spot_strike_frame, text="Strike Price").grid(row=0, column=0, padx=(0, 5))
        self.strike_price_label = ttk.Label(crush_frame, text="N/A", font=("Arial", 10, "bold"))
        self.strike_price_label.grid(row=0, column=5, padx=(0, 20))
        
        ttk.Label(spot_strike_frame, text="Pre-Earnings Close").grid(row=0, column=2, padx=(0, 5))
        self.pre_spot_label = ttk.Label(crush_frame, text="N/A", font=("Arial", 10, "bold"))
        self.pre_spot_label.grid(row=0, column=5, padx=(0, 20))
        
        ttk.Label(spot_strike_frame, text="Post-Earnings Spot(Next day average)").grid(row=0, column=4, padx=(0, 5))
        self.post_spot_label = ttk.Label(crush_frame, text="N/A", font=("Arial", 10, "bold"))
        self.post_spot_label.grid(row=0, column=5, padx=(0, 20))
        
        option_frame = ttk.LabelFrame(main_frame, text='ATM Straddle Pricing & P/L ', padding="5")
        option_frame.grid(row=5,column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(option_frame, text="Pre-Earnings Call").grid(row=0, column=0, padx=(0, 5))
        self.pre_call_label = ttk.Label(crush_frame, text="N/A", font=("Arial", 10, "bold"))
        self.pre_call_label.grid(row=0, column=5, padx=(0, 20))
        
        ttk.Label(option_frame, text="Post-Earnings Call").grid(row=0, column=2, padx=(0, 5))
        self.post_call_label = ttk.Label(crush_frame, text="N/A", font=("Arial", 10, "bold"))
        self.post_call_label.grid(row=0, column=5, padx=(0, 20))
        
        ttk.Label(option_frame, text="Call Change").grid(row=0, column=4, padx=(0, 5))
        self.call_loss_label = ttk.Label(crush_frame, text="N/A", font=("Arial", 10, "bold"))
        self.call_loss_label.grid(row=0, column=5, padx=(0, 20))
        
        ttk.Label(option_frame, text="Pre-Earnings Put").grid(row=1, column=0, padx=(0, 5))
        self.pre_put_label = ttk.Label(crush_frame, text="N/A", font=("Arial", 10, "bold"))
        self.pre_put_label.grid(row=1, column=5, padx=(0, 20))
        
        ttk.Label(option_frame, text="Post-Earnings Put").grid(row=1, column=2, padx=(0, 5))
        self.post_put_label = ttk.Label(crush_frame, text="N/A", font=("Arial", 10, "bold"))
        self.post_put_label.grid(row=1, column=5, padx=(0, 20))
        
        ttk.Label(option_frame, text="Put Change").grid(row=1, column=4, padx=(0, 5))
        self.put_loss_label = ttk.Label(crush_frame, text="N/A", font=("Arial", 10, "bold"))
        self.put_loss_label.grid(row=1, column=5, padx=(0, 20))
        
        ttk.Label(option_frame, text="Pre-Earnings Straddle").grid(row=2, column=0, padx=(0, 5))
        self.pre_straddle_label = ttk.Label(crush_frame, text="N/A", font=("Arial", 10, "bold"))
        self.pre_straddle_label.grid(row=2, column=5, padx=(0, 20))
        
        ttk.Label(option_frame, text="Post-Earnings Straddle").grid(row=2, column=2, padx=(0, 5))
        self.post_straddle_label = ttk.Label(crush_frame, text="N/A", font=("Arial", 10, "bold"))
        self.post_straddle_label.grid(row=2, column=5, padx=(0, 20))
        
        ttk.Label(option_frame, text="Straddle Change").grid(row=2, column=4, padx=(0, 5))
        self.straddle_loss_label = ttk.Label(crush_frame, text="N/A", font=("Arial", 10, "bold"))
        self.straddle_loss_label.grid(row=2, column=5, padx=(0, 20))
        
        ttk.Label(option_frame, text="LONG Straddle P/L").grid(row=3, column=0, padx=(0, 5))
        self.long_pl_label = ttk.Label(crush_frame, text="N/A", font=("Arial", 10, "bold"))
        self.long_pl_label.grid(row=3, column=1, columnspan=2, padx=(0, 20))
        
        ttk.Label(option_frame, text="SHORT Straddle P/L").grid(row=3, column=3, padx=(0, 5))
        self.short_pl_label = ttk.Label(crush_frame, text="N/A", font=("Arial", 10, "bold"))
        self.short_pl_label.grid(row=3, column=4, columnspan=2, padx=(0, 20))
        
        
        greek_frame = ttk.LabelFrame(main_frame, text="Greek Analysis", padding="5")
        greek_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), paddy=(0, 10))
        
        ttk.Label(greek_frame, text="Pre-Earnings Delta").grid(row=0, column=0, padx=(0, 5))
        self.pre_delta_label = ttk.Label(crush_frame, text="N/A", font=("Arial", 10, "bold"))
        self.pre_delta_label.grid(row=0, column=0, padx=(0, 20))
        
        ttk.Label(greek_frame, text="Post-Earnings Delta").grid(row=0, column=2, padx=(0, 5))
        self.post_delta_label = ttk.Label(crush_frame, text="N/A", font=("Arial", 10, "bold"))
        self.post_delta_label.grid(row=0, column=2, padx=(0, 20))
        
        ttk.Label(greek_frame, text="Delta Change").grid(row=0, column=4, padx=(0, 5))
        self.delta_change_label = ttk.Label(crush_frame, text="N/A", font=("Arial", 10, "bold"))
        self.delta_change_label.grid(row=0, column=4, padx=(0, 20))
        
        ttk.Label(greek_frame, text="Pre-Earnings Vega").grid(row=2, column=0, padx=(0, 5))
        self.pre_vega_label = ttk.Label(crush_frame, text="N/A", font=("Arial", 10, "bold"))
        self.pre_vega_label.grid(row=1, column=0, padx=(0, 20))
        
        ttk.Label(greek_frame, text="Post-Earnings Vega").grid(row=2, column=2, padx=(0, 5))
        self.post_vega_label = ttk.Label(crush_frame, text="N/A", font=("Arial", 10, "bold"))
        self.post_vega_label.grid(row=1, column=2, padx=(0, 20))
        
        ttk.Label(greek_frame, text="Vega Change").grid(row=2, column=4, padx=(0, 5))
        self.vega_change_label = ttk.Label(crush_frame, text="N/A", font=("Arial", 10, "bold"))
        self.vega_change_label.grid(row=1, column=4, padx=(0, 20))
        
        status_frame = ttk.LabelFrame(main_frame, text="Greek Analysis", padding="5")
        status_frame.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), paddy=(0, 10))
        
        self.status_text = scrolledtext.ScrolledText(status_frame, height=6, width=80)
        self.ststus_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        status_frame.columnconfigure(0, weight=1)
        
        plot_frame = ttk.LabelFrame(main_frame, text="IV Crush Visualization", padding = "5")
        plot_frame.grid(row=8, columnspan=2, coumn=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        
        self.fig, (self.ax1, self.ax2) = plt.pubplots(1, 2, figsize=(16, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.E, tk.W, tk.N, tk.S))
        
    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] message\n")
        
        self.status_text.see(tk.END)
        self.root.update_iddletasks()
        
    def connect_ib(self):
        try:
            host = self.host_var.get()        
            port = int(self.host_var.get())       
            
            self.log_message(f"Connecting to IB at {host} : {port}") 
            
            def connect_thread():
                try:
                    self.ib_app.connect(host, port)
                    self.ib_app.run()

                except Exception as e:
                    self.log_message(f"Connection error: {e}")
                    
            thread = threading.Thread(target=connect_thread, daemon=True)
            thread.start()
            
            for i in range(100):
                if self.ib_app.connected:
                    try:
                        server_version = self.ib_app.serverVersion()
                        if server_version is not None and server_version > 0:
                            break
                        
                    except:
                        pass
                    time.sleep()
                    
            if self.ib_app.connected:
                try:
                    server_version = self.ib_app.serverVersion()
                    if server_version is not None and server_version > 0:
                        self.connected = True
                        self.connect_btm.config(state="disabled")
                        self.disconnect_btm(state="normal")
                        self.analyze_btn(state="normal")
                        self.log_message(f"Successfully Connected to IB | Server version: {server_version}")
                    else:
                        self.log_message(f"Connected, but server version unavailable")
                except Exception as e:
                    self.log_message(f"Connected, but server version check failed: {e}")
            else:
                self.log_message("Failed to connect to IB")
        
        except Exception as e:
            self.log_message("Connection Error: {e}")
            
    def disconnecr_ib(self):
        try:
            self.ib_app.disconnect()
            self.connect = False
            self.connect_btm.config(state="normal")
            self.disconnect_btm.config(state="disabled")
            self.analyze_btn.config(state="disabled")
            
            self.clear_analysis_results()
            
            self.log_message("Disconnect from Interactive Brokers")
            
        except Exception as e:
            self.log_message(f"Disconnect Error: {e}")
            
    def clear_analysis_results(self):
        self.ax1.clear()
        self.ax2.clear()
        
        if self.ax1_twin.remove():
            try:
                self.ax1_twin = None
            
            except:
                pass
            self.ax1_twin = None
            
        self.canvas.draw()
        
        self.stock_price_label.config(text="N/A", foreground="black")
        self.vix_level_label.config(text="N/A", foreground="black")
        self.curr_iv_label.config(text="N/A", foreground="black")
        
        self.pre_iv_label.config(text="N/A", foreground="black")
        self.post_iv_label.config(text="N/A", foreground="black")
        self.iv_crush_label.config(text="N/A", foreground="black")
        
        self.pre_call_label.config(text="N/A", foreground="black")
        self.post_call_label.config(text="N/A", foreground="black")
        self.pre_put_label.config(text="N/A", foreground="black")
        self.post_put_label.config(text="N/A", foreground="black")
        self.put_loss_label.config(text="N/A", foreground="black")
        
        self.pre_straddle_label.config(text="N/A", foreground="black")
        self.post_straddle_label.config(text="N/A", foreground="black")
        self.straddle_loss_label.config(text="N/A", foreground="black")
        
        self.long_pl_label.config(text="N/A", foreground="black")
        self.short_pl_label.config(text="N/A", foreground="black")
        
        self.strike_price_label.config(text="N/A", foreground="black")
        self.pre_spot_label.config(text="N/A", foreground="black")
        self.post_spot_label.config(text="N/A", foreground="black")
        
        self.pre_delta_label.config(text="N/A", foreground="black")
        self.post_delta_label.config(text="N/A", foreground="black")
        self.delta_change_label.config(text="N/A", foreground="black")
        
        self.pre_vega_label.config(text="N/A", foreground="black")
        self.post_vega_label.config(text="N/A", foreground="black")
        self.vega_change_label.config(text="N/A", foreground="black")
        
        self.stock_data = None
        self.vix_data = None
        self.iv_data = None
        
        if hasattr(self, "ib_app") and self.ib_app:
            self.ib_app.hist_data.clear()
            
        self.log_message("Analyss results cleared - ready for new analysis")
        
        
    def black_scholes_call(self, S, K, T, r, sigma):
        d1 =(np.log(S/K) + (r+0.5*sigma**2)*T) / (sigma*np.squrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        call_price = S *norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        
        return call_price
    
    def black_scholes_put(self, S, K, T, r, sigma):
        d1 =(np.log(S/K) + (r+0.5*sigma**2)*T) / (sigma*np.squrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        put_price = K * np.exp(-r*T) * norm.cdf(d2) - S * norm.cdf(d1)
        
        return put_price
    
    def calculate_delta(self, S, K, T, r, sigma, option_type="call"):
        d1 =(np.log(S/K) + (r+0.5*sigma**2)*T) / (sigma*np.squrt(T))
    
        if option_type == "call":
            return norm.cdf(d1)
        else:
            return -norm.cdf(-d1)
        
    def calculate_vega(self, S, K, T, r, sigma):
        d1 =(np.log(S/K) + (r+0.5*sigma**2)*T) / (sigma*np.squrt(T))
        return S * norm.pdf(d1) * np.sqrt(T) / 100
    
    def analyze_iv_crush(self):
        if not self.connected or not self.ib_app.connected:
            messagebox.showerror("Error", "Not connected to IB")
            return 
        
        self.ticker = self.ticker_var.get().upper()
        earnings_date_str = self.earnings_date_var.get()
        
        try:
            self.earnings_date = datetime.strptime(earnings_date_str, "%y-%m-%d")
            
        except ValueError:
            messagebox.showerror(" Error", "Invalid Date format. Use: YYYY-MM-DD")
            return 
        
        self.log_message(f"Starting IV Crush analysis for {self.ticker} around earnings on {self.earnings_date}")          
        
        self.clear_analysis_results()
        
        start_date  =self.earnings_date - timedelta(days = 10)
        end_date  =self.earnings_date + timedelta(days = 10)
        
        self.ib_app.hist_data.clear()
        
        self.log_message(f"Querrying stock data for {self.ticker}")
        stock_contract = self.create_equity_contract(self.ticker)
        
        if 1 in self.ib_app.hist_data:
            del self.ib_app.hist_data[1]
            
        try:
            self.ib_app.requestHistoricalData(
                reqId=1,
                contract=stock_contract,
                endDateTime=end_date.strftime("%Y%m%d %H:%M:%S"),
                durationStr = "3 w",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=1,
                format_date=1,
                keepUpToDate=False,
                chartOptions=[]  
            )
            
        except Exception as e:
            self.log_message(f"Error requesting stock data: {e}")
            messagebox.showerror("Error", f"Failed to request stock data")
            return
        
        timeout = 15
        start_time = time.time()
        while 1 not in self.ib_app.hist_data and (time.time() - start_time) < timeout:
            time.sleep(.1)
        

        if 1 not in self.ib_app.hist_data:
            self.log_message("Failed to get stock price data")
            return

        stock_data = pd.DataFrame(self.ib_app.hist_data[1])
        stock_data["date"] = pd.to_datetime(stock_data["date"])
        stock_data.set_index("date", inplace=True)
        self.stock_data = stock_data
        
        self.log_message(f"Received {len(stock_data)} stock price datapoints")
        
        self.log_message("Querying VIX data")
        vix_contract = self.create_vix_contract()
        
        if 2 in self.ib_app.hist_data:
            del self.ib_app.hist_data[2]
            
        try:
            self.ib_app.requestHistoricalData(
                reqId=2,
                contract=vix_contract,
                endDateTime=end_date.strftime("%Y%m%d %H:%M:%S"),
                durationStr = "3 w",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=1,
                format_date=1,
                keepUpToDate=False,
                chartOptions=[]  
            )
            
        except Exception as e:
            self.log_message(f"Error requesting vix data: {e}")
        
        start_time = time.time()
        while 2 not in self.ib_app.hist_data and (time.time() - start_time) < timeout:
            time.sleep(.1)
            
        if 2 in self.ib_app.hist_data:
            vix_data = pd.DataFrame(self.ib_app.hist_data[2])
            vix_data["date"] = pd.to_datetime(vix_data["date"])
            vix_data.set_index("date", inplace=True)
            self.vix_data = vix_data
            
            self.log_message(f"Received {len(vix_data)} vix datapoints")
            
        else:
            self.log_message(f"VIX data not available")
            self.vix_data = None
            
        self.log_message(f"Querying implied volatility data for {self.ticker}...")
        
        if 3 in self.ib_app.hist_data:
            del self.ib_app.hist_data[3]
              
        try:
            self.ib_app.requestHistoricalData(
                reqId=3,
                contract=stock_contract,
                endDateTime=end_date.strftime("%Y%m%d %H:%M:%S"),
                durationStr = "3 w",
                barSizeSetting="1 day",
                whatToShow="OPTION_IMPLIED_VOLATILITY",
                useRTH=1,
                format_date=1,
                keepUpToDate=False,
                chartOptions=[]  
            )
            
        except Exception as e:
            self.log_message(f"Error requesting IV data: {e}")

        start_time = time.time()
        while 3 not in self.ib_app.hist_data and (time.time() - start_time) < timeout:
            time.sleep(.1)
            
        if 3 in self.ib_app.hist_data:
            iv_data = pd.DataFrame(self.ib_app.hist_data[3])
            iv_data["date"] = pd.to_datetime(iv_data["date"])
            iv_data.set_index("date", inplace=True)
            
            raw_iv = iv_data["close"]
            if raw_iv.max > 5:
                daily_iv_decimal = raw_iv / 100
                iv_data["implied_vol"] = daily_iv_decimal
                self.log_message(f"Received {len(iv_data)} IV datapoints - converted to annualized vol")
            
            else:
                iv_data["implied_vol"] = raw_iv
                self.log_message(f"Received {len(iv_data)} IV datapoints")
                
            self.iv_data = iv_data            
            self.log_message("Successfully processed implied vol data")
        else:
            self.log_message(f"IV data not available - annualized vol")
            self.iv_data = None
            
        self.perform_iv_crush_analysis()
        
    def perform_iv_crush_analysis(self):
        self.log_message("Performing IV crush analysis")
        
        earnings_date = self.earnings_date
        pre_earnings_date = earnings_date - timedelta(days=1)
        post_earnings_date = earnings_date + timedelta(days=1)
        
        stock_dates = self.stock_data.index
        
        pre_date_actual = stock_dates[stock_dates <= earnings_date].max() if len(stock_dates[stock_dates <= earnings_date]) > 0 else stock_dates.min()
        post_date_actual = stock_dates[stock_dates >= earnings_date].max() if len(stock_dates[stock_dates > earnings_date]) > 0 else stock_dates.max()
        
        self.log_message(f"Earnings Date: {earnings_date} | Pre-Date: {pre_date_actual} | Post-Date: {post_date_actual}")
        
        pre_stock_price = self.stock_data.loc[pre_date_actual, "close"]
        post_open = self.stock_data.loc[post_date_actual, "open"]
        post_close = self.stock_data.loc[post_date_actual, "close"]
        post_stock_price = (post_open + post_close) / 2

        gap = (post_open - pre_stock_price) / pre_stock_price * 100
        self.log_message(f"Overnight earnings gap: {gap}")
        
        total_move = (post_close - pre_stock_price) / pre_stock_price * 100
        self.log_message(f"Total Move %: {total_move}")

        self.stock_price_label(text = f"${post_stock_price}")
        
        if self.vix_data is not None:
            vix_dates = self.vix_data.index
            pre_vix_date = vix_dates[vix_dates <= pre_earnings_date].max() if len(vix_dates[vix_dates <= pre_earnings_date]) > 0 else vix_dates.min()
            post_vix_date = vix_dates[vix_dates <= pre_earnings_date].min() if len(vix_dates[vix_dates > pre_earnings_date]) > 0 else vix_dates.max()
            
            pre_vix = self.vix_dates.loc[pre_vix_date, "close"]
            post_vix = self.vix_dates.loc[post_vix_date, "close"]
            
            self.log_message(f" Pre Earnings VIX: {pre_vix}")
            self.log_message(f" Post Earnings VIX: {post_vix}")
            self.vix_level_label.config(text="N/A")

        else:
            self.vix_level_label.config(text="N/A")
        
        if self.i_data is not None:
            iv_dates = self.iv_data.index
            pre_iv_date = iv_dates[iv_dates <= pre_earnings_date].max() if len(iv_dates[iv_dates <= pre_earnings_date]) > 0 else iv_dates.min()
            post_iv_date = iv_dates[iv_dates > pre_earnings_date].min() if len(iv_dates[iv_dates > pre_earnings_date]) > 0 else iv_dates.max()
            
            pre_iv = self.iv_data.loc[pre_iv_date, "implied_vol"]
            post_iv = self.iv_data.loc[post_iv_date, "implied_vol"]
            
        self.log_message(f"Pre-Earnings IV: {pre_iv}")
        self.log_message(f"Post-Earnings IV: {post_iv}")
        
        iv_crush = (pre_iv - post_iv) / pre_iv* 100
        self.current_iv_label.config(text=f"{post_iv}")
        self.pre_iv_label.config(text=f"{pre_iv}")
        self.post_iv_label.config(text=f"{post_iv}")
        self.iv_crush_label.config(text=f"{iv_crush}")
        
        try:
            days_to_expiry = int(self.days_to_exp_var.get())
        except ValueError:
            days_to_expiry = 30
            
            self.days_to_exp_var.set("30")
            
        time_to_expiry = days_to_expiry / 365
        atm_strike_price = pre_stock_price
        
        pre_call_price = self.black_scholes_call(pre_stock_price, atm_strike_price, time_to_expiry, self.risk_free_rate, pre_iv)
        pre_put_price = self.black_scholes_put(pre_stock_price, atm_strike_price, time_to_expiry, self.risk_free_rate, pre_iv)
        
        post_call_price = self.black_scholes_call(post_stock_price, atm_strike_price, time_to_expiry, self.risk_free_rate, pre_iv)
        post_put_price = self.black_scholes_put(pre_stock_price, atm_strike_price, time_to_expiry, self.risk_free_rate, pre_iv)

        pre_straddle_price = pre_call_price + pre_put_price
        post_straddle_price = post_call_price + post_put_price
        
        call_change_dollar = post_call_price  - pre_call_price
        
        put_change_dollar = post_put_price - pre_put_price
        
        straddle_change_dollar = post_straddle_price - pre_straddle_price
        
        long_straddle_pnl = straddle_change_dollar
        short_straddle_pnl = -straddle_change_dollar
        
        pre_call_delta = self.calculate_delta(pre_stock_price, atm_strike_price, time_to_expiry, pre_iv)
        pre_put_delta = self.calculate_delta(pre_stock_price, atm_strike_price, time_to_expiry, pre_iv, "put")
        pre_straddle_delta = pre_call_delta + pre_put_delta
        
        post_call_delta = self.calculate_delta(pre_stock_price, atm_strike_price, time_to_expiry, pre_iv)
        post_put_delta = self.calculate_delta(pre_stock_price, atm_strike_price, time_to_expiry, pre_iv, "put")
        post_straddle_delta = post_call_delta + post_put_delta
        
        delta_change = post_straddle_delta - pre_straddle_delta
        
        pre_call_vega = self.calculate_vega(pre_stock_price, atm_strike_price, time_to_expiry, self.risk_free_rate, pre_iv)
        pre_straddle_vega = 2*pre_call_vega
        
        post_call_vega = self.calculate_vega(post_stock_price, atm_strike_price, time_to_expiry, self.risk_free_rate, post_iv)
        post_straddle_vega = 2*post_call_vega
        
        vega_change = post_straddle_vega - pre_straddle_vega
        
        self.pre_call_label.config(text=f"{pre_call_price:.2f}")
        self.post_call_label.config(text=f"{post_call_price:.2f}")
        call_color = "green" if call_change_dollar > 0 else "red"
        self.call_loss_label.config(text=f"{pre_call_price:.2f}", foreground=call_color)
            
        self.pre_put_label.config(text=f"{pre_put_price:.2f}")
        self.post_put_label.config(text=f"{post_put_price:.2f}")
        put_color = "green" if call_change_dollar > 0 else "red"
        self.put_loss_label.config(text=f"{pre_put_price:.2f}", foreground=put_color)
        
        self.pre_straddle_label.config(text=f"{pre_straddle_price:.2f}")
        self.post_straddle_label.config(text=f"{post_straddle_price:.2f}")
        straddle_color = "green" if call_change_dollar > 0 else "red"
        self.straddle_loss_label.config(text=f"{pre_straddle_price:.2f}", foreground=straddle_color)
        
        long_color = "green" if long_straddle_pnl > 0 else "red"
        short_color = "green" if short_straddle_pnl > 0 else "red"
        
        self.long_pnl_label.config(text=f"{long_straddle_pnl:.2f}", foreground=long_color)
        self.short_pnl_label.config(text=f"{short_straddle_pnl:.2f}", foreground=short_color)
            
        self.strike_price_label.config(text=f"{atm_strike_price}")
        
        post_spot_color = "green" if post_stock_price > pre_stock_price else "red"
        self.pre_spot_label.config(text=f"{pre_stock_price}", foreground="black")
        self.post_spot_label.config(text=f"{pre_stock_price}", foreground=post_spot_color)
        
        delta_color = "red" if abs(post_straddle_delta) < abs(pre_straddle_delta) else "green"
        self.pre_delta_label.config(text=f"{pre_straddle_delta}")
        self.pre_delta_label.config(text=f"{post_straddle_delta}")
        self.delta_change_label.config(text=f"{delta_change}", foreground=delta_color)
        
        vega_color = "red" if vega_cange > 0 else "green"
        self.pre_vega_label.config(text=f"{pre_straddle_vega}")
        self.pre_vega_label.config(text=f"{post_straddle_vega}")
        self.vega_change_label.config(text=f"{vega_change}", foreground=vega_color)
        
        self.create_visualizations()
        
    def create_visualizations(self):
        self.ax1.clear()
        self.ax2.clear()
        
        if self.ax1_twin == None:
            try:
                self.ax1_twin.remove()
                
            except:
                pass
            
            self.ax1_twin = None
            
        start = self.earnings_date - timedelta(days=5)
        end = self.earnings_date + timedelta(days=5)
            
        earnings_window = pd.date_range(
            start=start,
            end=end,
            freq="D"
        )
        
        window_stock = self.stock_data[
            (self.stock_data >= start) and  \
            (self.stock_data.index <= end)
        ]
        
        self.ax1.plot(window_stock.index, window_stock["close"], "b--", label="Stock Price")
        self.ax1.axvline(x=self.earnings_date, color="red", linestile="--",alpha=0.7, label="Earnings Date")    
        self.ax1.set_xlabel("Date")
        self.ax1.set_ylabel("$Price", color="blue")
        self.ax1.tick_params(axis="y", labelcolor="blue")
        self.ax1.set_title(f"{self.ticker} Stock Price Around Earnings")
        self.ax1.grid(True, alpha=.3)      
        self.ax1.legend(loc="upper left")  
        
        if self.iv_data is not None and len(self.iv_data > 0):
            window_iv = self.iv_data[
                (self.iv_data >= start) and  \
                (self.iv_data.index <= end)
            ]
        
            if len(window_iv) > 0:
                self.ax1_twin = self.ax1.twinx()
                iv_percentage = window_iv["implied_vol"] * 100

                self.ax1_twin.plot(window_stock.index, iv_percentage, "g-", label="Implied Vol Ann")
                self.ax1_twin.set_xlabel("Date")
                self.ax1_twin.set_ylabel("Annualized Implied Vol", color="green")
                self.ax1_twin.tick_params(axis="y", labelcolor="green")
                self.ax1_twin.set_title(f"{self.ticker} IV Around Earnings")
                self.ax1_twin.grid(True, alpha=.3)      
                self.ax1_twin.legend(loc="upper right")  
            
        if self.vix_data is not None and len(self.vix_data > 0):                
            window_vix = self.vix_data[
                (self.vix_data >= start) and  \
                (self.vix_data.index <= end)
            ]
        
            if len(window_iv) > 0:
                self.ax2 = self.ax1.twinx()

                self.ax2.plot(window_vix.index, window_vix["close"], "g-", label="VIX")
                self.ax2.axvline(x=self.earnings_date, color="red", linestile="--",alpha=0.7, label="Earnings Date")    
                self.ax2.set_xlabel("Date")
                self.ax2.set_ylabel("Vix Level")
                self.ax2.set_title(f"{self.ticker} Vix Around Earnings")
                self.ax2.grid(True, alpha=.3)      
                self.ax2.legend(loc="upper right")  
        
        else:
            pass
            
        self.ax1.tick_params(axis="x", rotation=45)
        if self.vix_data is not None:
            self.ax2.tick_params(axis="x", rotation=45)
            
        self.fig.tight_layout()
        self.canvas.draw()
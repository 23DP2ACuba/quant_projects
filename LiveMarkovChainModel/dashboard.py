import tkinter as tk
from tkinter import ttk, messagebox
import time
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle  
import matplotlib.dates as mdates   
import matplotlib.pyplot as plt
from datetime import datetime as datetime
from ibapp import IBApp, OHLCBar
from markov_model import MarkovRegime
from collections import deque
import threading as thr
from utils import Theme

class Dashboard(Theme):
    def __init__(self, root):
        
        self.root = root
        self.root.title("Live Market Data")
        self.root.geometry("1200x800")
        
        self.root.configure(bg=self.BGCOLOR)
        
        self.style = ttk.Style()
        self.style.theme_use(self.THEME)
        self.configure_dark_theme()
        
        self.ib_app = IBApp(callback=self.on_tick_data)
        self.connected = False
        self.steaming = False
        self.timeout_sec = 50
        
        self.bar_duration = 5
        self.max_bars = 10
        self.ohlc_bars = deque(maxlen=self.max_bars)
        self.current_bar = None
        self.bar_start_time = None
        self.price_history = deque(maxlen=100)
        self.last_upate_time = None
        
        self.bar_lock = thr.Lock()
        self.update_thread = None
        self.running = False
                
        self.setup_ui()
        self.setup_chart()
        
    def configure_dark_theme(self):        
        self.style.configure("TFrame", background=self.BGCOLOR)
        self.style.configure("TLabelframe", background=self.BGCOLOR, foreground=self.FGGRAY)
        self.style.configure("TLabelframe.Label", background=self.BGCOLOR, foreground=self.FGGRAY,
                             font=("Segoe UI", 10, "bold"))
        self.style.configure("TLabel", background=self.BGCOLOR, foreground=self.FGGRAY,
                             font=("Segoe UI", 10))
        self.style.configure("TLabel", background=self.BGCOLOR, foreground=self.FGGRAY,
                             font=("Segoe UI", 9, "bold"), padding=(10, 5))
        self.style.map("TButton",
                       background=[("active", "#2ea043"), ("disabled", self.DISABLED)])
        
        self.style.configure("TEntry", fieldbackground=self.ENTRYBG,
                             foreground=self.FGGRAY, insertcolor=self.FGGRAY)
        self.style.configure("Accent.TButton", background="#da3633", foreground=self.WHITE)
        self.style.map("Accent.TButton",
                       background=[("active", self.FGRED), ("disabled", self.DISABLED)])
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding=15)
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0,weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, sticky="ew",pady=(0, 15))
        
        title_lable= tk.Label(header_frame, text="Live Regime Swithcing",
                              font=("JetBrains Mono", 18, "bold"), 
                              bg=self.BGCOLOR, fg=self.FGBLUE)
        title_lable.pack(side="left")
        self.status_indicator = tk.Label(header_frame, text="DISCONNECTED", 
                                         font=("JetBrains Mono", 18, "bold"), 
                                        bg=self.BGCOLOR, fg=self.FGRED)
        self.status_indicator.pack(side="right", padx=10)
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="10")
        control_frame.grid(row=1, column=0, sticky="ew", pady=(0, 15))
        connection_section = ttk.Frame(control_frame)
        connection_section.pack(fill="x", pady=(0, 10))
        
        ttk.Label(connection_section, text="HOST: ").pack(side="left", padx=(0, 5))
        self.host_var = tk.StringVar(value="127.0.0.1")
        host_entry = ttk.Entry(connection_section, textvariable=self.host_var, width=12)
        host_entry.pack(side="left", padx=(0, 15))
        
        ttk.Label(connection_section, text="PORT: ").pack(side="left", padx=(0, 5))
        self.port_var = tk.StringVar(value="7497")
        port_entry = ttk.Entry(connection_section, textvariable=self.port_var, width=12)
        port_entry.pack(side="left", padx=(0, 15))
        
        self.connect_btn = ttk.Button(connection_section, text="Connect", command=self.connect_ib)
        self.connect_btn.pack(side="left", padx=(0, 5))
        
        self.disconnect_btn = ttk.Button(connection_section, text="Disconnect", command=self.disconnect_ib, state="disabled", style="Accent.TButton")
        self.disconnect_btn.pack(side="left", padx=(0, 5))
        
        sep = ttk.Separator(control_frame, orient="horizontal")
        sep.pack(fill="x", pady=10)
        
        data_section = ttk.Frame(control_frame)
        data_section.pack(fill="x")
        
        ttk.Label(data_section, text="Symbol").pack(side="left", padx=(0, 5))
        self.symbol_var = tk.StringVar(value="AAPL")
        symbol_entry = ttk.Entry(data_section, textvariable=self.symbol_var,
                                 width=10, font=("JetBrains Mono", 11))
        symbol_entry.pack(side="left")
        
        self.stream_btn = ttk.Button(data_section, text="Start Stream", command = self.toggle_stream, state="disabled")
        self.stream_btn.pack(side="left", padx=(0, 5))
        
        self.recal_btn = ttk.Button(data_section, text="Recalibrate", command=self.recalibrate_model, state="disabled")
        self.recal_btn.pack(side="left", padx=(0, 15))
        
        price_frame = ttk.Frame(data_section)
        price_frame.pack(side="left")
        
        ttk.Label(price_frame, text="Last Price:",
                  font=("Segor UI", 10)).pack(side="left", padx=(0, 5))
        self.price_label = tk.Label(price_frame, text="---.--",
                                   font=('JetBrains Mono', 16, 'bold'),
                                   bg=self.BGCOLOR, fg='#7ee787')
        self.price_label.pack(side="left")

                
        chart_frame = ttk.LabelFrame(main_frame, text="OHLCL Regime", padding=10)
        chart_frame.grid(row=2, column=0, sticky="nsew")
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        
        self.chart_container = ttk.Frame(chart_frame)
        self.chart_container.grid(row=0, column=0, sticky="nsew")
        self.chart_container.columnconfigure(0, weight=1)
        self.chart_container.rowconfigure(0, weight=1)
        
        stats_frame = ttk.Frame(main_frame)
        stats_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        
        self.stats_labels = {}
        stats = [("Bars", 0), ("High", "--"), ("Low", "--"), ("Regime", "--"), ("TIck/Bar", "0")]
        
        for i, (name, value) in enumerate(stats):
            frame = ttk.Frame(stats_frame)
            frame.pack(side="left", padx=15)
            ttk.Label(frame, text=f"{name}: ", font=("Segoe UI", 9)).pack(side="left")
            label = tk.Label(frame, text=value, font=("JetBrains Mono", 10, "bold"),
                              bg=self.BGCOLOR, fg=self.FGGRAY)
            label.pack(side="left", padx=(5, 0))
            self.stats_labels[name] = label
        
    def setup_chart(self):
        plt.style.use("dark_background")
        
        self.fig, self.ax = plt.subplots(figsize=(12, 6), facecolor=self.BGCOLOR)
        self.ax.set_facecolor(self.FACECOLOR)
        
        self.ax.tick_params(color=self.FGGRAY, labelsize=9)
        self.ax.spines["bottom"].set_color(self.DARKGRAY)
        self.ax.spines["top"].set_color(self.DARKGRAY)
        self.ax.spines["left"].set_color(self.DARKGRAY)
        self.ax.spines["right"].set_color(self.DARKGRAY)
        
        self.ax.grid(True, alpha=.2, color=self.DARKGRAY, linestyle="--")
        
        self.ax.set_xlabel("Time", color=self.FGGRAY, fontsize=10)
        self.ax.set_ylabel("Price", color=self.FGGRAY, fontsize=10)
        self.ax.set_title("Waiting for data...", color=self.FGGRAY, fontsize=12, fontweight="bold")
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.chart_container)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        
        
    def connect_ib(self):
        try:
            host = self.host_var.get()
            port = int(self.port_var.get())
            
            def connect_thread():
                try:
                    self.ib_app.connect(host, port, clientId=1)
                    self.ib_app.run()
                    
                except Exception as e:
                    print(f"Connection Error: {e}")
                    
            thread = thr.Thread(target=connect_thread, daemon=True)
            thread.start()
            
            for i in range(self.timeout_sec):
                if self.ib_app.connected:
                    break
                time.sleep(0.1)
                
            if self.ib_app.connected:
                self.connected =True
                self.connect_btn.config(state="disabled")
                self.disconnect_btn.config(state="normal")
                self.stream_btn.config(state="normal")
                self.status_indicator.config(text="CONNECTeD", fg=self.FGGREEN)
            else:
                messagebox.showerror("Error", "Failed to connect to TWS")
                
        except Exception as e:
            messagebox.showerror("Error", f"Connection error: {e}")

    def disconnect_ib(self):
        try:
            if self.streaming:
                self.stop_stream()
                
            self.ib_app.disconnect()
            self.connected = False
            self.connect_btn.config(state="normal")
            self.disconnect_btn.config(state="disabled")
            self.stream_btn.config(state="disabled")
            self.status_indicator.config(text="DISCONNECTED", fg="#f83149")
            
        except Exception as e:
            print(f"Disconnect error: {e}")
            
    def toggle_stream(self):
        if not self.streaming:
            self.start_stream()
        else:
            self.stop_stream()
            
            
            
    def start_stream(self):
        if not self.connected:
            return 
        
        symbol = self.symbol_var.get().upper()
        if not symbol:
            messagebox.showerror("Error", "No symbol name provided")
            return

        with self.bar_loc:
            self.ohlc_bars.clear()
            self.current_bar = None
            self.bar_start_time = None
            self.price_history.clear()
            self.regime_model = MarkovRegime()
        
        contract = self.ib_app.create_contract(symbol)
        
        self.ib_app.historical_data.clear()
        self.ib_app.hist_done.clear()
        self.ib_app.reqHistoricalData(2, contract, "", "300 S","5 sec", "TRADES", 1, 1, False, [])
        
        if self.ib_app.hist_done.wait(timeout=10) and 2 in self.ib_app.historical_data:
            self.regime_model.calibrate_model(self.ib_app.historical_data[2])
            print(f"Calibrated regime model with {len(self.ib_app.historical_data[2])} bars")
        
        self.ib_app.reqMktData(1, contract, "", False, False, [])
        
        self.streaming = True
        self.running = True
        self.stream_btn.config(text="Stop Stream", style="Accent.TButton") 
        self.recal_btn.config(state="normal")           
        self.status_indicator.config(text=f"Streaming {symbol}", fg=self.FGBLUE)
        
        self.update_thread = thr.Thread(target=self.bar_manager_loop, daemon=True)
        self.update_thread.start()
        
        self.update_chart_loop()
    
    def stop_stream(self):
        self.running = False
        self.streaming = False
        
        try:
            self.ib_app.cancelMktData(1)
            
        except Exception as e:
            print(f"Error cancelling market data request: {e}")
        
        self.stream_btn.config(text="Start Streaming", style="TButton")
        self.recal_btn.config(state="disabled")
        self.status_indicator.config(text="CONNECTED", fg=self.FGGREEN)
        
    def bar_manager_loop(self):
        while self.running:
            time.sleep(.1)
            
            while self.bar_lock:
                if self.current_bar is not None and self.bar_start_time is not None:
                    elapsed = (datetime.now() - self.bar_start_time).total_seconds()
                    
                    if elapsed >= self.bar_duration:
                        self.ohlc_bars.append(self.current_bar)
                        self.regime_model.get_regime(list(self.ohlc_bars))
                        last_price = self.current_bar.close
                        self.current_bar = OHLCBar(datetime.now(), last_price)
                        self.bar_start_time = datetime.now()
                        
    
    def update_chart_loop(self):
        if not self.running:
            return
        self.draw_ohlcv()
        self.update_stats()
        self._after_id = self.root.after(200, self.update_chart_loop)
        
    def draw_ohlc_chart(self):
        eps = 1e-3
        self.ax.clear()
        
        with self.bar_lock:
            bars = list(self.ohlc_bars)
            current = self.current_bar
            
        if current is not None:
            bars += [current]
            
        if not bars:
            self.ax.set_facecolor(self.FACECOLOR)
            self.ax.set_title("Waiting for data...", color=self.FGGRAY,
                              fontsize=12, fontweight="bold")
            self.ax.grid(True, alpha=.2, color=self.DARKGRAY, linestyle="--")
            return 
        
        all_prices = [bar.low for bar in bars] + [bar.high for bar in bars]
        price_min, price_max = min(all_prices), max(all_prices)
        pricerange = price_max - price_min
        pad = max(pricerange * .1, .01)
        y_min, y_max = price_min - pad + price_max + pad
        
        if current is not None:
            pass
        
        width = .6
        for i, bar in enumerate(bars):
            bg = Rectangle(
                (i - .5, y_min),
                1,
                y_max - y_min,
                face = (1, 1, 1, 0),
                edgecolor=None,
                alpha=0,
                zorder=0,
            )
            
            self.ax.add_patch(bg)
            color, edge_color = self.G_REG if bar.close >= bar.open else self.R_REG
            body_bottom, body_height = min(bar.oprn, bar.close), max(abs(bar.oprn, bar.close), eps)
            
            candle = Rectangle(
                (i - width/2, body_bottom),
                width,
                body_height,
                facecolor = color,
                edge_color = edge_color,
                linewidth = 1.5,
                alpha=.9,
                zorder=2
            )
            
            self.ax.add_patch(candle)
            self.ax.plot([i, i], [bar.low, body_bottom], color=edge_color, linewidth=1.5, zorder=1)
            self.ax.plot([i, i], [body_bottom+body_height, bar.high], color=edge_color, linewidth=1.5, zorder=1)
            
            if i == len(bars) - 1 and current is not None:
                self.ax.axvline(x=i, color=self.FGBLUE, alpha=.3, linestyle=":", linewidth=2)
                
            self.ax.set_facecolor(self.BGCOLOR)
            x_labels = [bar.timestamp.strftime("%H:%M:%S") for bar in bars]
            self.ax.set_xticks(range(len(bars)))
            self.ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
            self.ax.set_ylim(y_min, y_max)
            self.ax.set_xlim(-.5, max(self.max_bars - .5, len(bars) - .5))
            
            self.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x: .3f}"))
            self.ax.set_xlabel("Time", color=self.FGGRAY, fontsize=10)
            self.ax.set_ylabel("Price", color=self.FGGRAY, fontsize=10)
            
            symbol = self.symbol_var.get().upper()
            regime_names = ["LOW", "MED", "HIGH"]
            curr_regime = regime_names[bars[-1].regime] if bars else "N/A"
            self.ax.set_title(f"{symbol} - Regime: {curr_regime} | {len(bars)}/{self.max_bars} bars",
                              color = self.WHITE, fontsize=12, fontweight="bold")
            self.fig.tight_layout()
            self.canvas.draw_idle()            
    
    def update_stats(self):
        with self.bar_lock:
            bars = list(self.ohlc_bars)
            current = self.current_bar
            
        if current:
            bars += [current]
            
        if not bars:
            return
        
        self.stats_labels["Bars"].config(text=str(len(bars)))
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        
        self.stats_labels["High"].config(text=f"{max(highs)}")
        self.stats_labels["LOW"].config(text=f"{min(lows)}")
        
        regime_names = ["LOW", "MED", "HIGH"]
        regime_colors = self.REG_CLRS
        curr_regime = bars[-1].regime if bars else 0
        self.stats_labels["Regime".config(text=regime_names[curr_regime], fg=regime_colors[curr_regime])]
        
        if current:
           self.stats_labels["Ticks/Bae"].config(text=str(current.tick_count)) 
        
        
    def recalibrate_model(self):
        pass
    
    def on_tick_data(self, data_type, value, timestamp):
        if data_type == "price" and value > 0:
            with self.bar_lock:
                self.price_history.append((timestamp, value))
                
                if self.current_bar is None:
                    self.current_bar = OHLCBar(timestamp, value)
                    self.bar_start_time = timestamp
                    
                else:
                    self.current_bar.update(value)

                
            self.root.after(0, lambda: self.price_label.config(text=f"{value:.2f}"))    

    
    def on_closing(self):
        self.running = False
        if hasattr(self, "_after_id"):
            self.root.after_cancel(self._after_id)
            
        if self.connected:
            try:
                if self.streaming:
                    self.ib_app.cancelMktData(1)
                self.ib_app.disconnect()
                
            except:
                pass
        self.root.destroy()
    
import tkinter as tk
from tkinter import ttk, messagebox
import time
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle  
import matplotlib.dates as mdates   
from datetime import datetime as datetime
from ibapp import IBApp, OHLCBar
from markov_model import MarkovRegime
from collections import deque
import threading as thr


class Dashboard(MarkovRegime):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.root.title("Live Market Data")
        self.root.geometry("1200x800")
        self.root.configure(bg="#0d1117")
        
        self.style = ttk.Style()
        self.style.theme_use("clam")
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
        self.regime_model = MarkovRegime()
        
        self.bar_lock = thr.Lock()
        self.update_thread = None
        self.running = False
                
        self.setup_ui()
        self.setup_chart()
        
    def configure_dark_theme(self):
        bg_color = "#0d1117"
        fg_color = "#c9d1d9"
        accent_color = "#238636"
        entry_bg = "#161b22"
        
        self.style.configure("TFrame", background=bg_color)
        self.style.configure("TLabelframe", background=bg_color, foreground=fg_color)
        self.style.configure("TLabelframe.Label", background=bg_color, foreground=fg_color,
                             font=("Segoe UI", 10, "bold"))
        self.style.configure("TLabel", background=bg_color, foreground=fg_color,
                             font=("Segoe UI", 10))
        self.style.configure("TLabel", background=bg_color, foreground=fg_color,
                             font=("Segoe UI", 9, "bold"), padding=(10, 5))
        self.style.map("TButton",
                       background=[("active", "#2ea043"), ("disabled", "#21262d")])
        
        self.style.configure("TEntry", fieldbackground=entry_bg,
                             foreground=fg_color, insertcolor=fg_color)
        self.style.configure("Accent.TButton", background="#da3633", foreground="#ffffff")
        self.style.map("Accent.TButton",
                       background=[("active", "#f85149"), ("disabled", "#21262d")])
        
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
                              bg="#0d1117", fg="#58a6ff")
        title_lable.pack(side="left")
        self.status_indicator = tk.Label(header_frame, text="DISCONNECTED", 
                                         font=("JetBrains Mono", 18, "bold"), 
                                        bg="#0d1117", fg="#f85149")
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
                                   bg='#0d1117', fg='#7ee787')
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
                              bg="#0d1117", fg="#8b949e")
            label.pack(side="left", padx=(5, 0))
            self.stats_labels[name] = label
        
        
        
            
        
        
        
    def setup_chart(self):
        pass

    def connect_ib(self):
        try:
            host = self.host_var.get()
            port = int(self.port_var.get())
            
            def connect_thread():
                try:
                    self.ib_app.connect()
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
                self.status_indicator.config(text="CONNECTeD", fg="#7ee787")
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
        pass
    
    def stop_stream(self):
        pass
    
    def recalibrate_model(self):
        pass
    
    def on_tick_data(self):
        pass
    
    
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
    
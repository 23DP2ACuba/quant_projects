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


class Dashboard:
    def __init__(self, root):
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
        
        self.bar_lock = thr.lock()
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
                       background=[("active", "#2ea043"), ("diabled", "#21262d")])
        
        self.style.configure("TEntry", fieldbackground=entry_bg,
                             foreground=fg_color, insertcolor=fg_color)
        self.style.configure("Accent.TButton", background="#da3633", foreground="#ffffff")
        self.style.map("Accent.TButton",
                       background=[("active", "#f85149"), ("diabled", "#21262d")])
        
    def setup_ui(self):
        pass
    
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

    




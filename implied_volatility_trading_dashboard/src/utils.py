import tkinter as tk
import threading
import time
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime

class DisplayUtils:
    @staticmethod
    def create_root():
        root = tk.Tk()
        return root
    
    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.ststus_text.see(tk.END)
        self.root.update_idletasks()
        
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
        
        while 1 not in self.ib_app.historical_data  \
            and (time.time() - start_time) < timeout:
            time.sleep(0.1)
            
        if 1 in self.ib_app.historical_data:
            data = self.ib_app.historical_data[1]
            if len(data) > 0:
                self.equity_data = pd.DataFrame(data)
                self.equity_data["date"] = pd.to_dataframe(self.equity_data["date"])   
                self.equity_data.set_index("date", inplace = True)
                
                self.equity_data["implied_vol"] = self.equity_data["close"]
                
                self.log_message(f"Recieved {len(self.equity_data)}, \
                    implied volatility points for {symbol}")
                self.log_message(f"Date Range:  {self.equity_data.index.min()} \
                    : {self.equity_data.index.max()}")
                self.log_message("NOTE: ALL VALUES ARE ANNUALIZED")
                
                self.process_implied_volatility()
                self.analyze_btn.config(state="normal")
                
            else:
                self.log_message("No vol data recieved")
                self.equity_data = None
                
        else:
            self.log_message("NOTE: No vol data recieved \
                - MAY NOT BE AVAILABLE FOR SYMBOL")
            self.equity_data = None
            
            
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
            
    def disconnect_ib(self):
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

import warnings
import tkinter as tk
from dashboard import Dashboard
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

def main():
    root = tk.Tk()
    app = Dashboard(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
    
if __name__ == "__main__":
    main()
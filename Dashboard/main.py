import tkinter as tk
import warnings
from dashboard import EarningsDashboard
warnings.filterwarnings("ignore")
        
        
def main():
    root = tk.Tk()
    app = EarningsDashboard(root)
    root.mainloop()
    
if __name__ == "__main__":
    main()
        
        
        
        
        
        
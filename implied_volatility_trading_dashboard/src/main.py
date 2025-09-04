from utils import DisplayUtils
from ImpliedVolatilityDashboard import ImpliedVolatilityDashboard
def main():
    root = DisplayUtils.create_root()
    app = ImpliedVolatilityDashboard(root)
    root.mainloop()
    
    
if __name__ == "__main__":
    main()
    
    
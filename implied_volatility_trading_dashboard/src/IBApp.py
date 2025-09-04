from ibapi.client import EClient
from ibapi.wrapper import EWrapper
import warnings
warnings.filterwarnings('ignore')

class IBApp(EClient, EWrapper):
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False
        self.historical_data = {}
        
    def error(self):
        pass

    def nextValidId(self):
        pass
    
    def historicalData(self):
        pass
    
    def HistoricalDataEnd(self):
        pass
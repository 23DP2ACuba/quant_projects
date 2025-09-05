from ibapi.client import EClient
from ibapi.wrapper import EWrapper
import warnings
warnings.filterwarnings('ignore')

class IBApp(EClient, EWrapper):
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False
        self.historical_data = {}
        
    def error(self, reqId, errorCode, errorString):
        print(f"Error: {reqId}, {errorCode}, {errorString}")
        

    def nextValidId(self):
        pass
    
    def historicalData(self, reqId, bar):
        if reqId not in self.historical_data:
            self.historical_data[reqId] = []
        self.historical_data[reqId].append({
            "date": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        })
        
    
    def historicalDataEnd(self, reqId):
        print(f"Historical data recieved for {reqId}")
    
    
        
        

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
import threading as thr
from dataclasses import dataclass
from ibapi.contract import Contract 

class IBApp(EWrapper, EClient):
    def __init__(self, callback=None):
        EClient.__init__(self, self)
        self.callback = callback
        self.last_price = self.bid_price = self.ask_price = None
        self.historical_data = {}
        self.hist_done = thr.Event()
        self.connected = False
        
    def error(self, reqId, errorCode, errorStr, advancedOrderRejectJson=""):
        if errorCode in [2104, 2106, 2158, 2176]:
            return 
        if errorCode == 10167:
            print(f"Using delayed market data")
            
        print(f"Error {reqId}: {errorCode} - errorString")
        
    def nextValidId(self, orderId):
        self.connected = True
        print("Connected to TWS")
        
    def historicalData(self, reqId, bar):
        if reqId not in self.historical_data:
            self.historical_data[reqId] = []
        self.historical_data[reqId].append(
            {
                'o': bar.open, 
                'h': bar.high, 
                'l': bar.low, 
                'c': bar.close
            }
        )
    def historicalDataEnd(self, reqId, start, end):
        self.hist_done = True
    
    def tickPrice(self, reqId, tickType, price, attrib):
        if price <= 0:
            return 
        if tickType == 4:
            self.last_price = price
        elif tickType == 1:
            self.bid_price = price
        elif tickType == 2:
            self.ask_price = price

    
    def create_contract(self, symbol):
        contract = Contract()
        contract.symbol = symbol.upper()
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        return contract
    
    
        
        
@dataclass
class BarData:
    open: float
    high: float
    low: float
    close: float   
    
class OHLCBar:
    def __init__(self, timestamp, open_price):
        self._data = self._data = BarData(
            open_price, open_price, open_price, open_price
        )
        self.timestamp = timestamp
        self.tick_count = 1
        self.regime = 0   

    @property 
    def open(self): return self._data.open
    @property 
    def high(self): return self._data.high
    @property 
    def low(self): return self._data.low
    @property 
    def close(self): return self._data.close
    
    @property 
    def volatility(self):
        return (self.high - self.low) / self.close if self.close > 0 else 0
    
        
    def update(self, price):
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.tick_count += 1
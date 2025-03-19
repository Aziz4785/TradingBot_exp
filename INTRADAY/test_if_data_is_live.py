#the goal is to know if the data we receive is delayed or not
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.ticktype import TickTypeEnum
import threading
import time
import pandas as pd
class TestApp(EClient, EWrapper):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = {}
        self.historical_close = None  # To store the last bar's close from historical data
        self.live_close = None         # To store the live data close tick
        self._hist_complete = False    # Flag to know when historical data is finished

    def nextValidId(self, orderId):
        self.orderId = orderId

    def nextId(self):
        self.orderId += 1
        return self.orderId
    
    # --------------------
    # Historical Data Callbacks
    # --------------------
    def historicalData(self, reqId, bar):
        # Called repeatedly for each bar; print out each bar if you wish.
        #print(f"Historical data (reqId={reqId}): {bar}")
        # Keep updating the historical_close so that when the last bar is received it holds the close.
        self.historical_close = bar.close
        row = {"Date": bar.date, "Open": bar.open, "High": bar.high,
               "Low": bar.low, "Close": bar.close, "Volume": bar.volume}
        if reqId not in self.data:
            self.data[reqId] = pd.DataFrame([row])
        else:
            self.data[reqId] = pd.concat([self.data[reqId], pd.DataFrame([row])])

    def historicalDataEnd(self, reqId, start, end):
        print(f"Historical Data Ended for reqId={reqId}. Start: {start}, End: {end}")
        self._hist_complete = True
        # Optionally, cancel further historical data (not strictly needed once the end callback is received)
        #self.cancelHistoricalData(reqId)
        #print(f"Stored historical close value: {self.historical_close}")

    # --------------------
    # Live Data Callbacks
    # --------------------
    def marketDataType(self, reqId, marketDataType):
        print(f"MarketDataType. ReqId: {reqId}, Type: {marketDataType}")
    def tickPrice(self, reqId, tickType, price, attrib):
        tickName = TickTypeEnum.toStr(tickType)
        print(f"Live tick (reqId={reqId}): {tickName} price={price}")
        
        # Choose the tick type that corresponds to the "close" price.
        # For example, if your live data provides a tick with type "CLOSE" use that.
        # Some users may find that "LAST" (or another tick type) is the one they want.
        if tickName.upper() in ["CLOSE", "LAST"]:
            self.live_close = price
            # Once you have both values, compare them.
            if self._hist_complete and self.historical_close is not None:
                # Use a tolerance if needed when comparing floats.
                tolerance = 0.001
                if abs(self.historical_close - self.live_close) < tolerance:
                    print("The historical close and live close values match!")
                else:
                    print("Mismatch: historical close =", self.historical_close, 
                          "live close =", self.live_close)
            else:
                print("Historical close not yet available or complete.")

    def tickSize(self, reqId, tickType, size):
        tickName = TickTypeEnum.toStr(tickType)
        #print(f"Live tick size (reqId={reqId}): {tickName} size={size}")

    def error(self, reqId, errorCode, errorString, advancedOrderReject=""):
        print(f"Error. reqId: {reqId}, errorCode: {errorCode}, errorString: {errorString}")

def usTechStk(symbol, sec_type="STK", currency="USD", exchange="SMART"):
    contract = Contract()
    contract.symbol = symbol
    contract.secType = sec_type
    contract.currency = currency
    contract.exchange = exchange
    return contract 

def dataDataframe(TradeApp_obj, symbol):
    """Returns extracted historical data in dataframe format"""
    #print(TradeApp_obj.data)
    df = pd.DataFrame(TradeApp_obj.data[2]) #try 2
    #df.set_index("Date", inplace=True)
    return df

def main():
    app = TestApp()
    app.connect("127.0.0.1", 7497, clientId=0)
    
    # Start the socket thread.
    thread = threading.Thread(target=app.run)
    thread.start()
    
    # Give time for connection
    time.sleep(1)
    
    contract =usTechStk('META')
    # --------------------
    # Request Historical Data
    # --------------------
    app.reqHistoricalData(reqId=app.nextId(), 
                          contract=contract,
                          endDateTime='',
                          durationStr="5 D",
                          barSizeSetting='30 mins',
                          whatToShow='TRADES', #ADJUSTED_LAST
                          useRTH=0,
                          formatDate=1,
                          keepUpToDate=0,
                          chartOptions=[])
    time.sleep(2)
    df = dataDataframe(app, 'META')

    if df.empty:
        print(f"Empty dataframe for 'META'.")

    # Use the latest bar from the dataframe
    df = df.sort_index() #maybe not useful
    last_bar = df.iloc[-1]
    close_price = last_bar['Close']
    print("close price : ",close_price)
    # --------------------
    # Request Live Data
    # --------------------
    app.reqMarketDataType(1)
    

    #app.reqMktData(app.nextId(), contract, "", False, False, [])
    # Let the app run for a while to receive data
    time.sleep(10)
    
    # Disconnect cleanly
    app.disconnect()
    thread.join()

if __name__ == "__main__":
    main()

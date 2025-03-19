from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import pandas as pd
import threading
import time
import numpy as np
from utils import *
pd.options.display.max_rows = None

CANDLE_SIZE = '30 mins'
cleanup_files(['raw.csv'])

class TradeApp(EWrapper, EClient): 
    def __init__(self): 
        EClient.__init__(self, self) 
        self.data = {}
        
    def historicalData(self, reqId, bar):
        if reqId not in self.data:
            self.data[reqId] = pd.DataFrame([{"Date":bar.date,"Open":bar.open,"High":bar.high,"Low":bar.low,"Close":bar.close,"Volume":bar.volume}])
            print(f"self.data[{reqId}]")
        else:
            self.data[reqId] = pd.concat((self.data[reqId],pd.DataFrame([{"Date":bar.date,"Open":bar.open,"High":bar.high,"Low":bar.low,"Close":bar.close,"Volume":bar.volume}])))

    def error(self, reqId, errorCode, errorString, advancedOrderReject=""):
        print(f"reqId: {reqId}, errorCode: {errorCode}, errorString: {errorString}, orderReject: {advancedOrderReject}")

def usTechStk(symbol,sec_type="STK",currency="USD",exchange="SMART"):
    contract = Contract()
    contract.symbol = symbol
    contract.secType = sec_type
    contract.currency = currency
    contract.exchange = exchange
    return contract 

# def histData_old(req_num,contract,duration,candle_size):
#     """extracts historical data"""
#     app.reqHistoricalData(reqId=req_num, 
#                           contract=contract,
#                           endDateTime='',
#                           durationStr=duration,
#                           barSizeSetting=candle_size,
#                           whatToShow='ADJUSTED_LAST', #or TRADES??
#                           useRTH=1,
#                           formatDate=1,
#                           keepUpToDate=0,
#                           chartOptions=[])	 # EClient function to request contract details

def histData(req_num, contract, duration, candle_size, end_date=''):
    """extracts historical data"""
    app.reqHistoricalData(reqId=req_num,
                          contract=contract,
                          endDateTime=end_date, # Use the provided end_date here
                          durationStr=duration,
                          barSizeSetting=candle_size,
                          whatToShow='TRADES',
                          useRTH=0, #put 1
                          formatDate=1,
                          keepUpToDate=0,
                          chartOptions=[])
    
def websocket_con():
    app.run()
    
app = TradeApp()
app.connect(host='127.0.0.1', port=7497, clientId=23) #port 4002 for ib gateway paper trading/7497 for TWS paper trading
con_thread = threading.Thread(target=websocket_con, daemon=True)
con_thread.start()
time.sleep(10) # some latency added to ensure that the connection is established
app.reqMarketDataType(1)
time.sleep(10)
tickers = ["OKLO","META","AAPL","MSFT","JPM","V","AMZN","NKE","LRCX","PFE","VICI","IPG","F",
           "TRV","LW","DELL","HIMS","TSLA","AXON","NET","SHOP",
           "TTD","SNOW","TEAM","GOOG","ORCL","KO","WU","JNJ","RKLB",
           "IDXX","RMD","TSN","CME","BKR","NUE","ODFL","UBER","NVDA","AVGO",
           "AMD","NFLX","ASTS","CVS","SPY","CRWD","NEE","SBUX","LMT","COST",
           "QQQ","LLY","CRM","GS","WFC","VEEV","PM","IBM","VZ","XOM","VRTX",
           "FTNT","MA","PYPL","ANET","KKR","BAC","ABBV","PG","UNH","UA","UAL",
           "HD","SHW","CAT","AMGN","MCD"] 

#DO NOT USE THESE STOCKS : 
# LYB, AAP
#tickers = ["AAPL","LRCX"]
#end_date_str = "20250204 00:00:00 UTC" # Set the end date to today
"""for ticker in tickers:
    histData(tickers.index(ticker),usTechStk(ticker),duration='3 D',candle_size= '30 mins')
    time.sleep(5)"""
for ticker in tickers:
    print(ticker)
    histData(tickers.index(ticker), usTechStk(ticker), duration='230 D', candle_size= CANDLE_SIZE) 
    time.sleep(70) # Adding sleep to avoid pacing violation. Adjust as needed based on IBKR limits.

###################storing trade app object in dataframe#######################

# def dataDataframe_old(symbols,TradeApp_obj):
#     "returns extracted historical data in dataframe format"
#     df_data = {}
#     for i, symbol in enumerate(symbols):
#         df_data[symbol] = pd.DataFrame(TradeApp_obj.data[i])
#         df_data[symbol].set_index("Date",inplace=True)
#         # Add columns for prices 2 days ago and 1 day ago
#         df_data[symbol]['Close_1_day_ago'] = df_data[symbol]['Close'].shift(1+12)
#         df_data[symbol]['Close_2_days_ago'] = df_data[symbol]['Close'].shift(2+2*12)
#         """df_data[symbol]['dayOpen'] = #here i want the open price of the day (so the open of the first 30min bar of the day)
#         df_data[symbol]['prevDayOpen'] = #here i want the open price of the prev day (so the open of the first 30min bar of the prev day)
#         df_data[symbol]['prev2DayOpen'] = #same but for 2 days ago
#         df_data[symbol]['prevDayClose'] = #here i want the close price of the prev day (so the close of the last 30min bar of the prev day)
#         df_data[symbol]['prev2DayClose'] = #same but for 2 days ago"""
#         df_data[symbol]['Open_1_day_ago'] = df_data[symbol]['Open'].shift(1+12)

#         df_data[symbol]['return_1d'] = df_data[symbol]['Close'] / df_data[symbol]['Close_1_day_ago']
#         df_data[symbol]['return_2d'] = df_data[symbol]['Close'] / df_data[symbol]['Close_2_days_ago']
#         #df_data[symbol]['open_to_prev_close'] = df_data[symbol]['day_Open'] / df_data[symbol]['prevDayClose']
        
#     return df_data



def dataDataframe(symbols, TradeApp_obj):
    """returns extracted historical data in dataframe format"""
    df_data = {}
    
    for i, symbol in enumerate(symbols):
        print(symbol)
        # --- 1. Create a proper DataFrame from the raw data ---
        df = pd.DataFrame(TradeApp_obj.data[i])
        
        # Convert "Date" to datetime and make it the DataFrame index
        # (removing the trailing " US/Eastern" if needed)
        df["Date"] = df["Date"].str.replace(" US/Eastern", "", regex=False)
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d %H:%M:%S")
        
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)  # ensure it's sorted by datetime
        
        df = add_market_time_column(df)
        #df = df.loc["2025-01-20":"2025-01-22"] #for debugging
        #df.to_csv("debugging_from_raw3.csv", index=True)
        # --- 2. Add dayOpen ---
        # dayOpen = the open price of the first 30-min bar of that day
        df["day"] = df.index.date  # so we can group by date easily
        mask = df["market_time"] == "RTH"
        df_rth = df.loc[mask].copy()

        df_rth["dayOpen"] = df_rth.groupby("day")["Open"].transform("first")
        df_rth["dayClose"] = df_rth.groupby("day")["Close"].transform("last")
        df_rth["dayHigh"] = df_rth.groupby("day")["High"].transform("max")
        df_rth["dayLow"] = df_rth.groupby("day")["Low"].transform("min")

        df_rth = calculate_historical_highs(df_rth)
        df_rth = calculate_historical_Volumes(df_rth)
        df_rth = add_historical_values(df_rth)
        # Now create (or reset) the columns in the main df
        df = initialize_columns(df)

        # Copy the shifted values back into the original df for the RTH rows
        cols_to_copy = [
            "dayOpen", "dayClose", "dayHigh", 
            "dayHigh_1", "dayHigh_2", "dayHigh_3","dayLow","prevDayLow",
            "prevDayOpen", "prevDayClose", "prev2DayOpen", "prev2DayClose",
            "Close_1_day_ago", "Close_2_days_ago", "Open_1_day_ago","Volume1","Volume2","Volume3"
        ]
        df.loc[mask, cols_to_copy] = df_rth[cols_to_copy]

        

        # daily returns
        df = add_daily_returns(df)

        df = add_PM_columns(df)


        mask_ah = df["market_time"] == "AH"
        df_ah = df.loc[mask_ah].copy()
        df_ah["AH_max"] = df_ah.groupby("day")["High"].transform("max")
        df["AH_max"] = np.nan
        df.loc[mask_ah, "AH_max"] = df_ah["AH_max"]
        df["AH_max"] = df.groupby("day")["AH_max"].transform("max")

        ahmax_per_day = df.groupby("day")["AH_max"].first().sort_index()
        ahmax_per_day_shifted = ahmax_per_day.shift()
        df["AH_max_1dayago"] = df["day"].map(ahmax_per_day_shifted)


        # --- 3. Add EMA columns computed on RTH rows ---
        # Compute the EMAs per day (resetting each day) using the RTH Close values
        #mask_rth = df["market_time"] == "RTH" #todo add ema 30 and 8
        df = calculate_emas(df)
        df = calculate_stds(df)
        #df = calculate_smas(df)
        # Drop the helper 'day' column if you prefer
        df.drop("day", axis=1, inplace=True)
        
        df_data[symbol] = df
    
    return df_data

#extract and store historical data in dataframe
historicalData = dataDataframe(tickers,app)
#historicalData.to_csv('raw.csv', encoding='utf-8', index=False)
# Print the historical data for verification
"""for ticker in tickers:
    print(f"Historical Data for {ticker}:")
    print(historicalData[ticker].head(130))
    print("\n")"""

available_features_by_stock = {}
all_data = pd.concat(historicalData.values(), keys=historicalData.keys())
all_data.to_csv('raw_before_clean.csv', encoding='utf-8')
all_data = all_data[all_data["market_time"] == "RTH"]
for ticker in tickers:
    #print("percentage of None for the column dayOpen")
    #print(all_data.loc[ticker]["dayOpen"].isnull().mean())
    #1) find the columns that have less than 10% None for that ticker
    available_features_by_stock[ticker] = list(all_data.loc[ticker].columns[all_data.loc[ticker].isnull().mean() < 0.1])
#all_data = all_data.dropna()


#save available_features_by_stock to a json file
import json
with open('available_features_by_stock.json', 'w') as fp:
    json.dump(available_features_by_stock, fp, indent=4)
all_data.to_csv('raw.csv', encoding='utf-8')
print("Saved all historical data to raw.csv")

time.sleep(10) # Keep the script running for a while to receive data
app.disconnect()
con_thread.join()

#https://www.interactivebrokers.com/campus/trading-lessons/python-receiving-market-data/
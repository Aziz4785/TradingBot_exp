from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
import pandas as pd
from datetime import datetime, timedelta
import threading
import time
import datetime
import pickle
import numpy as np
from decimal import Decimal
from utils import *
pd.options.display.max_rows = None

# File to store positions_dict
# POSITIONS_DICT_FILE = "positions_dict.pkl"
# try:
#     with open(POSITIONS_DICT_FILE, "rb") as f:
#         positions_dict = pickle.load(f)
#     print("Loaded positions_dict from disk.")
# except FileNotFoundError:
#     positions_dict = {}
#     print("No positions_dict found on disk. Starting fresh.")

POSITIONS_DICT_FILE = "old_stuff12/positions_dict.json"
config_path = "old_stuff12/models_to_use.json"
models_dir  = "old_stuff12/allmodels"
scalers_dir = "old_stuff12/allscalers"

PAPER_TRADING=False
EMA_FEATURES = False
try:
    with open(POSITIONS_DICT_FILE, "r") as f:
        positions_dict = json.load(f)
    print("Loaded positions_dict from disk.")
except FileNotFoundError:
    positions_dict = {}
    print("No positions_dict found on disk. Starting fresh.")




max_buget_per_trade = 1100
with open(config_path, 'r') as file:
    config = json.load(file)
    stocks_from_json= get_stock_symbols_from_json(config)
with open("old_stuff12/bins_json.json", "r") as f:
    loaded_bin_dict_json = json.load(f)
    for col in loaded_bin_dict_json:
        loaded_bin_dict_json[col] = np.array(loaded_bin_dict_json[col])

from SuperModel import * 
super_model = SuperModel(config_path, models_dir, scalers_dir)

six_bins_columns = extracts_all_features_from_json(config)

# Global dictionary to store trade info (entry time, order IDs, etc.)
trade_info = {}  # key: ticker, value: dict with keys: entry_time, buy_price, quantity, parent_order_id, tp_order_id, sl_order_id

# -------------------------
# IBKR API Classes and Functions
# -------------------------
class TradeApp(EWrapper, EClient): 
    def __init__(self): 
        EClient.__init__(self, self) 
        self.data = {}
        self.total_cash_value = None
        self.pos_df = pd.DataFrame(columns=['Account', 'Symbol', 'SecType',
                                            'Currency', 'Position', 'Avg cost'])
        self.order_df = pd.DataFrame(columns=['PermId', 'ClientId', 'OrderId',
                                              'Account', 'Symbol', 'SecType',
                                              'Exchange', 'Action', 'OrderType',
                                              'TotalQty', 'CashQty', 'LmtPrice',
                                              'AuxPrice', 'Status'])
        self.portfolio = {}

    def historicalData(self, reqId, bar):
        # Build a DataFrame row-by-row
        row = {"Date": bar.date, "Open": bar.open, "High": bar.high,
               "Low": bar.low, "Close": bar.close, "Volume": bar.volume}
        if reqId not in self.data:
            self.data[reqId] = pd.DataFrame([row])
        else:
            self.data[reqId] = pd.concat([self.data[reqId], pd.DataFrame([row])])
       
    def error(self, reqId, errorCode, errorString, advancedOrderReject=""):
        print(f"reqId: {reqId}, errorCode: {errorCode}, errorString: {errorString}, orderReject: {advancedOrderReject}")


    def nextValidId(self, orderId):
        super().nextValidId(orderId)
        self.nextValidOrderId = orderId
        print("NextValidId:", orderId)

    def updateAccountValue(self, key: str, val: str, currency: str, accountName: str):
        #This value (often labeled as NetLiquidation) represents your total account value (cash plus the market value of positions).
        print(f"Account Update - {key}: {val} {currency} (Account: {accountName})")
        # Optionally, update your local record of capital, e.g.:
        if key == "NetLiquidation":
            self.current_capital = float(val)

    def updatePortfolio(self, contract: Contract, position: float, marketPrice: float, marketValue: float,
                      averageCost: float, unrealizedPNL: float, realizedPNL: float, accountName: str):
        #The updatePortfolio callback is triggered whenever there’s an update to any of your positions
        print(f"Portfolio Update - {contract.symbol}: Position={position}, Market Value={marketValue}, "
            f"Unrealized P&L={unrealizedPNL}, Realized P&L={realizedPNL}")
    
    def pnl(self, reqId: int, dailyPnL: float, unrealizedPnL: float, realizedPnL: float):
        print(f"Daily PnL Update - ReqId: {reqId}, DailyPnL: {dailyPnL}, "
            f"Unrealized: {unrealizedPnL}, Realized: {realizedPnL}")

    def position(self, account: str, contract: Contract, position: Decimal, avgCost: float):
        super().position(account, contract, position, avgCost)
        
        print(f"Position update: Account: {account}, Symbol: {contract.symbol}, Position: {position}, AvgCost: {avgCost}")
        
        symbol = contract.symbol
        pos = float(position)
        
        global positions_dict, POSITIONS_DICT_FILE
        
        if pos != 0:
            self.portfolio[symbol] = {"position": pos, "avgCost": avgCost, "account": account}
            # Update positions_dict with open date if new position
            if symbol not in positions_dict:
                positions_dict[symbol] = {
                    'open_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'quantity': pos,
                    'avg_cost': avgCost
                }
                with open(POSITIONS_DICT_FILE, "w") as f:
                    json.dump(positions_dict, f)
        else:
            if symbol in self.portfolio:
                print(f"Position for {symbol} closed. Removing from portfolio.")
                del self.portfolio[symbol]
            if symbol in positions_dict:
                print(f"Position for {symbol} closed. Removing from positions_dict.")
                del positions_dict[symbol]
                with open(POSITIONS_DICT_FILE, "w") as f:
                    json.dump(positions_dict, f)

    def positionEnd(self):
        print("PositionEnd: All positions have been updated.")
        
    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        if tag == "TotalCashValue":
            try:
                self.total_cash_value = float(value)
            except ValueError:
                self.total_cash_value = None
            print(f"Total Cash Value: {value} {currency}")

    #def accountSummaryEnd(self, reqId: int):
        #print("Account Summary Received")
        
    """def openOrder(self, orderId, contract, order, orderState):
        super().openOrder(orderId, contract, order, orderState)
        dictionary = {"PermId": order.permId, "ClientId": order.clientId, "OrderId": orderId, 
                      "Account": order.account, "Symbol": contract.symbol, "SecType": contract.secType,
                      "Exchange": contract.exchange, "Action": order.action, "OrderType": order.orderType,
                      "TotalQty": order.totalQuantity, "CashQty": order.cashQty, 
                      "LmtPrice": order.lmtPrice, "AuxPrice": order.auxPrice, "Status": orderState.status}
        self.order_df = self.order_df.append(dictionary, ignore_index=True)"""


def usTechStk(symbol, sec_type="STK", currency="USD", exchange="SMART"):
    contract = Contract()
    contract.symbol = symbol
    contract.secType = sec_type
    contract.currency = currency
    contract.exchange = exchange
    return contract 

def histData(req_num, contract, duration, candle_size):
    """Extracts historical data"""
    #print(f"histData({req_num},{contract},{duration},{candle_size})")
    app.reqHistoricalData(reqId=req_num, 
                          contract=contract,
                          endDateTime='',
                          durationStr=duration,
                          barSizeSetting=candle_size,
                          whatToShow='TRADES', #ADJUSTED_LAST
                          useRTH=0,
                          formatDate=1,
                          keepUpToDate=0,
                          chartOptions=[])

def createBracketOrder(parent_order_id, action, quantity, entry_price, take_profit_price, stop_loss_price):
    parent = Order()
    parent.orderId = parent_order_id
    parent.action = action
    parent.orderType = "MKT" #"LMT"
    parent.totalQuantity = quantity
    parent.lmtPrice = entry_price
    parent.transmit = False  # Do not transmit until the children are ready

    # Take Profit Order (child)
    profit_taker = Order()
    profit_taker.orderId = parent_order_id + 1
    profit_taker.parentId = parent_order_id  # Link to parent
    profit_taker.action = "SELL" if action == "BUY" else "BUY"
    profit_taker.orderType = "LMT"
    profit_taker.totalQuantity = quantity
    profit_taker.lmtPrice = take_profit_price
    profit_taker.transmit = False  # Wait until stop loss is ready
    profit_taker.tif = 'GTC'

    # Stop Loss Order (child)
    stop_loss = Order()
    stop_loss.orderId = parent_order_id + 2
    stop_loss.parentId = parent_order_id  # Link to parent
    stop_loss.action = "SELL" if action == "BUY" else "BUY"
    stop_loss.orderType = "STP"
    stop_loss.totalQuantity = quantity
    stop_loss.auxPrice = stop_loss_price
    stop_loss.transmit = True  # This final transmission sends all orders together
    stop_loss.tif = 'GTC'

    return parent, profit_taker, stop_loss


# def limitOrder(direction, quantity, lmt_price):
#     order = Order()
#     order.action = direction
#     order.orderType = "LMT"
#     order.totalQuantity = quantity
#     order.lmtPrice = lmt_price
#     return order

# def marketOrder(direction, quantity):
#     order = Order()
#     order.action = direction
#     order.orderType = "MKT"
#     order.totalQuantity = quantity
#     order.tif = "IOC"
#     return order

# def stopOrder(direction, quantity, st_price):
#     order = Order()
#     order.action = direction
#     order.orderType = "STP"
#     order.totalQuantity = quantity
#     order.auxPrice = st_price
#     return order

def dataDataframe(TradeApp_obj, symbols, symbol):
    """Returns extracted historical data in dataframe format"""
    df = pd.DataFrame(TradeApp_obj.data[symbols.index(symbol)])
    #df.set_index("Date", inplace=True)
    return df

def websocket_con():
    app.run()

# -------------------------
# Connect to IBKR and define symbols, capital, etc.
# -------------------------
app = TradeApp()
if PAPER_TRADING:
    port_number = 7497
else:
    port_number = 7496

app.connect(host='127.0.0.1', port=port_number, clientId=23)
con_thread = threading.Thread(target=websocket_con, daemon=True)
con_thread.start()
time.sleep(2)  # wait to ensure connection is established
tickers = sorted(config.keys(), key=lambda ticker: config[ticker]["specificity"], reverse=True)
app.reqMarketDataType(1)
time.sleep(2) 

# (Optional) Get initial positions to compare later if needed.
"""
app.reqPositions()
time.sleep(5)
initial_pos = {key: 0 for key in tickers}
initial_pos_df = app.pos_df[app.pos_df["SecType"]=="STK"]
for key in initial_pos_df["Symbol"]:
    if key in initial_pos:
        initial_pos[key] = int(initial_pos_df[initial_pos_df["Symbol"]==key]["Position"].values[0])
"""
#from datetime import datetime, timedelta
def close_old_positions(app, days_threshold=1):
    """
    Closes all positions that have been open for more than specified days
    Args:
        app: TradeApp instance
        days_threshold: Number of days after which to close positions (default: 1)
    """
    current_time = datetime.now()
    threshold_date = current_time - timedelta(days=days_threshold)
    
    # Ensure we have the latest position data
    app.reqPositions()  # Request position updates
    time.sleep(2)       # Give time for position updates to come through
    
    positions_to_close = {}
    
    # Check each position in portfolio against positions_dict
    for symbol, pos_data in app.portfolio.items():
        if symbol in positions_dict:
            open_date_str = positions_dict[symbol].get('open_date')
            if open_date_str:
                try:
                    open_date = datetime.strptime(open_date_str, '%Y-%m-%d %H:%M:%S')
                    if open_date < threshold_date:
                        positions_to_close[symbol] = {
                            'position': pos_data['position'],
                            'account': pos_data['account']
                        }
                except ValueError:
                    print(f"Warning: Invalid date format for {symbol}")

    if not positions_to_close:
        print("No positions older than 1 day found.")
        return

    # Process each position that needs to be closed
    for symbol, pos_info in positions_to_close.items():
        position = pos_info['position']
        if position == 0:
            continue
            
        print(f"Closing position for {symbol} - Quantity: {position}")
        
        # Create contract
        contract = usTechStk(symbol)
        
        # First, cancel any open orders (bracket orders) for this symbol
        app.reqAllOpenOrders()
        time.sleep(0.5)  # Wait for open orders to be received
        
        # Get current order ID
        if not hasattr(app, 'nextValidOrderId'):
            app.reqIds(-1)
            time.sleep(0.5)
        
        order_id = app.nextValidOrderId
        
        # Create market order to close position
        action = "SELL" if position > 0 else "BUY"
        quantity = abs(position)
        
        market_order = Order()
        market_order.orderId = order_id
        market_order.action = action
        market_order.orderType = "MKT"
        market_order.totalQuantity = quantity
        market_order.transmit = True
        
        # Place the closing order
        app.placeOrder(order_id, contract, market_order)
        
        # Update positions_dict
        if symbol in positions_dict:
            del positions_dict[symbol]
        
        # Save updated positions_dict
        with open(POSITIONS_DICT_FILE, "w") as f:
            json.dump(positions_dict, f)
        
        app.nextValidOrderId += 1  # Increment order ID
        
        print(f"Placed market {action} order for {quantity} shares of {symbol}")
    
    time.sleep(1)  # Give time for orders to process
    
def extract_features(df):
    df["Date"] = df["Date"].str.replace(" US/Eastern", "", regex=False)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d %H:%M:%S")
    
    df['day_of_week'] = df['Date'].dt.dayofweek 
    df['time_in_minutes'] = df['Date'].dt.hour * 60 + df['Date'].dt.minute

    #df['time_in_minutes'] = ((df['Date'].dt.hour * 60 + df['Date'].dt.minute + 15) // 30) * 30

    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)  # ensure it's sorted by datetime

    df = add_market_time_column(df)
    df["day"] = df.index.date
    mask = df["market_time"] == "RTH"
    df_rth = df.loc[mask].copy()
    df_rth = calculate_daily_ohlc(df_rth)
    df_rth=calculate_historical_highs(df_rth)
    df_rth = add_historical_values(df_rth)

    df['day'] = df.index.date
    unique_dates = sorted(df['day'].unique())
    if len(unique_dates) < 3:
        return None  # Not enough data for feature extraction.
    
    df = initialize_columns(df)

    cols_to_copy = [
        "dayOpen", "dayClose", "dayHigh", 
        "dayHigh_1", "dayHigh_2", "dayHigh_3","dayLow","prevDayLow",
        "prevDayOpen", "prevDayClose", "prev2DayOpen", "prev2DayClose",
        "Close_1_day_ago", "Close_2_days_ago", "Open_1_day_ago"
    ]
    df.loc[mask, cols_to_copy] = df_rth[cols_to_copy]
    
    df = add_daily_returns(df)
    df = add_PM_columns(df)

    df = add_AH_columns(df)

    # --- 3. Add EMA columns computed on RTH rows ---
    # Compute the EMAs per day (resetting each day) using the RTH Close values
    if EMA_FEATURES == True:
        df = calculate_emas(df)
    df = calculate_stds(df)
    df = add_shifted_fractional_diff_from_DEPRADO(df)
    
    #df = add_volatilities(df)
    #df = add_momentum(df)
    df = additional_ratios(df)
    df = additional_Close_ratios(df)
    df = additional_PM_ratios(df)
    df = calculate_slopes(df)
    
    
    #TODO ADD a boolean to see if there is a local minima on the  dayHigh_3
    if EMA_FEATURES == True:
        df = calculate_ema_ratios(df)
    df = calculate_highs_and_lows(df)
    df.drop("day", axis=1, inplace=True)
    df = df[df['market_time'] == 'RTH']

    #last_row_nan = check_last_row_nans(df)['columns_with_nan']
    #print("columns Nan in last row : ",last_row_nan)
    #df = df.dropna()

    row_features = df.iloc[-1].copy()

    row_features = assign_bin_categories(row_features, loaded_bin_dict_json)
    #for colum in six_bins_columns:
        #assign_class(row_features, colum, bin_dict)
    #if row_features.isnull().any():
        #return None
            
    return row_features

# -------------------------
# Main Strategy Function
# -------------------------
def main(total_available_cash):
    print("main()")
    global positions_dict
    app.reqPositions()
    time.sleep(2)  
    print("total_available_cash : ",total_available_cash)
    nbr=0
    for ticker in tickers:
        if total_available_cash<max_buget_per_trade:
            print(" not enough cash to trade")
            break
        if ticker =='LYB' or ticker =='BIO':
            continue
        # Only process stocks that are not already in positions_dict.
        if ticker in positions_dict:
            print(f"Skipping {ticker} because it is already in positions_dict (bought on {positions_dict[ticker]}).")
            continue

        print(f"Processing ticker: {ticker}")

        # Clear previous data for this ticker and request historical data
        app.data.pop(tickers.index(ticker), None)
        histData(tickers.index(ticker), usTechStk(ticker), '8 D', '30 mins')
        time.sleep(2)  # Allow time for data to be collected

        if tickers.index(ticker) not in app.data:
            print(f"No historical data for {ticker}.")
            continue

        # Build the DataFrame from the historical data
        df = dataDataframe(app, tickers, ticker)
        if df.empty:
            print(f"Empty dataframe for {ticker}.")
            continue

        # Use the latest bar from the dataframe
        df = df.sort_index() #maybe not useful
        last_bar = df.iloc[-1]
        close_price = last_bar['Close']
        last_bar_date = None
        try:
            bar_datetime = pd.to_datetime(df.index[-1])
            last_bar_date = last_bar.get('Date')  # Returns None if 'Date' doesn't exist
        except Exception as e:
            print(f"Error parsing date for ticker {ticker}: {e}")
            continue

        # Extract features from the DataFrame
        row_features = extract_features(df)
        #print("row_features : ")
        #print(row_features)
        if row_features is None:
            print(f"Not enough data to extract features for {ticker}.")
            continue
        if close_price!=row_features['Close']: #todo check also:time_in_minutes if it coherent
            print("ERROR : last close price is different than the close price after feature processing")
            return
        # Prepare the input data for the model (assuming six_bins_columns is defined)
        row_features['date_after_0125'] = int(1)
        row_features['date_after_0924'] = int(1)
        input_data = {col: row_features[col] for col in six_bins_columns}

        #input_data["day_of_week"] = row_features["day_of_week"]
        try:
            #print("input data : ",input_data)
            prediction = super_model.predict(ticker, input_data)
        except ValueError as ve:
            print(f"Error predicting for {ticker}: {ve}")
            prediction = 0

        print("bar_datetime : ",bar_datetime)
        if last_bar_date is not None:
            print("last bar date : ",last_bar_date)
        print(f"Ticker: {ticker}, Close Price: {close_price}, Prediction: {prediction}")

        # If prediction equals 1 and we're not already in a position, then buy.
        if prediction == 1 :
            quantity = int(max_buget_per_trade / close_price)
            if close_price>=20:
                tp_price = round(close_price * 1.012, 2)  # Take Profit
                sl_price = round(close_price * 0.988, 2)  # Stop Loss
            else:
                tp_price = round(close_price * 1.02, 2)  # Take Profit
                sl_price = round(close_price * 0.98, 2)  # Stop Loss
            if quantity <= 0:
                print(f"Insufficient capital to buy {ticker}.")
                continue

            # Request a new order id for the parent (market BUY) order.
            app.reqIds(-1)
            time.sleep(3)
            parent_order_id = app.nextValidOrderId
            print(f"Placing BUY order for {ticker} with order id {parent_order_id} for quantity {quantity}.")
            parent, profit_taker, stop_loss = createBracketOrder(
                parent_order_id=parent_order_id,
                action="BUY",
                quantity=quantity,
                entry_price=close_price,       # or your desired entry price
                take_profit_price=tp_price,      # calculated take profit price
                stop_loss_price=sl_price         # calculated stop loss price
            )
            #profit_taker.outsideRth = True
            app.placeOrder(parent.orderId, usTechStk(ticker), parent)
            app.placeOrder(profit_taker.orderId, usTechStk(ticker), profit_taker)
            app.placeOrder(stop_loss.orderId, usTechStk(ticker), stop_loss)
            time.sleep(5)  # Wait a few seconds for the order to fill
            # Record the time of purchase in positions_dict
            positions_dict[ticker] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Recorded purchase for {ticker} at {positions_dict[ticker]}.")
            print("so we stop here for this 30 min session")
            nbr+=1
            #break
    with open(POSITIONS_DICT_FILE, "w") as f:
        json.dump(positions_dict, f)
    print("Saved positions_dict to disk.")

# -------------------------
# Main Loop: Run the strategy repeatedly
# -------------------------

def is_time_in_range():
    now = datetime.datetime.now()
    current_time = now.time()
    start_time = datetime.time(14, 39)  # 16h05
    end_time = datetime.time(21, 50)   # 21h50
    return start_time <= current_time <= end_time
 
def end_of_day():
    now = datetime.datetime.now()
    current_time = now.time()
    start_time = datetime.time(21, 58)  # 16h05
    return start_time <= current_time

app.reqAccountSummary(9001, "All", "TotalCashValue")
time.sleep(5)
if app.total_cash_value is None:
    print("Account summary not yet received.")
    exit()
else:
    start_cash = app.total_cash_value
    if PAPER_TRADING:
        goal = app.total_cash_value+max_buget_per_trade*0.02
    else:
        #goal = app.total_cash_value*1.02
        goal = 3000

print(" our goal for today is to reach : ",goal)
goal_reached = check_if_goal_reached(goal,app.total_cash_value)  
while not goal_reached:
    if not is_time_in_range():
        print("Outside of operating hours (16:05-21:50). Waiting...")
        time.sleep(300)  # Sleep for 5 minutes before checking again
        break
        continue
    print(".")
    #if end_of_day():
        #print("we will close old positions")
        #close_old_positions(app)
    now = datetime.datetime.now()
    minutes_in_block = now.minute % 30

    if minutes_in_block >= 29:
        if len(stocks_from_json)<=4:
            time.sleep(15)
        print(f"Running main()")
        main(app.total_cash_value)
        time.sleep(120)
    elif minutes_in_block >= 27:
        print("waiting for 15 sec")
        time.sleep(15)
    elif minutes_in_block >= 26:
        print("waiting for 30 sec")
        time.sleep(30)
    elif minutes_in_block >= 24:
        print("waiting for 1 minute")
        time.sleep(60)
    elif minutes_in_block >= 20:
        print("waiting for 3 minutes")
        time.sleep(180)
    else:
        waiting_time = max(10,(-(4/5)*minutes_in_block+20)*60)
        print(f"Waiting for next execution window. (minutes_in_block = {minutes_in_block}) waiting for : {waiting_time/60} min")
        time.sleep(waiting_time)  # Sleep
        
    print("..")
    goal_reached = check_if_goal_reached(goal,app.total_cash_value)

print("Goal reached, ending program.")
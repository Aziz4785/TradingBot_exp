


# -*- coding: utf-8 -*-
"""
Backtesting template for an ML-based trading strategy.
The strategy:
    - Loads a custom ML model (SuperModel).
    - Uses as features: EMA13, EMA48, and the day's Open price.
    - If super_model.predict(ticker, features)==1 on a given day:
          * Enter the trade at the Open.
          * Exit as soon as either:
              - Price reaches a take profit target of +1% (i.e. Open * 1.01), or
              - Price reaches a stop loss of -1% (i.e. Open * 0.99).
          * If neither level is reached during the day, exit at the Close.
    - The backtest is performed on data between 20/01/2025 and 31/01/2025.

@author: [Your Name]
"""

#########################
# IMPORTS & IB API SETUP
#########################
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import pandas as pd
import numpy as np
import threading
import time
import pickle
from copy import deepcopy
from utils import *
from ibapi.ticktype import TickTypeEnum 
from collections import defaultdict
from SuperModel import * 
import matplotlib.pyplot as plt
pd.options.display.max_rows = None

config_path = "models_to_use.json"
models_dir  = "allmodels"
scalers_dir = "allscalers"
clean_path = "clean.csv"
bins_file = "bins_json.json"
start_date = "2025-02-10"
end_date   = "2025-03-21"
# with open('available_features_by_stock.json') as f:
#     available_features_by_stock = json.load(f)

from datetime import datetime, timedelta
cutoff_date0125 = datetime(2025, 1, 1)
cutoff_date0924 = datetime(2024, 9, 1)
clean_df = pd.read_csv(clean_path)
clean_df = clean_df.sample(frac=1).reset_index(drop=True)
clean_df.drop_duplicates(inplace=True)
clean_df = clean_df.dropna()
clean_df["Date"] = pd.to_datetime(clean_df["Date"])
beginning_of_unseen_data = clean_df['Date'].max()

with open(config_path, 'r') as file:
    config = json.load(file)
#with open(f"C:/Users/aziz8/Documents/tradingBot/bins.pkl", "rb") as f:
    #bin_dict = pickle.load(f)
with open(bins_file, "r") as f:
    loaded_bin_dict_json = json.load(f)
    for col in loaded_bin_dict_json:
        loaded_bin_dict_json[col] = np.array(loaded_bin_dict_json[col])

super_model = SuperModel(config_path, models_dir, scalers_dir)
six_bins_columns = extracts_all_features_from_json(config)

# IB API class to retrieve historical data
class TradeApp(EWrapper, EClient): 
    def __init__(self): 
        EClient.__init__(self, self) 
        self.data = {}
        
    def tickPrice(self, reqId, tickType, price, attrib):
        # Convert the tickType integer to a string description
        tick_type_str = TickTypeEnum.to_str(tickType)
        print(f"Tick Price. ReqId: {reqId}, tickType: {tickType} ({tick_type_str}), price: {price}")
        
        # Check if the tick corresponds to delayed data
        if tick_type_str.startswith("DELAYED"):
            print("This is a delayed tick.")
        else:
            print("This is a live tick.")


    def historicalData(self, reqId, bar):
        # Build a DataFrame row-by-row
        row = {"Date": bar.date, "Open": bar.open, "High": bar.high,
               "Low": bar.low, "Close": bar.close, "Volume": bar.volume}
        if reqId not in self.data:
            self.data[reqId] = pd.DataFrame([row])
        else:
            self.data[reqId] = pd.concat([self.data[reqId], pd.DataFrame([row])])
        #print(f"reqID:{reqId}, date:{bar.date}, open:{bar.open}, high:{bar.high}, low:{bar.low}, close:{bar.close}, volume:{bar.volume}")
    def error(self, reqId, errorCode, errorString, advancedOrderReject=""):
        print(f"reqId: {reqId}, errorCode: {errorCode}, errorString: {errorString}, orderReject: {advancedOrderReject}")

def usTechStk(symbol, sec_type="STK", currency="USD", exchange="SMART"):
    contract = Contract()
    contract.symbol = symbol
    contract.secType = sec_type
    contract.currency = currency
    contract.exchange = exchange
    return contract 

def histData(req_num, contract, duration, candle_size):
    """Extracts historical data via IB API"""
    app.reqHistoricalData(
        reqId=req_num, 
        contract=contract,
        endDateTime='',
        durationStr=duration,
        barSizeSetting=candle_size,
        whatToShow='TRADES', #ADJUSTED_LAST
        useRTH=0,
        formatDate=1,
        keepUpToDate=0,
        chartOptions=[]
    )

def websocket_con():
    app.run()

#########################
# START IB API CONNECTION
#########################
app = TradeApp()
app.connect(host='127.0.0.1', port=7497, clientId=23)  # adjust port/clientId as needed
con_thread = threading.Thread(target=websocket_con, daemon=True)
con_thread.start()
time.sleep(10)  # wait to ensure connection is established
app.reqMarketDataType(1) # 1 = Live, 3 = Delayed, etc.
time.sleep(10) 
# List of tickers to backtest
#tickers = ["META", "MSFT", "INTC", "MSFT", "AAPL"]
tickers = stock_symbols = list(config.keys())
print(tickers)

for ticker in tickers:
    print(ticker)
    try:
        histData(tickers.index(ticker), usTechStk(ticker), '100 D', '30 mins')
        time.sleep(30)
    except Exception as e:
        print(e)
        print("Unable to extract data for", ticker)

def dataDataframe(symbols, TradeApp_obj):
    print("-----dataDataframe()-----")
    "Returns historical data for each symbol in a dictionary of DataFrames."
    df_data = {}
    for symbol in symbols:
        df = pd.DataFrame(TradeApp_obj.data[symbols.index(symbol)])
        #df.set_index("Date", inplace=True)
        df_data[symbol] = df
    return df_data

# Get the historical data extracted by IB API
historicalData = dataDataframe(tickers, app)

# We assume that the Date index is convertible to datetime
ohlc_dict = {}
for ticker in tickers:
    print("filling data for ticker: ",ticker)
    df = historicalData[ticker].copy()
    
    df["Date"] = df["Date"].str.replace(" US/Eastern", "", regex=False)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d %H:%M:%S")
    df['day_of_week'] = df['Date'].dt.dayofweek 
    #df['time_in_minutes'] = ((df['Date'].dt.hour * 60 + df['Date'].dt.minute + 15) // 30) * 30
    df['time_in_minutes'] = df['Date'].dt.hour * 60 + df['Date'].dt.minute
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)  # ensure it's sorted by datetime
    
    df = add_market_time_column(df)
    #df = df.loc[start_date:end_date] #for debugging
    #df.to_csv("debugging_from_backtest3.csv", index=True)
    df["day"] = df.index.date
    mask = df["market_time"] == "RTH"
    df_rth = df.loc[mask].copy()
    df_rth = calculate_daily_ohlc(df_rth)
    df_rth=calculate_historical_highs(df_rth)
    df_rth = add_historical_values(df_rth)
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
    #print("df columns : ",df.columns)
    df = add_AH_columns(df)
    df = calculate_emas(df)
    df = calculate_stds(df)
    df = add_shifted_fractional_diff_from_DEPRADO(df)
    df = add_volatilities(df)
    df["mom_5"] = add_momentum(df, ticker)
    #print("df columns before calling additional_ratios: ",df.columns)
    df = additional_ratios(df)
    df = additional_Close_ratios(df)
    #print("df columns before calling additional_PM_ratios: ",df.columns)
    df = additional_PM_ratios(df)
    df = calculate_slopes(df)
    #TODO ADD a boolean to see if there is a local minima on the  dayHigh_3

    df = calculate_ema_ratios(df)
    df = calculate_highs_and_lows(df)
    df = df.loc[start_date:end_date]
    df.drop("day", axis=1, inplace=True)
    df = df[df['market_time'] == 'RTH']

    #df.to_csv("debugging_from_backtest2.csv", index=True)
    #df = df.dropna()
    #if ticker =='TSN':
        #df.to_csv(f"{ticker}_debug.csv", index=True)
    ohlc_dict[ticker] = df

def determine_trade_parameters(entry):
    """Determine stop loss, take profit, and absolute return based on entry price."""
    if entry >= 20:
        return entry * 0.988, entry * 1.012
    return entry * 0.98, entry * 1.02
###########################################
# LOAD THE CUSTOM ML MODEL (SuperModel)
###########################################
# (Make sure that SuperModel is defined/importable from your module)

tickers_ret = defaultdict(list)
trade_count = defaultdict(int)
trade_data = defaultdict(dict)
debugging_results = []
for ticker in tickers:
    df = ohlc_dict[ticker]
    print("predicting...", ticker)
    trading_days = df.index.normalize().unique()
    df['prediction'] = 0
    df['actual_result'] = -1 #-1 means we dont know the result
    df['target_hit_on'] = None
    # Loop through each row/bar in the DataFrame
    for idx, row in df.iterrows():
        
        row_features = row.copy()
        row_features = assign_bin_categories(row_features, loaded_bin_dict_json)
        #if ticker =='TSN':
            #print(f"at data : {idx} , actual_result = {row['actual_result']}")
        # Build the input dictionary with the discretized feature names.
        input_data = {}
        row_features['date_after_0125'] = int(pd.to_datetime(idx) > cutoff_date0125)
        row_features['date_after_0924'] = int(pd.to_datetime(idx) > cutoff_date0924)

        input_data = {col: row_features[col] for col in six_bins_columns}
        input_data["day_of_week"] = row_features["day_of_week"]
        
        # Get the prediction from the ML model.
        
        try:
            prediction = super_model.predict(ticker, input_data)
            df.loc[idx, 'prediction'] = prediction
        except ValueError as ve:
            if ticker =='TSN':
                print(input_data)
            print(f"Error for {ticker} on {idx}: {ve}")
            prediction = 0
        #if ticker =='TSN':
            #print(f" prediction for date  {idx} = {prediction}")

        if prediction != 1:
            tickers_ret[ticker].append(0)
            continue
        # Enter trade at current bar’s close.
        entry = row["Close"]
        trade_count[ticker] += 1
        # Create a new entry in trade_data, storing the entry price
        # trade_data[ticker][trade_count[ticker]] will hold a list of [entry_price, exit_price]
        trade_data[ticker][trade_count[ticker]] = [entry]

        stop_loss, take_profit = determine_trade_parameters(entry)     
        
        # Determine the current trading day and find the next trading day from our list.
        trade_date = idx.normalize()  # normalize the timestamp to remove the time part
        try:
            current_day_index = trading_days.get_loc(trade_date)
            current_trading_date = trade_date
        except KeyError:
            print(f"Warning: {trade_date} not found in trading_days for ticker {ticker}.")
            current_day_index = None

        next_trading_date = (
            trading_days[current_day_index + 1] if current_day_index is not None and current_day_index < len(trading_days) - 1 else None
        )

        exit_price, target_hit = None, None # Will hold the last bar's close of the next trading day
        if next_trading_date is None:
            trade_data[ticker][trade_count[ticker]].append(entry)
            tickers_ret[ticker].append(0)
            if ticker =='JNJ':
                print(f" NEXT TRADING DATA IS nONE")
            continue

        # Get the integer location of the current row.
        current_loc = df.index.get_loc(idx)
        # Loop over subsequent bars until we pass the next trading day.
        for idx2, row2 in df.iloc[current_loc+1:].iterrows():
            row_date = idx2.normalize()
            if row_date > next_trading_date:
                break
            if target_hit is None:
                if row2["High"] >= take_profit:
                    df.at[idx, 'actual_result'] = 1
                    df.at[idx, 'target_hit_on'] = idx2
                    target_hit = "tp"
                elif row2["Low"] <= stop_loss:
                    df.at[idx, 'actual_result'] = 0
                    df.at[idx, 'target_hit_on'] = idx2
                    target_hit = "sl"
            if target_hit is None and row_date == next_trading_date:
                df.at[idx, 'actual_result'] = 0
                exit_price = row2["Close"]

        # Compute the trade return based on what happened during the next trading day.
        if exit_price is None and target_hit is None: 
            ret = 0
            trade_data[ticker][trade_count[ticker]].append(entry)
        else: #exit price is not None or target is not None
            if target_hit == "tp":
                ret = (take_profit / entry) - 1
                trade_data[ticker][trade_count[ticker]].append(take_profit)
                df.loc[idx, 'actual_result'] = 1
            elif target_hit == "sl":
                ret = (stop_loss / entry) - 1
                trade_data[ticker][trade_count[ticker]].append(stop_loss)
                df.loc[idx, 'actual_result'] = 0
            else:
                if exit_price is None:
                    print(" WE SHOULD NOT HAVE THIS CASE")
                trade_data[ticker][trade_count[ticker]].append(exit_price)
                df.loc[idx, 'actual_result'] = 0
                ret = (exit_price / entry) - 1

        tickers_ret[ticker].append(ret)
    if ticker in ['AAPL','TSN','KO']:
        df[['Open','High','Low','Close','prediction','actual_result','target_hit_on']].to_csv(f"debugging_{ticker}_from_backtest.csv", index=True)
#That will show you which trades are missing an exit or (less commonly) have more than 2 entries.
for t_id, trade_list in trade_data[ticker].items():
    if len(trade_list) != 2:
        print("WARNING !!!!!!!!!!!!!!!!!!")
        print(f"Ticker: {ticker}, Trade ID: {t_id}, trade_list: {trade_list}")


#we will compute the precison on rows that occur after last testing date:
from datetime import datetime,timedelta
for ticker in tickers:
    df_ticker = ohlc_dict[ticker].copy()
    df_ticker = df_ticker.reset_index()
    df_ticker["Date"] = pd.to_datetime(df_ticker["Date"])
    start_date = beginning_of_unseen_data
    end_date = min(datetime.today(), start_date + timedelta(days=6))
    filtered_df = df_ticker[(df_ticker["Date"] > start_date) & (df_ticker["Date"] <= end_date) & (df_ticker["actual_result"] != -1)]
    # Compute precision
    true_positives = ((filtered_df["prediction"] == 1) & (filtered_df["actual_result"] == 1)).sum()
    false_positives = ((filtered_df["prediction"] == 1) & (filtered_df["actual_result"] == 0)).sum()
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else None
    
    if precision is not None and precision >= 0.65:
        good_ticker = 1
    elif precision is None:
        good_ticker = 0
    else:
        good_ticker = -1
    print(f"PRECISION FOR {ticker}  = {precision}    -> {good_ticker}")
    if ticker in ['AAPL','TSN']:
        print("because ]start date = ",start_date)
        print(f"and enddate = {end_date}]")
        print(f"and number of true positives = {true_positives}")
        print(f"and number of false positives (0 predicted as 1)= {false_positives}")



# 1) Turn each ticker's return list into a pd.Series with a Date index
returns_dict = {}
for ticker in tickers:
    df = ohlc_dict[ticker]
    # Create a Series of returns indexed by the same dates as df
    returns_dict[ticker] = pd.Series(tickers_ret[ticker], index=df.index)
    #print(f"here are all the return for ticker: {ticker}")
    #print(returns_dict[ticker] )

# 2) Combine into one "wide" DataFrame
returns_df = pd.DataFrame(returns_dict)

returns_df["ret"] = returns_df.mean(axis=1)
print("returns_df : ")
print(returns_df.head(30))
###########################################
# (OPTIONAL) CALCULATE STRATEGY PERFORMANCE KPIs
###########################################
def CAGR(DF):
    """Compute the Cumulative Annual Growth Rate based on period returns."""
    df = DF.copy()
    df["cum_return"] = (1 + df["ret"]).cumprod()
    # 'n' is the number of years; adjust the denominator if your data frequency is not daily.
    n = len(df) / 252  
    return df["cum_return"].iloc[-1] ** (1/n) - 1

def volatility(DF):
    """Annualized volatility."""
    df = DF.copy()
    return df["ret"].std() * np.sqrt(252)

def sharpe(DF, rf=0.025):
    """Sharpe Ratio; rf is the risk-free rate."""
    volat = volatility(DF)
    if volat==0:
        return 0
    return (CAGR(DF) - rf) / volat

def max_dd(DF):
    """Maximum Drawdown."""
    df = DF.copy()
    df["cum_return"] = (1 + df["ret"]).cumprod()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    df["drawdown_pct"] = df["drawdown"] / df["cum_roll_max"]
    return df["drawdown_pct"].max()

overall_cagr = CAGR(returns_df)
overall_sharpe = sharpe(returns_df, 0.025)
overall_max_dd = max_dd(returns_df)

print("Overall CAGR:", overall_cagr)
print("Overall Sharpe Ratio:", overall_sharpe)
print("Overall Max Drawdown:", overall_max_dd)

###########################################
# (OPTIONAL) VISUALIZE THE CUMULATIVE RETURNS
###########################################

#beginning_of_unseen_data = pd.Timestamp("2025-02-12 09:30:00")
PLOT_ONLY_MEAN_DATA = False
if PLOT_ONLY_MEAN_DATA:
    
    cumprod_series = (1 + returns_df["ret"]).cumprod()
    cumprod_series.plot(title="ML Strategy Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    if beginning_of_unseen_data in cumprod_series.index:
        plt.axvline(x=beginning_of_unseen_data, color='red', linestyle='--')
    else:
        print("date not in x axis")
    plt.show()
else:
    # stock_columns = [col for col in returns_df.columns if col != 'ret']
    # cumulative_returns = (1 + returns_df[stock_columns]).cumprod()
    # colors = get_distinct_colors(len(stock_columns))
    # # Create the plot with distinct colors
    # fig, ax = plt.subplots(figsize=(12, 6))
    # for idx, column in enumerate(stock_columns):
    #     cumulative_returns[column].plot(ax=ax, color=colors[idx])

    # # Add the overall strategy returns
    # strategy_returns = (1 + returns_df["ret"]).cumprod()
    # strategy_returns.plot(ax=ax, linewidth=3, color='black', label='Mean')

    # # Customize the plot
    # plt.title("Individual Stocks Cumulative Returns")
    # plt.xlabel("Date")
    # plt.ylabel("Cumulative Return")

    # # Add the vertical line for unseen data if applicable
    # if beginning_of_unseen_data in cumulative_returns.index:
    #     plt.axvline(x=beginning_of_unseen_data, color='red', linestyle='--')

    # # Adjust legend position and layout
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.tight_layout()  # Adjust layout to prevent legend cutoff
    # plt.show()
    # Assuming returns_df is your DataFrame
    stock_columns = [col for col in returns_df.columns if col != 'ret']

    # 1. Store the original index in a separate variable
    original_dates = returns_df.index

    # Reset the index to numbers (0, 1, 2, ...)
    df_numeric_index = returns_df.reset_index(drop=True)

    # Calculate cumulative returns
    cumulative_returns = (1 + df_numeric_index[stock_columns]).cumprod()
    colors = get_distinct_colors(len(stock_columns))

    # Create the plot with distinct colors
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, column in enumerate(stock_columns):
        cumulative_returns[column].plot(ax=ax, color=colors[idx])

    # Add the overall strategy returns
    strategy_returns = (1 + df_numeric_index["ret"]).cumprod()
    strategy_returns.plot(ax=ax, linewidth=3, color='black', label='Mean')

    # 5. Manually set x-ticks and their labels
    #    For example, let's put a label every 5 observations (you can adjust as you like)
    step = 12 
    x_ticks = range(0, len(df_numeric_index), step)
    ax.set_xticks(x_ticks)
    # Format the corresponding date labels, e.g. 'YYYY-MM-DD' (or any other format)
    x_labels = [original_dates[i].strftime('%Y-%m-%d') for i in x_ticks]
    ax.set_xticklabels(x_labels, rotation=75)  # rotate if needed


    # Customize the plot
    plt.title("Individual Stocks Cumulative Returns")
    plt.xlabel("Time Step")  # Change label to reflect numeric index
    plt.ylabel("Cumulative Return")

    # If you still want a vertical line for the 'unseen data' point:
    # 1. Convert your original date index to numeric by locating the position:
    beginning_of_unseen_data_numeric = returns_df.index.get_loc(beginning_of_unseen_data)
    plt.axvline(x=beginning_of_unseen_data_numeric, color='red', linestyle='--')

    # Adjust legend position and layout
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()  # Adjust layout to prevent legend cutoff
    plt.show()

# Trade-Level KPIs (assuming trade_data is correctly populated)
trade_df = {}
for ticker in tickers:
    if trade_data[ticker]:
        trade_df[ticker] = pd.DataFrame(trade_data[ticker]).T
        trade_df[ticker].columns = ["trade_entry_pr", "trade_exit_pr"]
        trade_df[ticker]["return"] = trade_df[ticker]["trade_exit_pr"] / trade_df[ticker]["trade_entry_pr"]
    else:
        trade_df[ticker] = pd.DataFrame(columns=["trade_entry_pr", "trade_exit_pr", "return"])  # Empty DataFrame


#print("Sample of trade_data for one ticker:")
#print(trade_data["WU"])  # or whichever ticker

#print("Head of trade_df for one ticker:")
#print(trade_df["WU"].head())

win_rate = {}
mean_ret_pt = {}
mean_ret_pwt = {}
mean_ret_plt = {}
max_cons_loss = {}
for ticker in tickers:
    print("Calculating trade-level KPIs for", ticker)
    win_rate[ticker] = winRate(trade_df[ticker])      
    mean_ret_pt[ticker] = meanretpertrade(trade_df[ticker])
    mean_ret_pwt[ticker] = meanretwintrade(trade_df[ticker])
    mean_ret_plt[ticker] = meanretlostrade(trade_df[ticker])
    max_cons_loss[ticker] = maxconsectvloss(trade_df[ticker])

KPI_df = pd.DataFrame([win_rate, mean_ret_pt, mean_ret_pwt, mean_ret_plt, max_cons_loss],
                      index=["Win Rate", "Mean Return Per Trade", "MR Per Winning Trade", "MR Per Losing Trade", "Max Cons Loss"])      
print(KPI_df.T)
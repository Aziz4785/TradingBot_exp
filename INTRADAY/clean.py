"""
1) add to_buy column
to_buy intraday : if it does 0.4% before 16h
to_buy 1day : if it does 1% before the end of the next day
to_buy 2day : if it does 1.5% before the end of the next next day
2) clean the data
2) print statistics on data
2)add some new columns
2) discretization of data
"""
import pandas as pd
import sys
import pickle
import json
import numpy as np
from utils import *
from datetime import datetime, timedelta

cleanup_files(['models_to_use.json', 'bins_json.json','clean.csv'])
#load available_features_by_stock.json
with open('available_features_by_stock.json') as f:
    available_features_by_stock = json.load(f)
df = pd.read_csv(f"raw.csv")
print("length of df before dropping duplicates : ",len(df))
df = df.drop_duplicates()
#delete rows where column Close is None
df = df.dropna(subset=['Close'])


#drop na on some columns for each stock :
filtered_df = pd.DataFrame()
for stock in df['Stock'].unique():  
    # Get the stock-specific rows
    stock_data = df[df['Stock'] == stock]
    # Get the columns to check for None values for this stock
    if stock in available_features_by_stock:
        columns_to_check = available_features_by_stock[stock]
        stock_filtered = stock_data.dropna(subset=columns_to_check)
        filtered_df = pd.concat([filtered_df, stock_filtered])
    else:
        # If stock not in available_features_by_stock, drop na on all columns
        stock_data = stock_data.dropna()
        filtered_df = pd.concat([filtered_df, stock_data])
df = filtered_df.reset_index(drop=True)

ADD_INTRADAY_COLUMN = False
#cutoff_date_100 = datetime.now() - timedelta(days=100)
cutoff_date0724 = datetime(2024, 7, 1)
cutoff_date0924 = datetime(2024, 9, 1)
cutoff_date1124 = datetime(2024, 11, 1)
cutoff_date0125 = datetime(2025, 1, 1)


df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Stock','Date']).reset_index(drop=True)


df['take_profit_intrday'] = df['Close'] * 1.012
df['stop_loss_intrday']   = df['Close'] * 0.99
df['TradeDate'] = df['Date'].dt.date  # separate "day" from the timestamp

unique_dates = df.groupby('Stock')['TradeDate'].apply(lambda x: sorted(list(set(x))))
next_1d_map, next_2d_map = {}, {}

for stock, dates in unique_dates.items():
    next_1d = {dates[i]: dates[i+1] if i+1 < len(dates) else None for i in range(len(dates))}
    next_2d = {dates[i]: dates[i+2] if i+2 < len(dates) else None for i in range(len(dates))}
    next_1d_map[stock] = next_1d
    next_2d_map[stock] = next_2d

df['next_1d'] = df.apply(lambda row: next_1d_map.get(row['Stock'], {}).get(row['TradeDate'], None), axis=1)
#df['next_2d'] = df.apply(lambda row: next_2d_map.get(row['Stock'], {}).get(row['TradeDate'], None), axis=1)
# Create grouped data for quick access
grouped = df.groupby(['Stock', 'TradeDate'])
group_dict = {key: group for key, group in grouped}

def calculate_to_buy(row, n_days, tp_multiplier, sl_multiplier, tp_for_small_price, sl_for_small_price):
    stock = row['Stock']
    current_date = row['TradeDate']
    current_dt = row['Date']
    current_close = row['Close']
    if n_days == 1:
        next_dates = [row['next_1d']]
    elif n_days == 2:
        next_dates = [row['next_1d'], row['next_2d']]
    else:
        return 0
    
    # Check if there are no next dates
    has_next_dates = any(nd is not None for nd in next_dates)

    # Collect relevant rows: same day after current time and next_dates
    relevant = []
    
    # Same day rows after current datetime
    same_day = group_dict.get((stock, current_date), pd.DataFrame()) #dictionary group_dict has keys as tuples (stock, trade date) and values are DataFrames containing all rows for that stock on that day
    if not same_day.empty:
        same_day_after = same_day[same_day['Date'] > current_dt]
        relevant.append(same_day_after)
    
    # Add rows from next dates
    for nd in next_dates:
        if nd is not None:
            next_day = group_dict.get((stock, nd), pd.DataFrame())
            relevant.append(next_day)
    
    if not relevant:
        return 0
    combined = pd.concat(relevant).sort_values('Date')

    if current_close<20:
        tp = current_close * tp_for_small_price
        stop_loss = current_close * sl_for_small_price
    else:
        tp = current_close * tp_multiplier
        stop_loss = current_close * sl_multiplier

    to_buy_val = None if not has_next_dates else 0
    for _, r in combined.iterrows():
        if r['High'] >= tp:
            to_buy_val = 1
            break
        if r['Low'] <= stop_loss:
            to_buy_val = 0
            break
    return to_buy_val



#KEEP ONLY stocks that have at least one row with this maximum date:
print("length of df before removing stocks with few dates : ",len(df))
df['Date_only'] = df['Date'].dt.date
max_date = df['Date_only'].max()
print("Maximum date among all rows:", max_date)
# Define tolerance as 1 day and select stocks with any date within that range
tolerance = timedelta(days=1)
stocks_with_max_date = df.loc[(max_date - df['Date_only']) <= tolerance, 'Stock'].unique()
print("Stocks with max date (within tolerance):", stocks_with_max_date)
df = df[df['Stock'].isin(stocks_with_max_date)].copy()


# Calculate 'to_buy_1d' columns
print("adding to_buy columns..")
df['to_buy_1d'] = df.apply(lambda row: calculate_to_buy(row, 1, 1.012, 0.988, 1.02, 0.98), axis=1)
df = df.drop(columns=["next_1d"])
#df['to_buy_2d'] = df.apply(lambda row: calculate_to_buy(row, 2, 1.015, 0.985, 1.03, 0.97), axis=1)
df = df.sort_values(['Stock','TradeDate','Date'])
if ADD_INTRADAY_COLUMN:
    df['to_buy_intraday'] = 0
    df = df.groupby(['Stock','TradeDate'], group_keys=False).apply(process_intraday_signals)


# 1. Count how many rows each stock has
counts = df['Stock'].value_counts()
# 2. Calculate mean and std of these counts
mean_count = counts.mean()
std_count = counts.std()
print("mean count of rows per stock:", mean_count)
# 3. Define a cutoff (e.g., 2 standard deviations from the mean)
cutoff_lower = mean_count - 300
cutoff_upper = mean_count + 400
# 4. Determine which stocks are within these cutoffs
valid_stocks = counts[
    (counts >= cutoff_lower) & (counts <= cutoff_upper)
].index
# 5. Filter your main DataFrame to only keep these “valid” stocks
print()
print("length of df before removing stocks with few dates : ",len(df))
removed_stocks = set(df['Stock'].unique()) - set(valid_stocks)
df = df[df['Stock'].isin(valid_stocks)].copy()
print("length of df after removing stocks with few dates : ",len(df))
print("Stocks removed due to having few dates:", removed_stocks)
df.drop(columns=['Date_only',"take_profit_intrday", "stop_loss_intrday", "TradeDate", "next_1d","next_2d", "market_time"], inplace=True, errors='ignore')
print()

# 3) print statistics on data
stats_str = df.describe(include='all').to_string()
with open('clean_dataframe_statistics.txt', 'w') as file:
    file.write("DataFrame Statistics:\n")
    file.write(stats_str)
if ADD_INTRADAY_COLUMN:
    percentages = df[['to_buy_intraday', 'to_buy_1d']].mean() * 100
else:
    if 'to_buy_2d' in df.columns:
        percentages = df[['to_buy_1d', 'to_buy_2d']].mean() * 100
    else:
        percentages = df[['to_buy_1d']].mean() * 100
print("Percentages of 1s in each column:")
print(percentages.round(2).to_string()) 


#4)adding new columns
def calculate_metrics(df):
    # Create a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    max_shift = 14
    #create columns for the previous close prices (prev_close, prev2_close, ..., prev14_close)
    for shift in range(1, max_shift + 1):
        col_name = get_col_name(shift)
        result_df[col_name] = result_df.groupby('Stock')['Close'].shift(shift)

    coeffs = [1, -0.8, -0.08, -0.032, -0.0176, -0.0113, -0.0079, -0.0059,
          -0.0045, -0.0036, -0.003, -0.0025, -0.0021]  
    
    # Calculate FD_1, FD_2, FD_3 by applying an offset to the coefficients
    for fd, offset in zip(['FD_1', 'FD_2', 'FD_3'], [0, 1, 2]):
        result_df[fd] = sum(coeff * result_df[get_col_name(i + offset)]
                            for i, coeff in enumerate(coeffs))
        
    # For rows where any of the required previous values is missing, set the new columns to None
    fd1_cols = [get_col_name(i) for i in range(1, 13)]
    result_df.loc[result_df[fd1_cols].isnull().any(axis=1), 'FD_1'] = None

    # For FD_2, if it uses shifts 1 to 13:
    fd2_cols = [get_col_name(i) for i in range(1, 14)]
    result_df.loc[result_df[fd2_cols].isnull().any(axis=1), 'FD_2'] = None

    # For FD_3, if it uses shifts 2 to 14:
    fd3_cols = [get_col_name(i) for i in range(2, 15)]
    result_df.loc[result_df[fd3_cols].isnull().any(axis=1), 'FD_3'] = None

    # Optionally, drop the temporary shifted columns
    columns_to_drop = [get_col_name(shift) for shift in range(1, 15)]
    result_df.drop(columns=columns_to_drop, inplace=True)
    
    result_df = add_volatilities(result_df)
    result_df = add_momentum(result_df)
    
    result_df = additional_ratios(result_df)

    result_df = additional_Close_ratios(result_df)

    result_df['day_of_week'] = result_df['Date'].dt.dayofweek 
    result_df['time_in_minutes'] = result_df['Date'].dt.hour * 60 + result_df['Date'].dt.minute
    result_df['date_after_0924'] = (result_df['Date'] > cutoff_date0924).astype(int)
    result_df['date_after_1124'] = (result_df['Date'] > cutoff_date1124).astype(int)
    result_df['date_after_0724'] = (result_df['Date'] > cutoff_date0724).astype(int)
    result_df['date_after_0125'] = (result_df['Date'] > cutoff_date0125).astype(int)
    
    #result_df['time_in_minutes'] = ((result_df['Date'].dt.hour * 60 + result_df['Date'].dt.minute + 15) // 30) * 30
    
    result_df = additional_PM_ratios(result_df)
    
    result_df = calculate_slopes(result_df)
    result_df = calculate_volume_slopes(result_df)
    
    #TODO ADD a boolean to see if there is a local minima on the  dayHigh_3
    
    result_df = calculate_ema_ratios(result_df)
    #result_df = calculate_sma_ratios(result_df)
    result_df = calculate_highs_and_lows(result_df)
    return result_df 

def process_stock_group(group):
    stock = group.name  # Each group's name is the stock identifier
    group = add_shifted_fractional_diff_from_DEPRADO(group)

    # Add volatility columns to the group
    group = add_volatilities(group)
    
    # Compute momentum if available
    mom_5 = add_momentum(group, stock, available_features_by_stock)
    if mom_5 is not None:
        group["mom_5"] = mom_5

    group = additional_ratios(group,available_features_by_stock[stock])
    group = additional_Close_ratios(group,available_features_by_stock[stock])
    group['day_of_week'] = group['Date'].dt.dayofweek 
    group['time_in_minutes'] = group['Date'].dt.hour * 60 + group['Date'].dt.minute
    group['date_after_0924'] = (group['Date'] > cutoff_date0924).astype(int)
    group['date_after_1124'] = (group['Date'] > cutoff_date1124).astype(int)
    group['date_after_0724'] = (group['Date'] > cutoff_date0724).astype(int)
    group['date_after_0125'] = (group['Date'] > cutoff_date0125).astype(int)

    group = additional_PM_ratios(group , available_features_by_stock[stock])
    group = calculate_slopes(group,available_features_by_stock[stock])
    group = calculate_volume_slopes(group,available_features_by_stock[stock])
    group = calculate_ema_ratios(group,available_features_by_stock[stock])
    group = calculate_highs_and_lows(group)
    return group

df = df.sort_values(['Stock','Date'])
# Process each group and combine the results into a new DataFrame.
df = df.groupby("Stock", group_keys=False).apply(process_stock_group)

#df = calculate_metrics(df)
print("len of df before removing extreme rows: ",len(df))
df = remove_extreme_rows(df, exclude_columns=['Stock','Date'])
print("len of df after removing extreme rows: ",len(df))

#2) clean the data
#print("cleaning the data..")
# print("len of df before dropping na: ",len(df))
# df = df.dropna()
df = df.dropna(subset=['to_buy_1d']) #because these rows are the last date for each stock and we don't know if we reached the target or not
# print("len of df after dropping na: ",len(df))
pd.options.display.max_columns = 60

# 3) print statistics on data
print("number of unique values of time_in_minutes")
print(df['time_in_minutes'].nunique())
"""
print("DataFrame Statistics:")
print(
    df[
        [
            'PM_min_to_prevDayOpen_ratio',
            'PM_max_to_Close_ratio',
            'PM_min_to_Close_ratio',
            'high_slope_rel',
            'high_quad_p_rel',
            'high_quad_q_rel',
            'Close_to_EMA_48'
        ]
    ].describe()
)
"""
def bin_column(dataframe, col_name, num_bins, drop_original=True, display=False, use_quantile=True):
    """
    Bins a column into 'num_bins' bins (equal-frequency) and replaces it with a 
    new column "<col_name>_class". Returns the bin edges for reference.
    """
    # We'll use qcut instead of cut for quantile-based binning (equal frequency)
    col_class_name = f"{col_name}_class"
    if use_quantile:
        dataframe[col_class_name], bin_edges = pd.qcut(
            dataframe[col_name],
            q=num_bins,
            labels=range(1, num_bins + 1),
            duplicates='drop',
            retbins=True
        )
    else:
        dataframe[col_class_name], bin_edges = pd.cut(
                dataframe[col_name],
                bins=num_bins,                # Number of segments
                labels=range(1, num_bins + 1),
                retbins=True
            )
    if display:
        print(f"\nRanges for {col_name} bins:")
        for i in range(len(bin_edges)-1):
            print(f"Bin {i+1}: {bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}")
    # Optionally drop the original column
    if drop_original:
        dataframe.drop(columns=[col_name], inplace=True)

    return bin_edges


df.drop(columns=["Open", "High","Low","High_10","Low_10","High_20","Low_20","dayOpen", "dayClose", "prevDayOpen","prevDayClose" ,"prev2DayOpen","Close_1_day_ago","PM_max_time_in_sec"
                 ,"PM_min_time_in_sec","AH_max","Close_2_days_ago","prev2DayClose","PM_max_1dayago","AH_max_1dayago","Open_1_day_ago","dayHigh",
                 "dayHigh_1","dayHigh_2","dayHigh_3","prevDayLow","dayLow","EMA_3","EMA_13","EMA_48","PM_max","PM_min","prev_close","prev2_close","prev3_close","prev4_close","prev5_close","prev6_close","prev7_close","prev8_close","prev9_close","prev10_close","prev11_close"], inplace=True)


bin_dict = {}
six_bins_columns = ['return_1d','return_2d','open_to_prev_close','PM_time_diff','PM_min_to_open_ratio','PM_range_to_open_ratio','PM_range_to_close_ratio',
                    'Close_to_open_ratio','dayOpen_to_prevDayClose','hist_close_ratio','dayOpen_to_prevDayOpen_ratio','dayOpen_to_prev2DayOpen_ratio',
                    'Open_1_day_ago_to_Close_1_day_ago_ratio','return_1d_to_return_2d_ratio',
                    'prev2DayClose_to_prevDayClose_ratio','PM_max_vs_PM_max_1dayago','AH_max_1dayago_to_Close','AH_max_1dayago_vs_PM_max',
                    'AH_max_1dayago_vs_prevDayClose','Close_to_prevDayClose','Close_to_Close_1_day_ago','Close_to_prevDayOpen','PM_max_to_PM_min_ratio',
                    'PM_max_to_dayOpen_ratio','PM_max_to_prevDayClose_ratio',
                    'PM_min_to_prevDayClose_ratio','PM_max_to_prevDayOpen_ratio',
                    'PM_min_to_prevDayOpen_ratio','PM_max_to_Close_ratio','PM_min_to_Close_ratio',
                    'high_slope_rel','high_quad_p_rel','high_quad_q_rel','Close_to_prevDayLow','Close_to_prevDayHigh',
                    'ema_ratio1','ema_ratio2','Close_to_EMA_48','close_to_High10',
                    'close_to_Low10','close_to_High20','close_to_Low20','vol_50','PM_max_to_min_ratio',
                    'vol_20','vol_5','return',"mom_5",'Close','PM_volume_sum','PM_volume_max',
                    'Volume','slope_a_vol_rel','coef_p_vol_rel','coef_q_vol_rel','STD_10','STD_30']

for col in six_bins_columns:
    if col in df.columns:
        try:
            if col =='PM_time_diff':# or col=='Close':
                bin_dict[col] = bin_column(df, col, num_bins=4, drop_original=True, display=False, use_quantile=False)
            elif col =='Close' or col=='Volume':
                bin_dict[col] = bin_column(df, col, num_bins=11, drop_original=False, display=False, use_quantile=False)
            elif col=='STD_10' or col=='STD_30':
                bin_dict[col] = bin_column(df, col, num_bins=8)
            else:
                bin_dict[col] = bin_column(df, col, num_bins=6)
        except ValueError as e:
            print(f"Error occurred with column: {col}")
            print(f"Error message: {str(e)}")
            print(f"Sample of column data:\n{df[col].head()}")
            print(f"Column unique values: {df[col].nunique()}")
            print(f"Column value counts:\n{df[col].value_counts()}")
            sys.exit(1) 
df.to_csv('clean.csv', encoding='utf-8', index=False)
#with open(f"bins.pkl", "wb") as f:
    #pickle.dump(bin_dict, f)

# Convert all bin edges to lists so that they are JSON serializable:
serializable_bin_dict = {}
for col, bin_edges in bin_dict.items():
    # If bin_edges is a NumPy array, convert it to a list
    if isinstance(bin_edges, np.ndarray):
        serializable_bin_dict[col] = bin_edges.tolist()
    else:
        serializable_bin_dict[col] = bin_edges

# Save the dictionary to a JSON file:
with open("bins_json.json", "w") as f:
    json.dump(serializable_bin_dict, f, indent=4)
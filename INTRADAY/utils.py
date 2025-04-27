import pandas as pd
import numpy as np
import json
import os
import random
from sklearn.metrics import precision_score, recall_score, fbeta_score, matthews_corrcoef, confusion_matrix, accuracy_score, precision_recall_curve

def load_bins(bins_file_json):
    with open(bins_file_json, "r") as f:
        loaded_bin_dict_json = json.load(f)
        for col in loaded_bin_dict_json:
            loaded_bin_dict_json[col] = np.array(loaded_bin_dict_json[col])
    return loaded_bin_dict_json

def add_market_time_column(df):
    """
    Add a 'market_time' column to the dataframe based on the time of day:
    - PM (Pre-Market): Before 09:30:00
    - RTH (Regular Trading Hours): Between 09:30:00 and 15:30:00
    - AH (After Hours): After 15:30:00
    
    Parameters:
    df (pandas.DataFrame): DataFrame with datetime index
    
    Returns:
    pandas.DataFrame: DataFrame with new 'market_time' column
    """
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Extract time from datetime index
    df['time'] = df.index.time
    
    # Create market_time column based on conditions
    df['market_time'] = 'RTH'  # Default value
    df.loc[df['time'] < pd.to_datetime('09:30:00').time(), 'market_time'] = 'PM'
    df.loc[df['time'] > pd.to_datetime('15:30:00').time(), 'market_time'] = 'AH'
    
    # Drop the temporary time column
    df.drop('time', axis=1, inplace=True)
    
    return df

def calculate_score(precision, specificity, recall):
    weights = {
        'precision': 1.0,
        'specificity': 1.2,
        'recall': 0.4
    }
    
    score = (precision * weights['precision'] + 
             specificity * weights['specificity'] + 
             recall * weights['recall'])
    
    return score

def assign_class(new_row, column_name, bin_dict):
    """
    Assigns a class label to new_row[column_name + '_class'] 
    based on binning rules in bin_dict[column_name].
    """
    val = new_row[column_name]
    if pd.isna(val):
        new_row[f"{column_name}_class"] = np.nan
    else:
        # pd.cut returns a Categorical object; we take the first (and only) element.
        cat = pd.cut(
            [val],
            bins=bin_dict[column_name],
            labels=range(1, len(bin_dict[column_name])),
            include_lowest=True
        )
        new_row[f"{column_name}_class"] = cat[0]


def winRate(DF):
    "function to calculate win rate of intraday trading strategy"
    df = DF["return"]
    pos = df[df>1]
    neg = df[df<1]
    if len(pos+neg) ==0:
        return 0
    return (len(pos)/len(pos+neg))*100

def meanretpertrade(DF):
    df = DF["return"]
    df_temp = (df-1).dropna()
    return df_temp[df_temp!=0].mean()

def meanretwintrade(DF):
    df = DF["return"]
    df_temp = (df-1).dropna()
    return df_temp[df_temp>0].mean()

def meanretlostrade(DF):
    df = DF["return"]
    df_temp = (df-1).dropna()
    return df_temp[df_temp<0].mean()
def maxconsectvloss(DF):
    df = DF["return"]
    df_temp = df.dropna()
    df_temp2 = np.where(df_temp < 1, 1, 0)
    
    count_consecutive = []
    seek = 0
    for val in df_temp2:
        if val == 0:
            seek = 0
        else:
            seek += 1
            count_consecutive.append(seek)
    
    # If we never appended anything, it means no losses were found:
    if len(count_consecutive) == 0:
        return 0  # or return None, or handle however you prefer
    
    return max(count_consecutive)

def calculate_daily_ohlc(df):
    df = df.copy()
    df["dayOpen"] = df.groupby("day")["Open"].transform("first")
    df["dayClose"] = df.groupby("day")["Close"].transform("last")
    df["dayHigh"] = df.groupby("day")["High"].transform("max")
    df["dayLow"] = df.groupby("day")["Low"].transform("min")
    return df

def calculate_historical_Volumes(df):
    df = df.copy()
    df["Volume1"] = df["Volume"].shift(3) 
    df["Volume2"] = df["Volume"].shift(2) 
    df["Volume3"] = df["Volume"].shift(1) 
    return df
def calculate_historical_highs(df):
    df = df.copy()
    df["dayHigh_1"] = df["dayHigh"].shift(39)  # 3 days ago
    df["dayHigh_2"] = df["dayHigh"].shift(26)  # 2 days ago
    df["dayHigh_3"] = df["dayHigh"].shift(13)  # previous day
    return df

def check_last_row_nans(df):
    """
    Identifies columns that contain NaN values in the last row of a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame to check
        
    Returns:
    --------
    dict
        A dictionary containing:
        - 'columns_with_nan': List of column names containing NaN
        - 'columns_without_nan': List of column names without NaN
        - 'last_row_values': Dictionary of all column values in the last row
        
    Example:
    --------
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, np.nan],
    ...     'B': [4, 5, 6],
    ...     'C': [7, 8, np.nan]
    ... })
    >>> result = check_last_row_nans(df)
    >>> print(result['columns_with_nan'])
    ['A', 'C']
    """
    
    # Get the last row
    last_row = df.iloc[-1]
    
    # Find columns with NaN in last row
    columns_with_nan = last_row[last_row.isna()].index.tolist()
    
    # Create dictionary of all values in last row
    last_row_values = last_row.to_dict()
    
    return {
        'columns_with_nan': columns_with_nan,
        'last_row_values': last_row_values
    }

def initialize_columns(df):
    columns_to_init = [
        "dayOpen", "dayClose", "dayHigh", "dayHigh_1", "dayHigh_2", "dayHigh_3",
        "dayLow", "prevDayLow", "prevDayOpen", "prev2DayOpen", "prevDayClose",
        "prev2DayClose", "Close_1_day_ago", "Close_2_days_ago", "Open_1_day_ago"
    ]
    df = df.copy()
    for col in columns_to_init:
        df[col] = np.nan
    return df

def add_historical_values(df):
    df = df.copy()
    df["prevDayLow"] = df["dayLow"].shift(13)
    df["prevDayOpen"]  = df["dayOpen"].shift(13)
    df["prev2DayOpen"] = df["dayOpen"].shift(26)
    df["prevDayClose"] = df["dayClose"].shift(13)
    df["prev2DayClose"] = df["dayClose"].shift(26)
    df["Close_1_day_ago"] = df["Close"].shift(13)
    df["Close_2_days_ago"] = df["Close"].shift(26)
    df["Open_1_day_ago"] = df["Open"].shift(13)
    return df

def add_daily_returns(df):
    df = df.copy()
    df["return_1d"] = df["Close"] / df["Close_1_day_ago"]
    df["return_2d"] = df["Close"] / df["Close_2_days_ago"]
    df['open_to_prev_close'] = df['dayOpen'] / df['prevDayClose']
    return df

def add_AH_columns(df):
    df = df.copy()
    mask_ah = df["market_time"] == "AH"
    df_ah = df.loc[mask_ah].copy()
    df_ah["AH_max"] = df_ah.groupby("day")["High"].transform("max")
    df["AH_max"] = np.nan
    df.loc[mask_ah, "AH_max"] = df_ah["AH_max"]
    df["AH_max"] = df.groupby("day")["AH_max"].transform("max")

    ahmax_per_day = df.groupby("day")["AH_max"].first().sort_index()
    ahmax_per_day_shifted = ahmax_per_day.shift()
    df["AH_max_1dayago"] = df["day"].map(ahmax_per_day_shifted)
    return df
def add_PM_columns(df):
    df = df.copy()
    mask_pm = df["market_time"] == "PM"
    df_pm = df.loc[mask_pm].copy()
    df_pm["PM_max"] = df_pm.groupby("day")["High"].transform("max")
    df_pm["PM_max_time_in_sec"] = df_pm.groupby("day")["High"].transform(
        lambda x: x.idxmax().hour*3600 
    )
    df["PM_max"] = np.nan
    df["PM_max_time_in_sec"] = np.nan
    df.loc[mask_pm, "PM_max"] = df_pm["PM_max"]
    df.loc[mask_pm, "PM_max_time_in_sec"] = df_pm["PM_max_time_in_sec"]
    df["PM_max"] = df.groupby("day")["PM_max"].transform("max")
    df["PM_max_time_in_sec"] = df.groupby("day")["PM_max_time_in_sec"].transform("max")
    
    df_pm["PM_min"] = df_pm.groupby("day")["Low"].transform("min")
    df_pm["PM_min_time_in_sec"] = df_pm.groupby("day")["Low"].transform(
        lambda x: x.idxmin().hour*3600 
    )
    df["PM_min"] = np.nan
    df["PM_min_time_in_sec"] = np.nan
    df.loc[mask_pm, "PM_min"] = df_pm["PM_min"]
    df.loc[mask_pm, "PM_min_time_in_sec"] = df_pm["PM_min_time_in_sec"]
    df["PM_min"] = df.groupby("day")["PM_min"].transform("min")
    df["PM_min_time_in_sec"] = df.groupby("day")["PM_min_time_in_sec"].transform("min")

    pmmax_per_day = df.groupby("day")["PM_max"].first().sort_index()
    pmmax_per_day_shifted = pmmax_per_day.shift()
    df["PM_max_1dayago"] = df["day"].map(pmmax_per_day_shifted)


    # Sum of volume during the PM session per day
    df_pm["PM_volume_sum"] = df_pm.groupby("day")["Volume"].transform("sum")
    df_pm["PM_volume_max"] = df_pm.groupby("day")["Volume"].transform("max")
    df["PM_volume_sum"] = np.nan
    df["PM_volume_max"] = np.nan
    df.loc[mask_pm, "PM_volume_sum"] = df_pm["PM_volume_sum"]
    df.loc[mask_pm, "PM_volume_max"] = df_pm["PM_volume_max"]
    df["PM_volume_sum"] = df.groupby("day")["PM_volume_sum"].transform("max")
    df["PM_volume_max"] = df.groupby("day")["PM_volume_max"].transform("max")
    return df

def calculate_all_metrics(y_test,y_pred):
    if np.all(y_pred == 0):
        return  (None, None, 1, None, None)
    #if np.all(y_pred == 1):
        #return (None, None, None, None, None)
     
    cm = confusion_matrix(y_test, y_pred)
    if cm.size >= 4:     
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        # Precision for the positive class
        precision = precision_score(y_test, y_pred) #Measures how many predicted "1"s are correct.
        recall = recall_score(y_test, y_pred) #Measures how many actual "1"s are captured
        # F0.5 Score (places more weight on precision)
        f05 = fbeta_score(y_test, y_pred, beta=0.5) #Harmonic mean of precision and recall
        mcc = matthews_corrcoef(y_test, y_pred) # takes into account all four confusion matrix components, but it does not specifically weight FPs more than FNs. Good if you want an overall sense of how well the model is doing

        return precision,recall,specificity,f05,mcc 
    else:
        return  (None, None, None, None, None)
    

def balance_binary_target(df, target_column):
    print(f"before balancing the column {target_column} of the dataframe ")
    count_0 = (df[target_column] == 0).sum()
    count_1 = (df[target_column] == 1).sum()
    print(f" count 0 : {count_0}")
    print(f" count 1 : {count_1}")
    
    rows_to_remove = abs(count_1 - count_0)
    
    if rows_to_remove > 0:
        if count_1 > count_0:
            indices_to_remove = df[df[target_column] == 1].index
        else:
            indices_to_remove = df[df[target_column] == 0].index
        
        indices_to_remove = random.sample(list(indices_to_remove), rows_to_remove)
        df = df.drop(indices_to_remove)
    return df

def extract_number_of_neg_slopes(df_stock):
    df_stock = df_stock.sort_values(['Date'])
    """
    Count how many times 'Close' decreases between two consecutive rows.
    
    Parameters:
        df_stock (pd.DataFrame): A DataFrame filtered for a specific stock, 
                                 sorted by 'Date'.
    
    Returns:
        int: Number of times the 'Close' value decreases between consecutive rows.
    """
    close_diff = df_stock['Close'].diff()
    return (close_diff < 0).sum()

def extract_redness_index(df_stock: pd.DataFrame) -> float:
    """
    Quantify how "red" a stock has been over the period in df_stock.

    We compute pct_change() on 'Close', then sum:
      - the absolute values of all negative returns
      - all positive returns
    The redness index is:
        redness = (sum |negative returns|) / (sum |negative returns| + sum positive returns)

    Parameters:
        df_stock (pd.DataFrame): Must contain 'Date' and 'Close' columns.
                                 'Date' need not be sorted.

    Returns:
        float: Redness index in [0,1]. 
               1.0 ⇒ all moves were down; 
               0.0 ⇒ all moves were up; 
               0.5 ⇒ equal weight of up‑ and down‑magnitude.
               np.nan if there are no non‑zero moves.
    """
    # ensure chronological order
    df = df_stock.sort_values('Date')
    # compute daily returns
    returns = df['Close'].pct_change().dropna()

    # separate positive and negative moves
    neg_total = returns[returns < 0].abs().sum()
    pos_total = returns[returns > 0].sum()
    total = neg_total + pos_total

    if total == 0:
        return float('nan')  # no movement

    return neg_total / total

def compute_number_of_2reds_14window_startend(df_stock: pd.DataFrame) -> float:
    #to do  : create a similar function, that compute the diff between max and min of the 14 days
    df = df_stock.sort_values('Date').reset_index(drop=True)
    if len(df) < 14:
        return float('nan')
    start_prices = df['Close'].iloc[:-13].values
    end_prices = df['Close'].iloc[13:].values
    red_count = np.sum(end_prices <= start_prices * 0.98)
    
    return red_count


def additional_ratios(df,available_features_by_stock=None):
    df = df.copy()
    #if available_features_by_stock is None or ('PM_max_time_in_sec' in available_features_by_stock and 'PM_min_time_in_sec' in available_features_by_stock): 
        #df['PM_time_diff'] = df['PM_max_time_in_sec'] - df['PM_min_time_in_sec']
    if available_features_by_stock is None or ('PM_max' in available_features_by_stock and 'PM_min' in available_features_by_stock): 
        df['PM_max_to_min_ratio'] = df['PM_max'] / df['PM_min']
    if available_features_by_stock is None or ('PM_min' in available_features_by_stock and 'dayOpen' in available_features_by_stock):
        df['PM_min_to_open_ratio'] = df['PM_min'] / df['dayOpen']
    if available_features_by_stock is None or ('PM_max' in available_features_by_stock and 'dayOpen' in available_features_by_stock and 'PM_min' in available_features_by_stock):
        df['PM_range_to_open_ratio'] = (df['PM_max'] - df['PM_min']) / df['dayOpen']
    if available_features_by_stock is None or ('PM_max' in available_features_by_stock and 'PM_min' in available_features_by_stock and 'Close' in available_features_by_stock):
        df['PM_range_to_close_ratio'] = (df['PM_max'] - df['PM_min']) / df['Close']
    if available_features_by_stock is None or ('dayOpen' in available_features_by_stock):
        df['Close_to_open_ratio'] = df['Close'] / df['dayOpen']
    if available_features_by_stock is None or ('dayOpen' in available_features_by_stock and 'prevDayClose' in available_features_by_stock):
        df['dayOpen_to_prevDayClose'] = df['dayOpen'] / df['prevDayClose']
    if available_features_by_stock is None or ('Close_1_day_ago' in available_features_by_stock and 'Close_2_days_ago' in available_features_by_stock):
        df['hist_close_ratio'] = df['Close_1_day_ago'] / df['Close_2_days_ago']
    if available_features_by_stock is None or ('dayOpen' in available_features_by_stock and 'prevDayOpen' in available_features_by_stock):
        df['dayOpen_to_prevDayOpen_ratio'] = (df['dayOpen'] / df['prevDayOpen'])
    if available_features_by_stock is None or ('dayOpen' in available_features_by_stock and 'prev2DayOpen' in available_features_by_stock):
        df['dayOpen_to_prev2DayOpen_ratio'] = (df['dayOpen'] / df['prev2DayOpen'])
    if available_features_by_stock is None or ('Open_1_day_ago' in available_features_by_stock and 'Close_1_day_ago' in available_features_by_stock):
        df['Open_1_day_ago_to_Close_1_day_ago_ratio'] = (df['Open_1_day_ago'] / df['Close_1_day_ago'])
    if available_features_by_stock is None or ('return_1d' in available_features_by_stock and 'return_2d' in available_features_by_stock):
        df['return_1d_to_return_2d_ratio'] = (df['return_1d'] / df['return_2d'])
    
    if available_features_by_stock is None or ('prev2DayClose' in available_features_by_stock and 'prevDayClose' in available_features_by_stock):
        df['prev2DayClose_to_prevDayClose_ratio'] = df['prev2DayClose'] / df['prevDayClose']
    if available_features_by_stock is None or ('PM_max' in available_features_by_stock and 'PM_max_1dayago' in available_features_by_stock):
        df['PM_max_vs_PM_max_1dayago'] = df['PM_max'] / df['PM_max_1dayago']
    if available_features_by_stock is None or ('AH_max_1dayago' in available_features_by_stock and 'Close' in available_features_by_stock):
        df['AH_max_1dayago_to_Close'] = df['AH_max_1dayago'] / df['Close']
    if available_features_by_stock is None or ("AH_max_1dayago" in available_features_by_stock and "PM_max" in available_features_by_stock):
        df['AH_max_1dayago_vs_PM_max'] = df['AH_max_1dayago'] / df['PM_max']
    if available_features_by_stock is None or ('AH_max_1dayago' in available_features_by_stock and 'prevDayClose' in available_features_by_stock):
        df['AH_max_1dayago_vs_prevDayClose'] = df['AH_max_1dayago'] / df['prevDayClose']

    if available_features_by_stock is None or ('prevDayClose' in available_features_by_stock and 'Close' in available_features_by_stock):
        df['Close_to_prevDayLow'] = df['Close'] / df['prevDayLow']
    if available_features_by_stock is None or ('dayHigh_3' in available_features_by_stock):
        df['Close_to_prevDayHigh'] = df['Close'] / df['dayHigh_3']
    return df

def calculate_volume_slopes(df,available_features_by_stock=None):
    df = df.copy()
    if available_features_by_stock is None or ('Volume1' in available_features_by_stock and 'Volume2' in available_features_by_stock and 'Volume3' in available_features_by_stock):
        df['slope_a_vol_rel'] = (-6*df['Volume1'] - 2*df['Volume2'] + 2*df['Volume3'] + 6*df['Volume']) / 20
        df['coef_p_vol_rel'] = (df['Volume1'] - df['Volume2'] -df['Volume3'] + df['Volume']) / 4
        df['coef_q_vol_rel'] = (-21*df['Volume1'] +13* df['Volume2'] +17*df['Volume3'] -9* df['Volume']) / 20

        df['slope_a_vol_rel'] = np.where(
            df['Volume1'] != 0, 
            (df['slope_a_vol_rel'].astype(float) / df['Volume1'].astype(float)) * 100, 
            np.nan 
        )
        df['coef_p_vol_rel'] = np.where(
            df['Volume1'] != 0,
            (df['coef_p_vol_rel'].astype(float) / df['Volume1'].astype(float)) * 100,
            np.nan
        )
        
        df['coef_q_vol_rel'] = np.where(
            df['Volume1'] != 0, 
            (df['coef_q_vol_rel'].astype(float) / df['Volume1'].astype(float)) * 100, 
            np.nan 
        )
    return df
def calculate_slopes(df,available_features_by_stock=None):
    df = df.copy()
    if available_features_by_stock is None or ('dayHigh_1' in available_features_by_stock and 'dayHigh_3' in available_features_by_stock):
        df['high_slope_rel'] = (df['dayHigh_3'] - df['dayHigh_1']) / 2  #https://chatgpt.com/share/679ce81a-ae24-800e-83f9-61efd07dad61
    if available_features_by_stock is None or ('dayHigh_1' in available_features_by_stock and 'dayHigh_2' in available_features_by_stock and 'dayHigh_3' in available_features_by_stock):
        df['high_quad_p_rel'] = (df['dayHigh_1']+ df['dayHigh_3']-2*df['dayHigh_2']) / 2
        df['high_quad_q_rel'] = (-5*df['dayHigh_1']-3* df['dayHigh_3']+8*df['dayHigh_2']) / 2

    if available_features_by_stock is None or ('dayHigh_1' in available_features_by_stock and 'dayHigh_3' in available_features_by_stock):
        df['high_slope_rel'] = (df['high_slope_rel']/df["Close"])*100
    if available_features_by_stock is None or ('dayHigh_1' in available_features_by_stock and 'dayHigh_2' in available_features_by_stock and 'dayHigh_3' in available_features_by_stock):
        df['high_quad_p_rel'] = (df['high_quad_p_rel']/df["Close"])*100
        df['high_quad_q_rel'] = (df['high_quad_q_rel']/df["Close"])*100

    return df


def remove_extreme_rows(df, exclude_columns=None, threshold=10**9):
    if exclude_columns is None:
        exclude_columns = []
    columns_to_check = [col for col in df.columns if col not in exclude_columns]
    df_to_check = df[columns_to_check]
    mask_greater = (df_to_check  > threshold).any(axis=1)
    mask_less = (df_to_check  < -threshold).any(axis=1)
    mask_inf = df_to_check .isin([np.inf, -np.inf]).any(axis=1)
    combined_mask = mask_greater | mask_less | mask_inf
    return df[~combined_mask]

def process_intraday_signals(df_day):
    """
    For each row in df_day, look at *later* rows to see if we 
    hit the take_profit price before the stop_loss price.
    Return a list of 0/1 signals in the same order.
    """
    df_day = df_day.copy()  # to avoid SettingWithCopy warnings
    signals = []
    
    # We’ll walk forward for each row
    for i in range(len(df_day)):
        take_profit = df_day.iloc[i]['take_profit_intrday']
        stop_loss   = df_day.iloc[i]['stop_loss_intrday']
        
        # Default assumption is we do NOT hit the target => 0
        to_buy_val = 0
        
        # Look at subsequent rows *within this day*
        for j in range(i+1, len(df_day)):
            high_j = df_day.iloc[j]['High']
            low_j  = df_day.iloc[j]['Low']
            
            # Check if take-profit is reached before stop-loss
            if high_j >= take_profit:
                to_buy_val = 1
                break
            if low_j <= stop_loss:
                to_buy_val = 0
                break
        
        signals.append(to_buy_val)
    
    df_day['to_buy_intraday'] = signals
    return df_day

def check_if_goal_reached(goal,total_cash_value) :
    if total_cash_value>=goal:
        return True
    return False

def calculate_sma_ratios(df):
    df = df.copy()
    df['ratio_sma1'] = df['Close'] / df['SMA_40']
    df['ratio_sma2'] = df['SMA_40'] / df['SMA_50']
    df['ratio_sma3'] = df['Close'] / df['SMA_60']
    return df
def calculate_ema_ratios(df,available_features_by_stock=None):
    df = df.copy()
    if available_features_by_stock is None or ('EMA_13' in available_features_by_stock and 'EMA_48' in available_features_by_stock):
        df['ema_ratio1'] = df['EMA_13'] / df['EMA_48']
    if available_features_by_stock is None or ('EMA_3' in available_features_by_stock and 'EMA_13' in available_features_by_stock):
        df['ema_ratio2'] = df['EMA_3'] / df['EMA_13']
    if available_features_by_stock is None or ('EMA_48' in available_features_by_stock):
        df['Close_to_EMA_48'] = df['Close'] / df['EMA_48']
    return df
def calculate_highs_and_lows(df):
    df = df.copy()
    if 'Stock' in df.columns:
        # Group by Stock if the column exists
        df['High_10'] = (
            df.groupby('Stock')['High']
            .transform(lambda x: x.rolling(window=10).max())
        )
        df['Low_10'] = (
            df.groupby('Stock')['Low']
            .transform(lambda x: x.rolling(window=10).min())
        )
        df['High_20'] = (
            df.groupby('Stock')['High']
            .transform(lambda x: x.rolling(window=20).max())
        )
        df['Low_20'] = (
            df.groupby('Stock')['Low']
            .transform(lambda x: x.rolling(window=20).min())
        )
    else:
        # If 'Stock' column doesn't exist, assume all rows are for the same stock
        df['High_10'] = df['High'].rolling(window=10).max()
        df['Low_10'] = df['Low'].rolling(window=10).min()
        df['High_20'] = df['High'].rolling(window=20).max()
        df['Low_20'] = df['Low'].rolling(window=20).min()
        
    df['close_to_High10'] = df['Close'] / df['High_10']
    df['close_to_Low10'] = df['Close'] / df['Low_10']
    df['close_to_High20'] = df['Close'] / df['High_20']
    df['close_to_Low20'] = df['Close'] / df['Low_20']
    return df

def add_shifted_fractional_diff_from_DEPRADO(df):
    #target_date = pd.Timestamp("2025-03-04 10:00:00")
    #previous_rows = df.loc[:target_date].iloc[-21:-1][['Close']]
    #print(previous_rows)
    #target_pos = df.index.get_loc(target_date)
    max_shift = 14
    #print(f"Original 'Close' value at {target_date}: {df.at[target_date, 'Close']}")
    for shift in range(1, max_shift + 1):
        col_name = get_col_name(shift)
        if 'Stock' in df.columns:
            df[col_name] = df.groupby('Stock')['Close'].shift(shift)
        else:
            df[col_name] = df['Close'].shift(shift)

    coeffs = [1, -0.8, -0.08, -0.032, -0.0176, -0.0113, -0.0079, -0.0059,
          -0.0045, -0.0036, -0.003, -0.0025, -0.0021]  
    for fd, offset in zip(['FD_1', 'FD_2', 'FD_3'], [0, 1, 2]):
        df[fd] = sum(coeff * df[get_col_name(i + offset)]
                            for i, coeff in enumerate(coeffs))
    fd1_cols = [get_col_name(i) for i in range(1, 13)]
    df.loc[df[fd1_cols].isnull().any(axis=1), 'FD_1'] = None
    fd2_cols = [get_col_name(i) for i in range(1, 14)]
    df.loc[df[fd2_cols].isnull().any(axis=1), 'FD_2'] = None
    fd3_cols = [get_col_name(i) for i in range(2, 15)]
    df.loc[df[fd3_cols].isnull().any(axis=1), 'FD_3'] = None

    return df
def add_momentum(group,stock,valid_columns_dict =None):
    # 5-bar liquidity-adjusted momentum
    # Numerator: Close(t) - Close(t-5)
    # Denominator: sum of volumes between t-5 and t (rolling 5 if intraday)
    # df["mom_5"] = (df["Close"] - df["Close"].shift(5)) \
    #                 / df["Volume"].rolling(window=5).sum()
    if valid_columns_dict is None or (stock in valid_columns_dict and 'Volume' in valid_columns_dict[stock]):
        mom_5 = (group["Close"] - group["Close"].shift(5)) / group["Volume"].rolling(window=5).sum()
    else: 
        mom_5 = None
    return mom_5
   
def add_volatilities(df):
    df = df.copy()
    df["return"] = df["Close"].pct_change()
    df["vol_5"]  = df["return"].rolling(window=5).std()
    df["vol_20"] = df["return"].rolling(window=20).std()
    df["vol_50"] = df["return"].rolling(window=50).std()
    return df
def additional_Close_ratios(df,available_features_by_stock = None):
    df = df.copy()
    if available_features_by_stock is None or ('prevDayClose' in available_features_by_stock):
        df['Close_to_prevDayClose'] = df['Close'] / df['prevDayClose']
    if available_features_by_stock is None or ('Close_1_day_ago' in available_features_by_stock):
        df['Close_to_Close_1_day_ago'] = df['Close'] / df['Close_1_day_ago']
    if available_features_by_stock is None or ('prevDayOpen' in available_features_by_stock):
        df['Close_to_prevDayOpen'] = df['Close'] / df['prevDayOpen']
    return df
def additional_PM_ratios(df,available_features_by_stock=None):
    df = df.copy()
    if available_features_by_stock is None or ('PM_max' in available_features_by_stock and 'PM_min' in available_features_by_stock):
        df['PM_max_to_PM_min_ratio'] = df['PM_max'] / df['PM_min']
    if available_features_by_stock is None or ('PM_max' in available_features_by_stock and 'dayOpen' in available_features_by_stock):
        df['PM_max_to_dayOpen_ratio'] = df['PM_max'] / df['dayOpen']
    if available_features_by_stock is None or ('PM_max' in available_features_by_stock and 'prevDayClose' in available_features_by_stock):
        df['PM_max_to_prevDayClose_ratio'] = df['PM_max'] / df['prevDayClose']
    if available_features_by_stock is None or ('PM_min' in available_features_by_stock and 'prevDayClose' in available_features_by_stock):
        df['PM_min_to_prevDayClose_ratio'] = df['PM_min'] / df['prevDayClose']
    if available_features_by_stock is None or ('PM_max' in available_features_by_stock and 'prevDayOpen' in available_features_by_stock):
        df['PM_max_to_prevDayOpen_ratio'] = df['PM_max'] / df['prevDayOpen']
    if available_features_by_stock is None or ('PM_min' in available_features_by_stock and 'prevDayOpen' in available_features_by_stock):
        df['PM_min_to_prevDayOpen_ratio'] = df['PM_min'] / df['prevDayOpen']
    if available_features_by_stock is None or ('PM_max' in available_features_by_stock and 'Close' in available_features_by_stock):
        df['PM_max_to_Close_ratio'] = df['PM_max'] / df['Close']
    if available_features_by_stock is None or ('PM_min' in available_features_by_stock and 'Close' in available_features_by_stock):
        df['PM_min_to_Close_ratio'] = df['PM_min'] / df['Close']

    return df

def calculate_smas_old(df):
    df = df.copy()
    
    df["SMA_40"] = (
        df.groupby(df['Date'].dt.date)["Close"]
        .transform(lambda x: x.rolling(window=40, min_periods=1).mean())
    )
    df["SMA_55"] = (
        df.groupby(df['Date'].dt.date)["Close"]
        .transform(lambda x: x.rolling(window=55, min_periods=1).mean())
    )
    df["SMA_70"] = (
        df.groupby(df['Date'].dt.date)["Close"]
        .transform(lambda x: x.rolling(window=70, min_periods=1).mean())
    )
    df['ratio_sma1'] = df['Close'] / df['SMA_40']
    df['ratio_sma2'] = df['SMA_40'] / df['SMA_55']
    df['ratio_sma3'] = df['Close'] / df['SMA_70']
    return df

def calculate_smas(df):
    df = df.copy()
    mask_rth = df["market_time"] == "RTH" 
    df.loc[mask_rth, "SMA_40"] = (
            df.loc[mask_rth]
              .groupby(df.loc[mask_rth].index.date)["Close"]
              .transform(lambda x: x.rolling(window=40).mean())
        )
    
    df.loc[mask_rth, "SMA_50"] = (
        df.loc[mask_rth]
            .groupby(df.loc[mask_rth].index.date)["Close"]
            .transform(lambda x: x.rolling(window=50).mean())
    )
    df.loc[mask_rth, "SMA_60"] = (
        df.loc[mask_rth]
            .groupby(df.loc[mask_rth].index.date)["Close"]
            .transform(lambda x: x.rolling(window=60).mean())
    )
    return df

def calculate_stds(df):
    #to do take into accoun High, Low
    df = df.copy()
    
    mask_rth = df["market_time"] == "RTH"

    df.loc[mask_rth, "STD_10"] = df.loc[mask_rth]["Close"].rolling(window=10).std()
    df.loc[mask_rth, "STD_30"] = df.loc[mask_rth]["Close"].rolling(window=30).std()
    
    return df
def calculate_emas(df):
    df = df.copy()
    mask_rth = df["market_time"] == "RTH" #todo add ema 30 and 8
    df.loc[mask_rth, "EMA_3"] = (
        df.loc[mask_rth, "Close"]
        .ewm(span=3, adjust=False)
        .mean()
    )
    
    df.loc[mask_rth, "EMA_13"] = (
        df.loc[mask_rth, "Close"]
        .ewm(span=13, adjust=False)
        .mean()
    )
    
    df.loc[mask_rth, "EMA_48"] = (
        df.loc[mask_rth, "Close"]
        .ewm(span=48, adjust=False)
        .mean()
    )
    return df

def get_col_name(shift):
    return 'Close' if shift == 0 else ('prev_close' if shift == 1 else f'prev{shift}_close')

def assign_bin_categories(row_features, bin_dict):
    """
    Assign bin categories to each feature in row_features based on the bin edges
    provided in bin_dict. Returns a modified copy of row_features.
    """
    for col, bin_edges in bin_dict.items(): #loaded_bin_dict_json or bin_dict
        if col in row_features:
            # pd.cut on a list with a single element, extract category from the resulting Categorical object
            # Use pd.cut on a list containing the single value.
            # This returns a Categorical object; we extract its first (and only) element.
            category = pd.cut(
                [row_features[col]], # List with one element.
                bins=bin_edges,
                labels=range(1, len(bin_edges)),
                include_lowest=True   # Ensure the lowest value is included.
            )[0]
            row_features[f"{col}_class"] = category

    return row_features

def extracts_all_features_from_json(json_file):
    subsets = [model["subset"] for model in json_file.values()]

    # Flatten the list of features
    all_features = [feature for subset in subsets for feature in subset]
    unique_features = list(set(all_features))
    return unique_features

def get_stock_symbols_from_json(data):
    #with open(json_file_path, "r") as file:
        #data = json.load(file)
    return list(data.keys())

def extract_features_of_stock(json_file,stock):
    subset = json_file[stock]["subset"]
    return subset

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

def plot_monthly_precision_all_tickers(df,ticker_precisions):
    

    # Set a color palette with enough distinct colors
    # E.g., 'tab10' or 'Paired' or 'husl' for many distinct colors
    sns.set_palette("tab10", n_colors=len(ticker_precisions))

    plt.figure(figsize=(12, 7))
    good_tickers = []
    very_good_tickers = []
    for ticker, monthly_data in ticker_precisions.items():
        months = sorted(monthly_data.keys())[1:-1]  # Remove the first and last month  because it has fewer rows
        precisions = [monthly_data[m] for m in months]
        precisions_without_None = [monthly_data[m] for m in months if monthly_data[m] is not None]
        print(f"{ticker} precision : ")
        print(precisions)
        if min(precisions_without_None) > 0.7:
            good_tickers.append(ticker)
        if min(precisions_without_None) > 0.78:
            very_good_tickers.append(ticker)
        plt.plot(months, precisions, linestyle='-', label=ticker)

    print("good tickers : ")
    print(good_tickers)
    print("verygood tickers : ")
    print(very_good_tickers)
    plt.xlabel("Month")
    plt.ylabel("Precision")
    plt.title("Monthly Prediction Precision by Ticker")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    return plt.gcf()

def plot_monthly_precision_all_tickers_old(df):
    # Create a dictionary to store monthly precision for each ticker
    ticker_precisions = {}
    
    # Calculate precision for each ticker
    for ticker in df['Stock'].unique():
        ticker_df = df[df['Stock'] == ticker]
        monthly_precision = {}
        
        # Group by month and calculate precision
        for month, group in ticker_df.groupby(ticker_df['Date'].dt.to_period('M').astype(str)):
            precision, _, _, _, _ = calculate_all_metrics(group['to_buy_1d'], group['prediction'])
            monthly_precision[month] = precision
        
        ticker_precisions[ticker] = monthly_precision
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    
    # Plot each ticker
    for ticker, monthly_data in ticker_precisions.items():
        months = sorted(monthly_data.keys())
        precisions = [monthly_data[m] for m in months]
        plt.plot(months, precisions, marker='o', linestyle='-', label=ticker)
    
    # Customize the plot
    plt.xlabel("Month")
    plt.ylabel("Precision")
    plt.title("Monthly Prediction Precision by Ticker")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    
    return plt.gcf()


def get_distinct_colors(n_colors):
    """
    Generate distinct colors using multiple color palettes
    """
    if n_colors < 6:
        # High-contrast, distinct colors suitable for visualization
        distinct_few = [
            '#e41a1c',  # red
            '#377eb8',  # blue
            '#4daf4a',  # green
            '#984ea3',  # purple
            '#ff7f00',  # orange
        ]
        return distinct_few[:n_colors]
    # Combine multiple seaborn color palettes for more distinct colors
    palette1 = sns.color_palette("husl", n_colors // 2)  # Evenly spaced colors in HUSL space
    palette2 = sns.color_palette("Set2", min(8, n_colors // 4))  # Qualitative color palette
    palette3 = sns.color_palette("Dark2", min(8, n_colors // 4))  # Another qualitative palette
    
    # Combine palettes and ensure we have enough colors
    all_colors = palette1 + palette2 + palette3
    # If we still need more colors, add a tab20 palette
    if len(all_colors) < n_colors:
        additional_colors = plt.cm.tab20(np.linspace(0, 1, n_colors - len(all_colors)))
        all_colors.extend(additional_colors)
    
    return all_colors[:n_colors]

import shutil
def cleanup_folders(folders_to_delete):
    """
    Delete specific folders and their contents if they exist and log the results.
    
    Args:
        folders_to_delete (list): List of folder paths to delete
    """
    print("The following folders are scheduled for deletion:")
    for file_name in folders_to_delete:
        print(f"- {file_name}")
    
    # Ask for user confirmation
    confirmation = input("Are you sure you want to delete these files? (yes/no): ").lower().strip()
    
    if confirmation != 'yes':
        print("Deletion aborted by user.")
        return
    
    for folder_name in folders_to_delete:
        try:
            if os.path.exists(folder_name):
                if os.path.isdir(folder_name):
                    shutil.rmtree(folder_name)
                    print(f"Successfully deleted folder {folder_name}")
                else:
                    print(f"{folder_name} is not a directory")
            else:
                print(f"{folder_name} does not exist - no action needed")
        except Exception as e:
            print(f"Error deleting folder {folder_name}: {str(e)}")

def cleanup_files(files_to_delete):
    """
    Delete specific files if they exist and log the results, with user confirmation.
    """
    # Show the list of files to be deleted
    print("The following files are scheduled for deletion:")
    for file_name in files_to_delete:
        print(f"- {file_name}")
    
    # Ask for user confirmation
    confirmation = input("Are you sure you want to delete these files? (yes/no): ").lower().strip()
    
    if confirmation != 'yes':
        print("Deletion aborted by user.")
        return
    
    # Proceed with deletion if confirmed
    for file_name in files_to_delete:
        try:
            if os.path.exists(file_name):
                os.remove(file_name)
                print(f"Successfully deleted {file_name}")
            else:
                print(f"{file_name} does not exist - no action needed")
        except Exception as e:
            print(f"Error deleting {file_name}: {str(e)}")

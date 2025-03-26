
import pandas as pd
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.tree import export_text
from sklearn.metrics import accuracy_score, classification_report, precision_score, confusion_matrix
from xgboost import XGBClassifier
import random
import numpy as np
from sklearn.svm import SVC
import pickle
import os 
import json
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from utils import *

TARGET = 'to_buy_1d'
SAVE_MODEL= 1
TESTING_LENGTH_IN_DAYS = 10
TRAINING_STARTING_LAG = 300
LOOKAHEAD_NUMBER = 24
USE_THIS_SCRIPT_FOR_METADATA_GENERATION = 1 #if false it means we use it to generate files for live trading
#if true it means we will generate files that will help us find the best meta parameters for the model

cleanup_files(['models_to_use.json','model_output.csv','debugging_from_backtest.csv'])
cleanup_folders(['allmodels','allscalers'])
folder = f'allmodels'
os.makedirs(folder, exist_ok=True)
scaler_folder = f'allscalers'
os.makedirs(scaler_folder, exist_ok=True)

df = pd.read_csv('clean.csv')
df = df.sample(frac=1).reset_index(drop=True)
df.drop_duplicates(inplace=True)

features_to_exclude = ['PM_time_diff_class'] #because i suspect models with this feature tend to overfit
df = df.drop(columns=features_to_exclude,errors  ='ignore')

print("number of None in the column TARGET : ",df[TARGET].isna().sum())
#DELETE THESE ROWS
df = df.dropna(subset=[TARGET])
#df = df.dropna()
#df = df[df['Stock'].isin(['BIO', 'TEAM','LW'])]



df["Date"] = pd.to_datetime(df["Date"])
today = pd.to_datetime("today")

#all_data = all_data.dropna()
if USE_THIS_SCRIPT_FOR_METADATA_GENERATION:
    cutoff = today.normalize() - pd.Timedelta(days=random.randint(8, 35))
    cutoff = cutoff + pd.Timedelta(hours=17)  # Set cutoff time at 17H

    TESTING_LENGTH_IN_DAYS = random.randint(2, 80)
    TRAINING_STARTING_LAG = random.randint(270, 350)
    print("cutoff = ",cutoff)
    print("TESTING_LENGTH_IN_DAYS = ",TESTING_LENGTH_IN_DAYS)
    print("TRAINING_STARTING_LAG = ",TRAINING_STARTING_LAG)
    original_df = df.copy()
    #delete all rows after cutoff
    df = df[df['Date']<=cutoff]

    weights = [40 - i for i in range(40)]
    LOOKAHEAD_NUMBER = random.choices(range(40), weights=weights, k=1)[0]
    # because each row's label depends on data in the next 24 hours (approx).
    lookahead = pd.Timedelta(hours=LOOKAHEAD_NUMBER)
    print("lookahead = ",lookahead)
else:
    cutoff = today 
    # because each row's label depends on data in the next 24 hours (approx).
    lookahead = pd.Timedelta(hours=LOOKAHEAD_NUMBER)

train_start = cutoff.normalize() - pd.Timedelta(days=TRAINING_STARTING_LAG)
train_end   = cutoff - pd.Timedelta(days=TESTING_LENGTH_IN_DAYS) #we don't normalize because lets say cutoff ends at 17H (end of market session), so test should also end at 17H (end of market session)
# Adjust train_end if it falls on a weekend
if train_end.weekday() == 5:  # Saturday
    train_end -= pd.Timedelta(days=1)
elif train_end.weekday() == 6:  # Sunday
    train_end -= pd.Timedelta(days=2)

# Define the two test splits:
if TESTING_LENGTH_IN_DAYS>=40:
    test_split1_start = train_end  
    test_split1_end   = cutoff - pd.Timedelta(days=int(TESTING_LENGTH_IN_DAYS/2))  

    test_split2_start = test_split1_end
    test_split2_end   = cutoff       

    test_split_global_start = cutoff - pd.Timedelta(days=100) 
    test_split_global_end = cutoff

    # print("test_split1_start : ",test_split1_start)
    # print("test_split1_end : ",test_split1_end)
    # print("test_split2_start : ",test_split2_start)
    # print("test_split2_end : ",test_split2_end)
    # print("test_split_global_start : ",test_split_global_start)
    # print("test_split_global_end : ",test_split_global_end)

models = {
    'DT1': DecisionTreeClassifier(), 
    #'DT2': DecisionTreeClassifier(max_depth=10, min_samples_split=4), this never give good results
    'DT3': DecisionTreeClassifier(max_depth=15, min_samples_leaf=5, min_samples_split=10),
    'DT4': DecisionTreeClassifier(max_depth=20, criterion='entropy', random_state=42),
    #'DT5': DecisionTreeClassifier(min_samples_leaf=3, max_features='sqrt', max_depth=6),
    'DT6': DecisionTreeClassifier(max_depth=8),
    #'DT7': DecisionTreeClassifier(max_depth=None,min_samples_split=5,min_samples_leaf=2,criterion='gini'), not good results
    'DT8':DecisionTreeClassifier(class_weight={0: 3.0, 1: 1.0}, max_depth=15, min_samples_leaf=5, min_samples_split=10),
    'DT9':DecisionTreeClassifier(class_weight={0: 2.0, 1: 1.0}, max_depth=4),

    'RF1': RandomForestClassifier(n_estimators=100, max_features=0.5),
    #'RF1a':RandomForestClassifier(n_estimators=100,max_features=0.5,min_impurity_decrease=0.01), #not good results
    #'RF4a': RandomForestClassifier(n_estimators=100,max_features=None,min_samples_leaf=1,min_samples_split=2),
    'RF3': RandomForestClassifier(n_estimators=100, max_features=None, max_depth=12),
    'RF4': RandomForestClassifier(n_estimators=200, max_features='log2', max_depth=6), 
    'RF5': RandomForestClassifier(n_estimators=150, max_features=None, max_depth=3),
    'RF6': RandomForestClassifier(n_estimators=300,max_depth=10, max_features='sqrt',min_samples_leaf=2,min_samples_split=5),
    'RF7': RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf = 20,max_features='sqrt'),
    'RF8': RandomForestClassifier(class_weight={0: 3.0, 1: 1.0}, n_estimators=150, max_features=None, max_depth=3),
    'RF9': RandomForestClassifier(n_estimators=200,max_features=0.3,min_weight_fraction_leaf=0.05, max_depth=8),

    'BG1': BaggingClassifier(estimator=DecisionTreeClassifier(criterion='entropy',class_weight='balanced', max_depth=10),n_estimators=100,max_samples=0.7,bootstrap=True),

    'XGB1': XGBClassifier(n_estimators=100),
    #'XGB3':XGBClassifier(n_estimators=300,colsample_bytree= 0.9,gamma=0.1,learning_rate=0.1,max_depth=7,min_child_weight=1,subsample=0.7),
    'XGB7':XGBClassifier(n_estimators=300,colsample_bytree= 0.8,gamma=0.1,learning_rate=0.1,max_depth=5,min_child_weight=1,subsample=0.9),
    'XGB9':XGBClassifier(n_estimators=300,colsample_bytree= 0.8,gamma=0,learning_rate=0.1,max_depth=7,min_child_weight=1,subsample=0.7),
    
    #'XGB16':XGBClassifier(n_estimators=300,colsample_bytree=  0.9, gamma= 0, learning_rate=0.3,max_depth=8,min_child_weight=1,subsample=0.8),
    #'XGB19':XGBClassifier(n_estimators=300, colsample_bytree=0.8, gamma=0, learning_rate=0.1, max_depth=7, min_child_weight=1, subsample=0.7, scale_pos_weight=3.0),
}

#with open('data.json', 'r') as file:
    #feature_subsets = json.load(file)

feature_subsets = [
    ['PM_max_to_dayOpen_ratio_class','AH_max_1dayago_vs_prevDayClose_class','PM_time_diff_class','dayOpen_to_prev2DayOpen_ratio_class'],
    ['AH_max_1dayago_vs_prevDayClose_class','PM_min_to_Close_ratio_class','dayOpen_to_prev2DayOpen_ratio_class','day_of_week'],
    ['PM_max_to_PM_min_ratio_class', 'Close_to_Close_1_day_ago_class', 'AH_max_1dayago_vs_prevDayClose_class', 'PM_max_to_prevDayClose_ratio_class', 'AH_max_1dayago_to_Close_class', 'PM_max_vs_PM_max_1dayago_class', 'PM_min_to_Close_ratio_class', 'open_to_prev_close_class', 'PM_max_to_dayOpen_ratio_class', 'PM_time_diff_class', 'dayOpen_to_prev2DayOpen_ratio_class', 'Close_to_prevDayClose_class', 'dayOpen_to_prevDayClose_class', 'Close_to_prevDayOpen_class'],
    ['dayOpen_to_prevDayOpen_ratio_class', 'dayOpen_to_prevDayClose_class', 'PM_time_diff_class', 'PM_min_to_open_ratio_class', 'AH_max_1dayago_vs_prevDayClose_class', 'PM_max_vs_PM_max_1dayago_class', 'PM_max_to_dayOpen_ratio_class', 'PM_min_to_Close_ratio_class', 'PM_max_to_prevDayOpen_ratio_class', 'Close_to_open_ratio_class', 'dayOpen_to_prev2DayOpen_ratio_class'],
    ['Close_to_prevDayClose_class', 'open_to_prev_close_class', 'AH_max_1dayago_to_Close_class', 'prev2DayClose_to_prevDayClose_ratio_class', 'PM_min_to_prevDayClose_ratio_class', 'PM_min_to_open_ratio_class', 'dayOpen_to_prevDayClose_class', 'PM_min_to_Close_ratio_class', 'AH_max_1dayago_vs_prevDayClose_class', 'PM_max_to_prevDayClose_ratio_class', 'dayOpen_to_prevDayOpen_ratio_class'],
    ['PM_min_to_prevDayOpen_ratio_class', 'Close_to_Close_1_day_ago_class', 'PM_max_to_Close_ratio_class', 'dayOpen_to_prevDayOpen_ratio_class', 'dayOpen_to_prevDayClose_class', 'AH_max_1dayago_vs_prevDayClose_class', 'return_2d_class', 'PM_min_to_Close_ratio_class', 'Close_to_prevDayClose_class', 'dayOpen_to_prev2DayOpen_ratio_class', 'PM_min_to_open_ratio_class', 'day_of_week'],
    ['PM_min_to_prevDayOpen_ratio_class', 'Close_to_Close_1_day_ago_class', 'AH_max_1dayago_vs_PM_max_class', 'PM_range_to_close_ratio_class', 'prev2DayClose_to_prevDayClose_ratio_class', 'day_of_week', 'dayOpen_to_prev2DayOpen_ratio_class', 'Close_to_prevDayOpen_class', 'dayOpen_to_prevDayClose_class', 'PM_max_to_prevDayOpen_ratio_class', 'PM_max_to_dayOpen_ratio_class', 'AH_max_1dayago_to_Close_class', 'PM_time_diff_class', 'Close_to_open_ratio_class'],
    ['day_of_week', 'open_to_prev_close_class', 'Close_to_prevDayOpen_class', 'PM_max_to_dayOpen_ratio_class', 'AH_max_1dayago_vs_prevDayClose_class', 'PM_min_to_Close_ratio_class', 'PM_min_to_open_ratio_class', 'PM_max_to_prevDayOpen_ratio_class', 'AH_max_1dayago_vs_PM_max_class', 'Close_to_open_ratio_class'],
    ['day_of_week', 'PM_min_to_prevDayClose_ratio_class', 'PM_time_diff_class', 'PM_range_to_close_ratio_class', 'PM_max_to_dayOpen_ratio_class', 'prev2DayClose_to_prevDayClose_ratio_class', 'PM_max_to_prevDayOpen_ratio_class', 'PM_min_to_open_ratio_class', 'dayOpen_to_prevDayOpen_ratio_class', 'PM_max_to_PM_min_ratio_class', 'Close_to_open_ratio_class', 'PM_min_to_Close_ratio_class', 'dayOpen_to_prev2DayOpen_ratio_class', 'Close_to_prevDayOpen_class'],
    ['PM_range_to_close_ratio_class', 'prev2DayClose_to_prevDayClose_ratio_class', 'high_quad_q_rel_class', 'day_of_week', 'Close_to_prevDayOpen_class', 'PM_max_to_dayOpen_ratio_class', 'AH_max_1dayago_to_Close_class', 'high_quad_p_rel_class', 'PM_min_to_prevDayOpen_ratio_class', 'Close_to_open_ratio_class', 'PM_max_to_Close_ratio_class', 'Close_to_prevDayClose_class', 'AH_max_1dayago_vs_prevDayClose_class', 'PM_min_to_Close_ratio_class', 'time_in_minutes'],
    ['dayOpen_to_prevDayClose_class', 'prev2DayClose_to_prevDayClose_ratio_class', 'AH_max_1dayago_vs_prevDayClose_class', 'Close_to_open_ratio_class', 'PM_range_to_open_ratio_class', 'PM_range_to_close_ratio_class', 'Close_to_Close_1_day_ago_class', 'high_slope_rel_class', 'day_of_week', 'PM_max_to_PM_min_ratio_class', 'PM_time_diff_class', 'PM_max_to_prevDayClose_ratio_class'],
    ['PM_min_to_open_ratio_class', 'AH_max_1dayago_vs_prevDayClose_class', 'PM_max_to_prevDayOpen_ratio_class', 'PM_min_to_Close_ratio_class', 'PM_min_to_prevDayOpen_ratio_class', 'PM_max_to_PM_min_ratio_class', 'prev2DayClose_to_prevDayClose_ratio_class', 'high_quad_p_rel_class', 'PM_time_diff_class', 'dayOpen_to_prevDayOpen_ratio_class', 'high_slope_rel_class', 'Close_to_open_ratio_class', 'AH_max_1dayago_vs_PM_max_class'],
    ['Close_to_prevDayHigh_class', 'dayOpen_to_prevDayOpen_ratio_class', 'PM_min_to_prevDayClose_ratio_class', 'high_slope_rel_class', 'AH_max_1dayago_to_Close_class', 'PM_max_vs_PM_max_1dayago_class', 'AH_max_1dayago_vs_PM_max_class', 'PM_max_to_prevDayOpen_ratio_class', 'day_of_week', 'dayOpen_to_prev2DayOpen_ratio_class', 'Close_to_EMA_48_class', 'PM_min_to_open_ratio_class', 'Close_to_Close_1_day_ago_class', 'Close_to_open_ratio_class', 'high_quad_p_rel_class'],
    ['AH_max_1dayago_vs_PM_max_class', 'PM_range_to_open_ratio_class', 'high_quad_p_rel_class', 'AH_max_1dayago_vs_prevDayClose_class', 'Close_to_Close_1_day_ago_class', 'PM_max_to_dayOpen_ratio_class', 'high_slope_rel_class', 'prev2DayClose_to_prevDayClose_ratio_class', 'Close_to_prevDayHigh_class', 'dayOpen_to_prevDayOpen_ratio_class', 'dayOpen_to_prevDayClose_class'],
    ['PM_max_to_prevDayClose_ratio_class', 'high_quad_q_rel_class', 'PM_min_to_prevDayOpen_ratio_class', 'AH_max_1dayago_vs_PM_max_class', 'high_quad_p_rel_class', 'PM_max_to_dayOpen_ratio_class'],
    ['prev2DayClose_to_prevDayClose_ratio_class', 'dayOpen_to_prev2DayOpen_ratio_class', 'PM_range_to_open_ratio_class', 'PM_time_diff_class', 'AH_max_1dayago_to_Close_class', 'PM_range_to_close_ratio_class', 'PM_max_to_dayOpen_ratio_class', 'PM_max_to_prevDayClose_ratio_class', 'Close_to_prevDayHigh_class', 'Open_1_day_ago_to_Close_1_day_ago_ratio_class', 'PM_min_to_open_ratio_class'],

    ['return_1d_class', 'coef_q_vol_rel_class'],#AAPL
    ['time_in_minutes', 'dayOpen_to_prevDayOpen_ratio_class'], #OKLO

    ['Open_1_day_ago_to_Close_1_day_ago_ratio_class', 'Close_to_open_ratio_class', 'PM_min_to_prevDayOpen_ratio_class', 'close_to_High10_class', 'PM_range_to_open_ratio_class'], #SHOP
    ['close_to_Low20_class', 'close_to_High20_class', 'return_1d_to_return_2d_ratio_class', 'PM_time_diff_class', 'slope_a_vol_rel_class', 'AH_max_1dayago_vs_prevDayClose_class', 'PM_max_vs_PM_max_1dayago_class', 'date_after_1124', 'PM_range_to_open_ratio_class', 'AH_max_1dayago_vs_PM_max_class'], #LLY
    ['high_quad_q_rel_class', 'PM_range_to_close_ratio_class', 'Open_1_day_ago_to_Close_1_day_ago_ratio_class', 'date_after_0125', 'ema_ratio2_class', 'coef_p_vol_rel_class', 'open_to_prev_close_class', 'PM_volume_max_class'], #V
    ['PM_min_to_prevDayClose_ratio_class', 'PM_min_to_open_ratio_class', 'date_after_0125'],#V
    ['PM_max_to_dayOpen_ratio_class', 'dayOpen_to_prevDayClose_class', 'PM_min_to_prevDayOpen_ratio_class', 'PM_max_vs_PM_max_1dayago_class', 'FD_3', 'Close_to_EMA_48_class', 'Close_class', 'AH_max_1dayago_vs_PM_max_class'],#AMZN
    ['PM_min_to_open_ratio_class', 'AH_max_1dayago_vs_prevDayClose_class', 'return_1d_to_return_2d_ratio_class'],#AMZN
    ['FD_1','FD_2','FD_3', 'time_in_minutes','STD_10_class','STD_30_class'],
    ['dayOpen_to_prevDayOpen_ratio_class', 'PM_time_diff_class', 'date_after_0924', 'high_quad_p_rel_class', 'prev2DayClose_to_prevDayClose_ratio_class', 'PM_max_vs_PM_max_1dayago_class'], #TSLA
    ['PM_max_to_dayOpen_ratio_class','AH_max_1dayago_vs_prevDayClose_class','dayOpen_to_prev2DayOpen_ratio_class','PM_min_to_Close_ratio_class','PM_time_diff_class','prev2DayClose_to_prevDayClose_ratio_class','day_of_week','PM_min_to_open_ratio_class','PM_max_to_prevDayClose_ratio_class','Close_to_open_ratio_class'],
    ['prev2DayClose_to_prevDayClose_ratio_class','dayOpen_to_prev2DayOpen_ratio_class','PM_min_to_open_ratio_class','dayOpen_to_prevDayOpen_ratio_class','AH_max_1dayago_vs_prevDayClose_class','PM_min_to_Close_ratio_class'],
    ['PM_max_to_min_ratio', 'FD_1', 'dayOpen_to_prevDayOpen_ratio_class', 'PM_max_vs_PM_max_1dayago_class', 'Close_to_prevDayHigh_class', 'PM_min_to_Close_ratio_class'],
    ]

unique_stocks_list = df['Stock'].unique().tolist()
print(f"there are {len(unique_stocks_list)} unique stocks")
best_for_stock = {}
stock_added_counter = set()
for stock in unique_stocks_list: 
    if len(stock_added_counter)>5:
        break
    print("stock : ",stock)
    available_features = []
    df_stock = df[df['Stock'] == stock].copy()

    # Calculate the fraction of missing (None/NaN) values in each column
    missing_fraction = df_stock.isna().mean()
    # Select columns that have less than 5% missing values
    good_columns = missing_fraction[missing_fraction < 0.05].index.tolist()
    available_features = set(good_columns)
    print(f"number of Columns with less than 5% missing values for {stock}: {len(good_columns)}")
    #drop None on these columns:
    df_stock = df_stock.dropna(subset=good_columns)

    init_percentage_of_1s = df_stock[TARGET].mean() * 100
    df_stock['Time'] = df_stock['Date'].dt.strftime('%H:%M')
    print("initial train end : ",train_end)
    adjusted_train_end = train_end - lookahead
    print("adjusted_train_end : ",adjusted_train_end)
    mask_train = (df_stock["Date"] >= train_start) & (df_stock["Date"] < adjusted_train_end)
    print(f"train on [{train_start}, {adjusted_train_end}[")
    mask_test  = (df_stock["Date"] >= train_end) & (df_stock["Date"] <= cutoff)
    print(f"test on [{train_end}, {cutoff}]")
    if TESTING_LENGTH_IN_DAYS>=40:
        mask_test1 = (df_stock["Date"] >= test_split1_start) & (df_stock["Date"] < test_split1_end)
        mask_test2 = (df_stock["Date"] >= test_split2_start) & (df_stock["Date"] <= test_split2_end)
        mask_test_global = (df_stock["Date"] >= test_split_global_start) & (df_stock["Date"] <= test_split_global_end)
        print(f"test1 on [{test_split1_start}, {test_split1_end}[")
        print(f"test2 on [{test_split2_start}, {test_split2_end}]")
        print(f"test_global on [{test_split_global_start}, {test_split_global_end}]")
    df_stock_train = df_stock[mask_train]
    df_stock_test  = df_stock[mask_test]
    training_length = len(df_stock_train)
    testing_length = len(df_stock_test)
    if TESTING_LENGTH_IN_DAYS>=40:
        df_stock_test1 = df_stock[mask_test1]
        df_stock_test2 = df_stock[mask_test2]
        df_stock_test_global = df_stock[mask_test_global]
    for i, subset in enumerate(feature_subsets):
        if i %12 ==0:
            print("  subset: ",i)
        valid_columns = [col for col in subset if col in df_stock_train.columns and col in available_features]
        X_train = df_stock_train[valid_columns]
        y_train = df_stock_train[TARGET]
        X_test = df_stock_test[valid_columns]
        y_test = df_stock_test[TARGET]
        if TESTING_LENGTH_IN_DAYS>=40:
            X_test_part1 = df_stock_test1[valid_columns]
            y_test_part1 = df_stock_test1[TARGET]
            X_test_part2 = df_stock_test2[valid_columns]
            y_test_part2 = df_stock_test2[TARGET]
            X_test_global = df_stock_test_global[valid_columns]
            y_test_global = df_stock_test_global[TARGET]
        #X_all = df_stock[valid_columns]
        #y_all = df_stock[TARGET]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        if TESTING_LENGTH_IN_DAYS>=40:
            X_test_scaled_firstHalf = scaler.transform(X_test_part1)
            X_test_scaled_secondHalf = scaler.transform(X_test_part2)
            X_test_scaled_global = scaler.transform(X_test_global)
        #X_all_scaled = scaler.transform(X_all)

        scaler_already_saved = False
        for model_name, model in models.items():
            model.fit(X_train_scaled, y_train) #for RF this will throw an error if X_train_scaled contains NaNs
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            if TESTING_LENGTH_IN_DAYS>=40:
                y_pred_first_half = model.predict(X_test_scaled_firstHalf)
                y_pred_second_half = model.predict(X_test_scaled_secondHalf)
                y_pred_global = model.predict(X_test_scaled_global)
                #y_pred_all = model.predict(X_all_scaled)

            #check trainign accuracy
            y_train_pred = model.predict(X_train_scaled)
            prec_training,_,spec_training,_,_ = calculate_all_metrics(y_train,y_train_pred)

            if np.all(y_pred == 0):
                #print(f"{model_name} predicts all 0")
                unique, counts = np.unique(y_test, return_counts=True)
                test_dist = dict(zip(unique, counts))
                #print(f"Actual distribution in y_test: 0s: {test_dist.get(0, 0)}, 1s: {test_dist.get(1, 0)}")
                continue
            if np.all(y_pred == 1):
                #print(f"{model_name} predicts all 1")
                continue
                
            if TESTING_LENGTH_IN_DAYS>=40:
                prec_h1,_,spec_half1,_,_ = calculate_all_metrics(y_test_part1,y_pred_first_half)
                prec_h2,_,spec_half2,_,_ = calculate_all_metrics(y_test_part2,y_pred_second_half)
                prec_global,_,spec_global,_,_ = calculate_all_metrics(y_test_global,y_pred_global)

                if any(metric is None for metric in [prec_h1, spec_half1, prec_h2, spec_half2, prec_global, spec_global]):
                    continue
                if (min(spec_half1, spec_half2) < 0.7 or
                    abs(spec_half1 - spec_half2) > 0.25 or
                    min(prec_h2, prec_global) < 0.7 or
                    abs(prec_h1 - prec_global) > 0.25 or 
                    min(prec_h2, prec_global) < 0.7):
                    # Model is not stable
                    continue

            stable_model = True
            groups_by_month = list(df_stock.groupby(df_stock['Date'].dt.to_period('M').astype(str)))
            min_precision_by_month = 999
            nbr_of_None_month_prec = 0
            std_precision_by_month = -1
            precisions_by_month = []
            for month, group in groups_by_month[1:-1]:
                X_month = group[valid_columns]
                y_month = group[TARGET]
                if X_month.empty or y_month.empty:
                    continue
                X_month_scaled = scaler.transform(X_month)
                y_pred_month = model.predict(X_month_scaled)
                precision_month, _, _, _, _ = calculate_all_metrics(y_month, y_pred_month)
                if precision_month is not None and precision_month < min_precision_by_month:
                    min_precision_by_month = precision_month
                elif precision_month is None:
                    nbr_of_None_month_prec += 1
                if precision_month is not None:
                    precisions_by_month.append(precision_month)
                if (precision_month is not None and precision_month < 0.7) or nbr_of_None_month_prec > 0.6 * min((len(groups_by_month)-2),1):
                    stable_model = False
                    break 
            if not stable_model:
                #print("model not stable")
                continue
            else:
                if precisions_by_month:
                    std_precision_by_month = np.std(precisions_by_month)

            precisions_by_time = []
            std_precision_by_time = -1
            min_precision_by_time = 999
            for time, group in df_stock.groupby('Time'):
                X_time = group[valid_columns]
                y_time = group[TARGET]
                X_time_scaled = scaler.transform(X_time)
                y_pred_time = model.predict(X_time_scaled)
                precision_time, _, _, _, _ = calculate_all_metrics(y_time, y_pred_time)
                if precision_time is not None and precision_time < 0.7:
                    stable_model = False
                    break 
                if precision_time is not None:
                    precisions_by_time.append(precision_time)
            
            if not stable_model:
                #print("model not stable")
                continue
            else:
                if precisions_by_time:
                    std_precision_by_time = np.std(precisions_by_time)
                    min_precision_by_time = min(precisions_by_time)
                else:
                    std_precision_by_time = -1  # Or some other default value
                    min_precision_by_time = 999
            #LATER IN THE CODE WE WILL SAVE THE MODEL..

            # Calculate accuracy
            if TESTING_LENGTH_IN_DAYS<=5:
                accuracy = accuracy_score(y_train, y_train_pred)
            else:
                accuracy = accuracy_score(y_test, y_pred)
            #results[model_name] = accuracy
            #y_pred = model.predict(X_test_scaled)
            
            if TESTING_LENGTH_IN_DAYS<=5:
                cm = confusion_matrix(y_train, y_train_pred)
            else:
                cm = confusion_matrix(y_test, y_pred)
            if cm.size >= 4:     
                if TESTING_LENGTH_IN_DAYS<=5:
                    #consider training metrics:
                    precision,recall,specificity,f05,mcc = calculate_all_metrics(y_train,y_train_pred)
                else:
                    precision,recall,specificity,f05,mcc = calculate_all_metrics(y_test,y_pred)
                #print(f"{model_name} - Overall Accuracy: {accuracy:.2%} | Specificity: {specificity:.2%} | Precision: {precision:.2%}, Recall: {recall:.2%} | MCC: {mcc:.2f}")
                if precision>0.75 and specificity>0.87 and recall>0 and precision<=0.92 and recall <=0.2: #and mcc>0.01
                    model_name_to_save = f'{model_name}_{stock}_{i}'
                    stock_added_counter.add(stock)
                    if stock not in best_for_stock:
                        with open(f'{folder}/{model_name_to_save}.pkl', 'wb') as model_file:
                            pickle.dump(model, model_file)
                        with open(f'{scaler_folder}/scaler_{i}_{stock}.pkl', 'wb') as scaler_file:
                            pickle.dump(scaler, scaler_file)
                        scaler_already_saved = True
                        best_for_stock[stock] = (model_name_to_save, precision, specificity,recall,i,std_precision_by_time,init_percentage_of_1s,min_precision_by_month,min_precision_by_time,std_precision_by_month,training_length,prec_training,testing_length,valid_columns)
                    else:
                        _, best_prec, best_spec ,best_recall, _,_,_,_,_,_,_,_,_,_= best_for_stock[stock]
                        # Compare using your two criteria:
                        if (calculate_score(precision, specificity, recall) > calculate_score(best_prec, best_spec, best_recall)):
                            print(f"{model_name} - Overall Accuracy: {accuracy:.2%} | Specificity: {specificity:.2%} | Precision: {precision:.2%}, Recall: {recall:.2%} | MCC: {mcc:.2f}")
                            best_for_stock[stock] = (model_name_to_save, precision, specificity,recall,i,std_precision_by_time,init_percentage_of_1s,min_precision_by_month,min_precision_by_time,std_precision_by_month,training_length,prec_training,testing_length,valid_columns)
                            with open(f'{folder}/{model_name_to_save}.pkl', 'wb') as model_file:
                                pickle.dump(model, model_file)
                            if not scaler_already_saved:
                                with open(f'{scaler_folder}/scaler_{i}_{stock}.pkl', 'wb') as scaler_file:
                                    pickle.dump(scaler, scaler_file)
                                scaler_already_saved=True

    if USE_THIS_SCRIPT_FOR_METADATA_GENERATION:
        if stock in best_for_stock:
            print(f"this stock has a model we will compute the precision on rows from cutoff {cutoff} TO CUTOFF+6days {cutoff+pd.Timedelta(days=6)}")
            #test the model on original_df after cutoff
            model_name, _, _,_,features_id,_,_,_,_,_,_,_,_,features_for_prediction = best_for_stock[stock]
            with open(f'{folder}/{model_name}.pkl', 'rb') as model_file:
                model = pickle.load(model_file)
            with open(f'{scaler_folder}/scaler_{features_id}_{stock}.pkl', 'rb') as scaler_file:
                scaler = pickle.load(scaler_file)
            print(f"we will use the model {model_name} and scaler : scaler_{features_id}_{stock}.pkl to predict")
            #rows of original_df after cutoff for that stock
            df_stock = original_df[original_df['Stock'] == stock].copy()
            #extract dates between cutoff and cutoff+6days
            df_stock = df_stock[df_stock['Date'] > cutoff]
            df_stock = df_stock[df_stock['Date'] <= cutoff+pd.Timedelta(days=6)]
            
            print(f"backtest on ]{cutoff}, {cutoff+pd.Timedelta(days=6)}]")
            y_true = df_stock[TARGET]
            X = df_stock[features_for_prediction]
            X.to_csv(f"debug_{stock}_X_pre_scaled.csv", index=True)
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)

            temp_df_stock = df_stock.copy()
            temp_df_stock['Predicted_Target'] = y_pred
            temp_df_stock[['Stock', 'Date', 'Close', TARGET, 'Predicted_Target']].to_csv(f"debug_{stock}_from_train2.csv", index=True)

            cm = confusion_matrix(y_true, y_pred)
            if cm.size>=4:
                TN, FP, FN, TP = cm.ravel()
                precision = TP / (TP + FP) if (TP + FP) != 0 else 0
                print("Confusion Matrix:")
                print(cm)
                print("\nDetailed Metrics:")
                print(f"True Positives (TP): {TP}")
                print(f"False Positives (FP): {FP}")
                print(f"True Negatives (TN): {TN}")
                print(f"False Negatives (FN): {FN}")
                print(f"\nPrecision = TP / (TP + FP) = {TP} / ({TP} + {FP}) = {precision:.4f}")

            precision,recall,specificity,f05,mcc = calculate_all_metrics(y_true,y_pred)
            if precision is not None and precision>=0.75:
                # i want to add the preoperty "good_model" = 1 to best_for_stock[stock]
                best_for_stock[stock] = best_for_stock[stock] + (1,precision,) #TO KEEP THE DATA AS A TUPLE
            elif precision is None:
                best_for_stock[stock] = best_for_stock[stock] + (0,-1,)
            else:
                best_for_stock[stock] = best_for_stock[stock] + (-1,precision,)
            #print(f"precision on rows from cutoff {cutoff} TO CUTOFF+6days {cutoff+pd.Timedelta(days=6)} : {precision:.2%} | Specificity: {specificity:.2%} | Recall: {recall:.2%} | MCC: {mcc:.2f}")
            


output_data = {}

if USE_THIS_SCRIPT_FOR_METADATA_GENERATION:
    for stock, (best_model, best_prec, best_spec,best_recall,features_id, std_precision_by_time,init_percentage_of_1s,min_precision_by_month,min_precision_by_time, std_precision_by_month,training_length,prec_training,testing_length,subset,good,live_precision) in best_for_stock.items():
        output_data[stock] = {
            "best_model": best_model,
            "precision": best_prec,     # raw float value (e.g., 0.95)
            "specificity": best_spec,   # raw float value (e.g., 0.90)
            "recall": best_recall,
            "subset_id": features_id,
            "testing_length_in_days": TESTING_LENGTH_IN_DAYS,
            "std_precision_by_time": std_precision_by_time,
            "init_percentage_of_1s": init_percentage_of_1s,
            "min_precision_by_month": min_precision_by_month,
            "min_precision_by_time": min_precision_by_time,
            "std_precision_by_month": std_precision_by_month,
            "training_length": training_length,
            "training_precision": prec_training,
            "testing_length": testing_length,
            "file_created_after_20_03": 1,
            "good_model": good,
            "precision_after_cutoff":live_precision,
            "cutoff_date": cutoff.strftime("%Y-%m-%d %H:%M:%S"),
            "lookahead_in_hours": LOOKAHEAD_NUMBER,
            "subset": subset         # a list that is already JSON serializable
        }
else:
    for stock, (best_model, best_prec, best_spec,best_recall,features_id, std_precision_by_time,init_percentage_of_1s,min_precision_by_month,min_precision_by_time, std_precision_by_month,training_length,prec_training,testing_length,subset) in best_for_stock.items():
        output_data[stock] = {
            "best_model": best_model,
            "precision": best_prec,     # raw float value (e.g., 0.95)
            "specificity": best_spec,   # raw float value (e.g., 0.90)
            "recall": best_recall,
            "subset_id": features_id,
            "testing_length_in_days": TESTING_LENGTH_IN_DAYS,
            "std_precision_by_time": std_precision_by_time,
            "init_percentage_of_1s": init_percentage_of_1s,
            "min_precision_by_month": min_precision_by_month,
            "min_precision_by_time": min_precision_by_time,
            "std_precision_by_month": std_precision_by_month,
            "training_length": training_length,
            "training_precision": prec_training,
            "testing_length": testing_length,
            "file_created_after_20_03": 1,
            "lookahead_in_hours": LOOKAHEAD_NUMBER,
            "subset": subset          # a list that is already JSON serializable
        }
with open("models_to_use.json", "w") as f:
    json.dump(output_data, f, indent=4, default=str)
print("Data saved to models_to_use.json")

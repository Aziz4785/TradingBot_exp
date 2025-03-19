
import pandas as pd
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
TESTING_LENGTH_IN_DAYS = 30
TRAINING_STARTING_LAG = 300
cleanup_files(['models_to_use.json','model_output.csv','debugging_from_backtest.csv'])
cleanup_folders(['allmodels','allscalers'])
folder = f'allmodels'
os.makedirs(folder, exist_ok=True)
scaler_folder = f'allscalers'
os.makedirs(scaler_folder, exist_ok=True)

df = pd.read_csv('clean.csv')
df = df.sample(frac=1).reset_index(drop=True)
df.drop_duplicates(inplace=True)
df = df.dropna()
#df = df[df['Stock'].isin(['BIO', 'TEAM','LW'])]

df["Date"] = pd.to_datetime(df["Date"])
today = pd.to_datetime("today").normalize()

#all_data = all_data.dropna()


train_start = today - pd.Timedelta(days=TRAINING_STARTING_LAG)
train_end   = today - pd.Timedelta(days=TESTING_LENGTH_IN_DAYS) 

# Define the two test splits:
if TESTING_LENGTH_IN_DAYS>=40:
    test_split1_start = train_end  
    test_split1_end   = today - pd.Timedelta(days=int(TESTING_LENGTH_IN_DAYS/2))  

    test_split2_start = test_split1_end
    test_split2_end   = today       

    test_split_global_start = today - pd.Timedelta(days=100) 
    test_split_global_end = today
models = {
    'DT1': DecisionTreeClassifier(), 
    #'DT2': DecisionTreeClassifier(max_depth=10, min_samples_split=4), this never give good results
    'DT3': DecisionTreeClassifier(max_depth=15, min_samples_leaf=5, min_samples_split=10),
    'DT4': DecisionTreeClassifier(max_depth=20, criterion='entropy', random_state=42),
    'DT5': DecisionTreeClassifier(min_samples_leaf=3, max_features='sqrt', max_depth=6),
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

    'XGB1': XGBClassifier(n_estimators=100),
    #'XGB3':XGBClassifier(n_estimators=300,colsample_bytree= 0.9,gamma=0.1,learning_rate=0.1,max_depth=7,min_child_weight=1,subsample=0.7),
    'XGB7':XGBClassifier(n_estimators=300,colsample_bytree= 0.8,gamma=0.1,learning_rate=0.1,max_depth=5,min_child_weight=1,subsample=0.9),
    'XGB9':XGBClassifier(n_estimators=300,colsample_bytree= 0.8,gamma=0,learning_rate=0.1,max_depth=7,min_child_weight=1,subsample=0.7),
    
    #'XGB16':XGBClassifier(n_estimators=300,colsample_bytree=  0.9, gamma= 0, learning_rate=0.3,max_depth=8,min_child_weight=1,subsample=0.8),
    'XGB17':XGBClassifier(n_estimators=500,learning_rate=0.05,max_depth=3,min_child_weight=3,subsample=0.8,colsample_bytree=0.7,gamma=1.0,reg_alpha=0.1,reg_lambda=1.0),
    'XGB18': XGBClassifier(n_estimators=400,learning_rate=0.05,max_depth=6,min_child_weight=2,subsample=0.85,colsample_bytree=0.85,gamma=0.2),
    'XGB19':XGBClassifier(n_estimators=300, colsample_bytree=0.8, gamma=0, learning_rate=0.1, max_depth=7, min_child_weight=1, subsample=0.7, scale_pos_weight=3.0),
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
    ['PM_min_to_prevDayClose_ratio_class', 'high_quad_q_rel_class'],
    ['PM_max_vs_PM_max_1dayago_class', 'AH_max_1dayago_vs_prevDayClose_class', 'return_1d_to_return_2d_ratio_class'],
    ['hist_close_ratio_class', 'high_slope_rel_class', 'open_to_prev_close_class'],
    ['PM_max_to_prevDayClose_ratio_class', 'high_quad_q_rel_class', 'PM_min_to_prevDayOpen_ratio_class', 'AH_max_1dayago_vs_PM_max_class', 'high_quad_p_rel_class', 'PM_max_to_dayOpen_ratio_class'],
    ['prev2DayClose_to_prevDayClose_ratio_class', 'dayOpen_to_prev2DayOpen_ratio_class', 'PM_range_to_open_ratio_class', 'PM_time_diff_class', 'AH_max_1dayago_to_Close_class', 'PM_range_to_close_ratio_class', 'PM_max_to_dayOpen_ratio_class', 'PM_max_to_prevDayClose_ratio_class', 'Close_to_prevDayHigh_class', 'Open_1_day_ago_to_Close_1_day_ago_ratio_class', 'PM_min_to_open_ratio_class'],

    ['PM_max_to_prevDayOpen_ratio_class', 'PM_min_to_prevDayOpen_ratio_class', 'PM_min_to_Close_ratio_class', 'high_quad_p_rel_class'],

    ['return_1d_class', 'coef_q_vol_rel_class'],#AAPL
    ['time_in_minutes', 'dayOpen_to_prevDayOpen_ratio_class'], #OKLO
    ['date_after_0125', 'coef_q_vol_rel_class', 'return_2d_class'], #OKLO

    ['Open_1_day_ago_to_Close_1_day_ago_ratio_class', 'Close_to_open_ratio_class', 'PM_min_to_prevDayOpen_ratio_class', 'close_to_High10_class', 'PM_range_to_open_ratio_class'], #SHOP
    ['date_after_1124', 'open_to_prev_close_class', 'high_quad_q_rel_class', 'PM_min_to_open_ratio_class', 'AH_max_1dayago_vs_PM_max_class'], #SNOW
    ['close_to_Low10_class', 'prev2DayClose_to_prevDayClose_ratio_class'],
    ['close_to_Low20_class', 'close_to_High20_class', 'return_1d_to_return_2d_ratio_class', 'PM_time_diff_class', 'slope_a_vol_rel_class', 'AH_max_1dayago_vs_prevDayClose_class', 'PM_max_vs_PM_max_1dayago_class', 'date_after_1124', 'PM_range_to_open_ratio_class', 'AH_max_1dayago_vs_PM_max_class'], #LLY
    ['high_quad_q_rel_class', 'PM_range_to_close_ratio_class', 'Open_1_day_ago_to_Close_1_day_ago_ratio_class', 'date_after_0125', 'ema_ratio2_class', 'coef_p_vol_rel_class', 'open_to_prev_close_class', 'PM_volume_max_class'], #V
    ['PM_min_to_prevDayClose_ratio_class', 'PM_min_to_open_ratio_class', 'date_after_0125'],#V
    ['PM_max_to_dayOpen_ratio_class', 'dayOpen_to_prevDayClose_class', 'PM_min_to_prevDayOpen_ratio_class', 'PM_max_vs_PM_max_1dayago_class', 'FD_3', 'Close_to_EMA_48_class', 'Close_class', 'AH_max_1dayago_vs_PM_max_class'],#AMZN
    ['PM_min_to_open_ratio_class', 'AH_max_1dayago_vs_prevDayClose_class', 'return_1d_to_return_2d_ratio_class'],#AMZN
    ['FD_1','FD_2','FD_3', 'time_in_minutes','STD_10_class','STD_30_class'],
    ['AH_max_1dayago_vs_PM_max_class', 'return_1d_class'],#TSLA
    ['dayOpen_to_prevDayOpen_ratio_class', 'PM_time_diff_class', 'date_after_0924', 'high_quad_p_rel_class', 'prev2DayClose_to_prevDayClose_ratio_class', 'PM_max_vs_PM_max_1dayago_class'], #TSLA
    ['open_to_prev_close_class', 'time_in_minutes'],
    ['PM_max_to_dayOpen_ratio_class','AH_max_1dayago_vs_prevDayClose_class','dayOpen_to_prev2DayOpen_ratio_class','PM_min_to_Close_ratio_class','PM_time_diff_class','prev2DayClose_to_prevDayClose_ratio_class','day_of_week','PM_min_to_open_ratio_class','PM_max_to_prevDayClose_ratio_class','Close_to_open_ratio_class'],
    ['prev2DayClose_to_prevDayClose_ratio_class','dayOpen_to_prev2DayOpen_ratio_class','PM_min_to_open_ratio_class','dayOpen_to_prevDayOpen_ratio_class','AH_max_1dayago_vs_prevDayClose_class','PM_min_to_Close_ratio_class'],
    ]

unique_stocks_list = df['Stock'].unique().tolist()
print(f"there are {len(unique_stocks_list)} unique stocks")
best_for_stock = {}

for stock in unique_stocks_list: 
    print("stock : ",stock)
    available_features = []
    df_stock = df[df['Stock'] == stock].copy()

    # Calculate the fraction of missing (None/NaN) values in each column
    missing_fraction = df_stock.isna().mean()
    # Select columns that have less than 5% missing values
    good_columns = missing_fraction[missing_fraction < 0.05].index.tolist()
    available_features = set(good_columns)
    print(f"number of Columns with less than 5% missing values for {stock}: {len(good_columns)}")

    init_percentage_of_1s = df_stock[TARGET].mean() * 100
    df_stock['Time'] = df_stock['Date'].dt.strftime('%H:%M')
    mask_train = (df_stock["Date"] >= train_start) & (df_stock["Date"] < train_end)
    mask_test  = (df_stock["Date"] >= train_end)   & (df_stock["Date"] <= today)
    if TESTING_LENGTH_IN_DAYS>=40:
        mask_test1 = (df_stock["Date"] >= test_split1_start) & (df_stock["Date"] < test_split1_end)
        mask_test2 = (df_stock["Date"] >= test_split2_start) & (df_stock["Date"] <= test_split2_end)
        mask_test_global = (df_stock["Date"] >= test_split_global_start) & (df_stock["Date"] <= test_split_global_end)

    df_stock_train = df_stock[mask_train] #TODO : try to balance this train df 
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
            #print("    model : ",model_name)
            # Train the model
            model.fit(X_train_scaled, y_train)
            
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
                if (min(spec_half1, spec_half2) < 0.75 or
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
            accuracy = accuracy_score(y_test, y_pred)
            #results[model_name] = accuracy
            y_pred = model.predict(X_test_scaled)
            y_pred = model.predict(X_test_scaled)
            
            cm = confusion_matrix(y_test, y_pred)
            if cm.size >= 4:     
                precision,recall,specificity,f05,mcc = calculate_all_metrics(y_test,y_pred)
                
                #print(f"{model_name} - Overall Accuracy: {accuracy:.2%} | Specificity: {specificity:.2%} | Precision: {precision:.2%}, Recall: {recall:.2%} | MCC: {mcc:.2f}")
                if precision>0.78 and specificity>0.87 and recall>0.01: #and mcc>0.01
                    model_name_to_save = f'{model_name}_{stock}_{i}'
                    
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

output_data = {}
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
        "subset": subset          # a list that is already JSON serializable
    }
with open("models_to_use.json", "w") as f:
    json.dump(output_data, f, indent=4)
print("Data saved to models_to_use.json")

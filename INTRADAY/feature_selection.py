from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,HistGradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from xgboost import XGBClassifier
import json
from sklearn.preprocessing import LabelEncoder
from utils import * 

model = XGBClassifier(n_estimators=300, colsample_bytree=0.8, gamma=0, learning_rate=0.1, max_depth=7, min_child_weight=1, subsample=0.7, scale_pos_weight=3.0)
TARGET_COLUMN = 'to_buy_1d'
USE_STOCK_COLUMN = False
BALANCE_DATA = False

df = pd.read_csv('clean.csv')
df = df.sample(frac=1).reset_index(drop=True)
df.drop_duplicates(inplace=True)
df = df.dropna()
df = df[df['Stock']=='PG']
features_to_exclude = ['PM_time_diff_class']
df = df.drop(columns=features_to_exclude)
random.seed(9)

def try_multiple_feature_selection(X_train_scaled, X_test_scaled, y_train, y_test, 
                                 model, k, feature_names,X_train_scaled_df,X_test_scaled_df): #k is the number of features to be selected
    results = {}
    
    # 1. Statistical Methods - ANOVA F-value
    selector_f = SelectKBest(score_func=f_classif, k=k)
    X_train_f = selector_f.fit_transform(X_train_scaled, y_train)
    X_test_f = selector_f.transform(X_test_scaled)
    selected_features_f = feature_names[selector_f.get_support()]
    
    model.fit(X_train_f, y_train)
    y_pred_f = model.predict(X_test_f)
    results['f_score'] = {
        'accuracy': accuracy_score(y_test, y_pred_f),
        'features': selected_features_f
    }
    
    # 2. Mutual Information
    selector_mi = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_mi = selector_mi.fit_transform(X_train_scaled, y_train)
    X_test_mi = selector_mi.transform(X_test_scaled)
    selected_features_mi = feature_names[selector_mi.get_support()]
    
    model.fit(X_train_mi, y_train)
    y_pred_mi = model.predict(X_test_mi)
    results['mutual_info'] = {
        'accuracy': accuracy_score(y_test, y_pred_mi),
        'features': selected_features_mi
    }
    
    # 3. L1-based Feature Selection
    selector_l1 = SelectFromModel(estimator=LogisticRegression(penalty='l1', 
                                solver='liblinear', C=0.1),
                                max_features=k)
    X_train_l1 = selector_l1.fit_transform(X_train_scaled, y_train)
    X_test_l1 = selector_l1.transform(X_test_scaled)
    selected_features_l1 = feature_names[selector_l1.get_support()]
    
    model.fit(X_train_l1, y_train)
    y_pred_l1 = model.predict(X_test_l1)
    results['l1'] = {
        'accuracy': accuracy_score(y_test, y_pred_l1),
        'features': selected_features_l1
    }
    
    # 4. Tree-based Feature Selection
    selector_tree = SelectFromModel(estimator=ExtraTreesClassifier(),
                                  max_features=k)
    X_train_tree = selector_tree.fit_transform(X_train_scaled, y_train)
    X_test_tree = selector_tree.transform(X_test_scaled)
    selected_features_tree = feature_names[selector_tree.get_support()]
    
    model.fit(X_train_tree, y_train)
    y_pred_tree = model.predict(X_test_tree)
    results['tree_based'] = {
        'accuracy': accuracy_score(y_test, y_pred_tree),
        'features': selected_features_tree
    }
    
    #5. random method
    best_accuracy = 0
    best_features = None
    best_precision = 0
    best_features_precision = None
    for i in range(150): #PUT 120
        if i%30==0:
            print(f"{i} -> {best_accuracy}")
        selected_features = random.sample(list(feature_names), k)
        X_selected_train = X_train_scaled_df[selected_features]
        X_selected_test = X_test_scaled_df[selected_features]
        #X_selected = X_train_scaled[selected_features]
        model.fit(X_selected_train, y_train)
        y_pred = model.predict(X_selected_test)
        #accuracy = accuracy_score(y_test, y_pred)
        fbeta = fbeta_score(y_test, y_pred, average='macro', beta=0.25)
        precision = precision_score(y_test, y_pred, zero_division=0)
        if fbeta > best_accuracy:
            best_accuracy = fbeta
            best_features = selected_features
        if precision >best_precision :
            best_precision = precision
            best_features_precision = selected_features
    results['random_method'] = {
        'fbeta': best_accuracy,
        'features': best_features,
        'precision': best_precision,
        'features_precision' : best_features_precision
    }
    return results

if BALANCE_DATA:
    df = balance_binary_target(df, TARGET_COLUMN)
percentages = df[['to_buy_1d']].mean() * 100
print(percentages.round(2).to_string()) 
if USE_STOCK_COLUMN:
    encoder = LabelEncoder()
    # Fit the encoder to the unique stock symbols and transform
    df['Stock_encoded'] = encoder.fit_transform(df['Stock'])


drop_columns = ['Stock', 'Date', 'Close', TARGET_COLUMN, 
                'to_buy_1d', 'to_buy_2d', 'to_buy_1d31', 
                'to_buy_intraday', "PM_max", "PM_min"]
X = df.drop(columns=drop_columns, errors='ignore')
# y = df[TARGET_COLUMN]
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.25, random_state=42, stratify=y
# )

df["Date"] = pd.to_datetime(df["Date"])
today = pd.to_datetime("today").normalize()
train_start = today - pd.Timedelta(days=300) #put 300
train_end   = today - pd.Timedelta(days=50) 
mask_train = (df["Date"] >= train_start) & (df["Date"] < train_end)
mask_test  = (df["Date"] >= train_end)   & (df["Date"] <= today)
df_train = df[mask_train]
df_test = df[mask_test]
X_train = df_train.drop(columns=drop_columns, errors='ignore')
y_train = df_train[TARGET_COLUMN]
X_test = df_test.drop(columns=drop_columns, errors='ignore')
y_test = df_test[TARGET_COLUMN]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, 
                                 columns=X_train.columns, 
                                 index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, 
                                columns=X_test.columns, 
                                index=X_test.index)


# Usage example:
for k in range(2, 14, 1):
    print(f"-----number of features: {k}")
    results = try_multiple_feature_selection(X_train_scaled, X_test_scaled, 
                                              y_train, y_test, model, k, 
                                              X.columns,X_train_scaled_df,X_test_scaled_df)

    for method, result in results.items():
        print(f"\n{method.upper()} Selection:")
        if 'accuracy' in result:
            print(f"Accuracy: {result['accuracy']:.4f}")
        elif 'fbeta' in result:
            print(f"fbeta: {result['fbeta']:.4f}")
        print("Selected Features:")
        print(result['features'])
        if 'precision' in result:
            print(f"Precision: {result['precision']:.4f}")
            print("Selected Features for precision:")
            print(result['features_precision'])


#with open('features.json', 'w') as file:
    #json.dump(data, file)


"""
fbeta: 0.5716
Selected Features:
['return_1d_to_return_2d_ratio_class', 'mom_5_class', 'Open_1_day_ago_to_Close_1_day_ago_ratio_class', 'vol_50_class', 'PM_max_vs_PM_max_1dayago_class', 'date_after_0924', 'Close_class', 'PM_range_to_close_ratio_class']

fbeta: 0.6656
Selected Features:
['open_to_prev_close_class', 'time_in_minutes']

fbeta: 0.6619
Selected Features:
['date_after_0125', 'dayOpen_to_prevDayClose_class', 'PM_max_to_prevDayOpen_ratio_class', 'PM_max_vs_PM_max_1dayago_class']
"""
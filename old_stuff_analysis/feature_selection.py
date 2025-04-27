from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.base import clone
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
from sklearn.metrics import precision_score, recall_score, fbeta_score, matthews_corrcoef, confusion_matrix, accuracy_score, precision_recall_curve


model = RandomForestClassifier(n_estimators=300,max_depth=10, min_samples_leaf=5,class_weight="balanced",max_features="sqrt")
TARGET_COLUMN = 'good_model'

df = pd.read_csv("old_stuff_analysis/balanced_targets_with0good.csv")
df.drop(columns=["Ticker"], inplace=True, errors='ignore')
df = df.sample(frac=1).reset_index(drop=True)
df.drop_duplicates(inplace=True)
df = df.dropna(subset=[TARGET_COLUMN])
df.loc[df[TARGET_COLUMN] == 0, TARGET_COLUMN] = 1
df.loc[df[TARGET_COLUMN] == -1, TARGET_COLUMN] = 0
threshold_percentage = 0.90
total_rows = len(df)
threshold_count = total_rows * threshold_percentage
columns_to_remove = []
for col in df.columns:
    counts = df[col].value_counts(dropna=False)
    if not counts.empty:
        most_frequent_count = counts.iloc[0]
        dominant_value = counts.index[0]
        if most_frequent_count >= threshold_count:
            # Optional: Handle NaN display nicely
            display_value = 'NaN' if pd.isna(dominant_value) else dominant_value
            percentage = (most_frequent_count / total_rows) * 100
            print(f"Column '{col}' has '{display_value}' for {percentage:.1f}% of rows ({most_frequent_count}/{total_rows}).")
            columns_to_remove.append(col)
df.drop(columns=columns_to_remove, inplace=True)
print(f"Removed columns: {columns_to_remove}")


def try_multiple_feature_selection(X_train_scaled, X_test_scaled, y_train, y_test, 
                                 model, k, feature_names,X_train_scaled_df,X_test_scaled_df): #k is the number of features to be selected
    results = {}

    #5. random method
    best = dict(
        accuracy=-np.inf, fbeta=-np.inf, precision=-np.inf,
        features_accuracy=None, features_fbeta=None, features_precision=None
    )
    for i in range(600): 
        selected_features = random.sample(list(feature_names), k) #uncomment if you don't want to force include
        X_selected_train = X_train_scaled_df[selected_features]
        X_selected_test = X_test_scaled_df[selected_features]
        #X_selected = X_train_scaled[selected_features]
        #model.fit(X_selected_train, y_train)
        clf = clone(model)
        clf.fit(X_selected_train, y_train)
        y_pred = clf.predict(X_selected_test)

        #y_pred = model.predict(X_selected_test)
        acc  = accuracy_score(y_test, y_pred)
        fb   = fbeta_score(y_test, y_pred, beta=0.15, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)

        if acc  > best['accuracy']:
            best['accuracy'], best['features_accuracy'] = acc, selected_features
        if fb   > best['fbeta']:
            best['fbeta'],    best['features_fbeta']    = fb, selected_features
        if prec > best['precision']:
            best['precision'], best['features_precision'] = prec, selected_features
        if (i+1) % 150 == 0:
            print(f"Iter {i+1} Best acc so far: {best['accuracy']:.4f}")

    results['random_method'] = best
    return results




X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_df = pd.DataFrame(X_train, columns=X.columns, index=X_train.index)
X_test_df = pd.DataFrame(X_test, columns=X.columns, index=X_test.index)
for k in range(2, 20, 1):
    print(f"-----number of features: {k}")
    res  = try_multiple_feature_selection(
        X_train_df, X_test_df, 
        y_train, y_test, 
        model, k, 
        X_train_df.columns, 
        X_train_df, X_test_df
    )

    for method, r in res.items():
        print(f"\n{method.upper()}")
        #print(f"  Acc.      : {r['accuracy']:.4f}")
        if 'fbeta' in r:
            print(f"  Fβ (0.15) : {r['fbeta']:.4f}")
            print("  Features  :", list(r['features_fbeta']))
        if 'precision' in r:
            print(f"  Precision : {r['precision']:.4f}")
            print("  Features  :", list(r['features_precision']))
        if 'accuracy' in r:
            print(f"  Accuracy  : {r['accuracy']:.4f}")
            print("  Features  :", list(r['features_accuracy']))


#with open('features.json', 'w') as file:
    #json.dump(data, file)


"""
fbeta: 0.8479
Selected Features:
['day_of_week', 'dayOpen_to_prevDayOpen_ratio_class']"""
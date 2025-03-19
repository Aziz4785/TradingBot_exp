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
from SuperModel import * 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


df = pd.read_csv('C:/Users/aziz8/Documents/tradingBot/clean.csv')
df = df.sample(frac=1).reset_index(drop=True)
df.drop_duplicates(inplace=True)
df = df.dropna()
MAIN_TICKER= 'ORCL'
TARGET_COLUMN = 'to_buy_1d'
df = df[df['Stock']==MAIN_TICKER]
df['Date'] = pd.to_datetime(df['Date'])
df['era'] = df['Date'].dt.date 
models_dir  = "allmodels"
scalers_dir = "allscalers"
config_path = "models_to_use.json"
with open(config_path, 'r') as file:
    config = json.load(file)

feature_set = extract_features_of_stock(config,MAIN_TICKER)
print(feature_set)
loaded_bin_dict_json = load_bins("bins_json.json")
super_model = SuperModel(config_path, models_dir, scalers_dir)
probabilities = super_model.predict_proba(MAIN_TICKER, df[feature_set])
# probabilities is an array of shape (num_samples, n_classes).
# For a binary classification model, probabilities[:, 1] typically gives the "positive" class probability.
df["prediction_prob"] = probabilities[:, 1]
"""
# Compute per-era mean of target and predictions
per_era_scores = df.groupby('era').apply(
    lambda x: x[['prediction_prob']].mean() - x[TARGET_COLUMN].mean()
)
print("per_era_scores : ")
print(per_era_scores )
# Plot results
per_era_scores.cumsum().plot(title="Cumulative Performance", kind="line", figsize=(8, 4))
plt.show()"""


# Or compute FPR directly
def false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0

per_era_fpr = df.groupby('era').apply(
    lambda x: false_positive_rate(x[TARGET_COLUMN], (x['prediction_prob'] > 0.5).astype(int)) 
    if len(x[TARGET_COLUMN].unique()) > 1 else None
)
plt.figure(figsize=(10,4))
per_era_fpr.plot(kind="bar", title="Per-Era AUC")
plt.tight_layout()  # Adjust layout to prevent text cutoff
plt.show()


per_era_auc = df.groupby('era').apply(
    lambda x: roc_auc_score(x[TARGET_COLUMN], x['prediction_prob']) if len(x[TARGET_COLUMN].unique()) > 1 else None
)

plt.figure(figsize=(10,4))
per_era_auc.plot(kind="bar", title="Per-Era AUC")
plt.tight_layout()  # Adjust layout to prevent text cutoff
plt.show()

score_series = 2 * (per_era_auc - 0.5)  # transforms AUC=0.5 to 0, AUC>0.5 to positive
cumulative_score = score_series.cumsum()

cumulative_score.plot(kind="line", title="Cumulative Score")
plt.show()
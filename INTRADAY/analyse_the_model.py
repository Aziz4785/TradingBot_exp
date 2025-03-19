

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
from sklearn.metrics import precision_score, recall_score, fbeta_score, matthews_corrcoef, confusion_matrix, accuracy_score, precision_recall_curve
from utils import *

#run this file to see the detailled steps for specificity and precision calculation
TARGET = 'to_buy_1d'
model_path = 'old_stuff4/allmodels/DT3_TSN_20.pkl'
config_path = "old_stuff4/models_to_use.json"
with open(model_path, 'rb') as file:
    model = pickle.load(file)
with open(config_path, 'r') as file:
    config = json.load(file)
scaler_path = 'old_stuff4/allscalers/scaler_20_TSN.pkl'
with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

df = pd.read_csv('old_stuff4/clean.csv')
df = df.sample(frac=1).reset_index(drop=True)
df.drop_duplicates(inplace=True)
df = df.dropna()
df = df[df['Stock'].isin(['TSN'])]
df["Date"] = pd.to_datetime(df["Date"])
today = pd.to_datetime("today").normalize()
start_analysis   = today - pd.Timedelta(days=55) 
end_analysis = today
mask_test  = (df["Date"] >= start_analysis)   & (df["Date"] <= end_analysis)

df_test = df.loc[mask_test].copy()
feature_columns =extract_features_of_stock(config,'TSN')
# Ensure your test set has the correct features
X_test = df_test[feature_columns]
y_true = df_test[TARGET]
X_test_scaled = scaler.transform(X_test)
# Generate predictions for each row
# (Assuming your trained model is stored in a variable named `model`)
y_pred = model.predict(X_test_scaled)
df_test['predicted_to_buy'] = y_pred
# If your model supports probabilities and you want more insight,
# you can get the prediction probability for the positive class (e.g., to_buy_1d == 1)
if hasattr(model, "predict_proba"):
    df_test['prediction_probability'] = model.predict_proba(X_test_scaled)[:, 1]

# Now df_test contains the original data plus your prediction details.
# You can view the detailed predictions for each row:
model_output = df_test[['Date']+feature_columns+[TARGET]+['predicted_to_buy']].sort_values(by='Date')
print(model_output)
model_output.to_csv('model_output_TSN.csv', encoding='utf-8', index=False)
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# For binary classification, the confusion matrix layout is:
#        Predicted: 0     Predicted: 1
# Actual: 0      TN              FP
# Actual: 1      FN              TP
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    print("\nDetailed Counts:")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP): {tp}")

    print(" specifiticity = tn / (tn + fp)")
    print(f" =  {tn} / ({tn} + {fp}) = {tn / (tn + fp)}")
    # Calculate Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Print step-by-step calculation
    print("\nDetailed Calculation:")
    print(f"Precision = TP / (TP + FP) = {tp} / ({tp} + {fp}) = {precision:.4f}")
    print(f"Recall    = TP / (TP + FN) = {tp} / ({tp} + {fn}) = {recall:.4f}")
else:
    print("The confusion matrix does not have a binary shape. Check the labels.")

# Additionally, you can print a full classification report:
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# -------------------------------
# 6. (Optional) Inspect Predictions with Details
# -------------------------------
# You can also display the test DataFrame with prediction details:
print("\nTest Data with Predictions:")
print(df_test[['Date', TARGET, 'predicted_to_buy', 'prediction_probability']].head())
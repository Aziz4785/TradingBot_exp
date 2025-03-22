import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, fbeta_score, matthews_corrcoef
from utils import *
from SuperModel import * 
import textwrap
from sklearn.metrics import roc_curve, roc_auc_score
#just run this file and see the charts one by one

models_dir  = "old_stuff12/allmodels"
scalers_dir = "old_stuff12/allscalers"
config_path = "old_stuff12/models_to_use.json"
clean_path = 'old_stuff12/clean.csv'
with open(config_path, 'r') as file:
    config = json.load(file)
stocks_from_json= get_stock_symbols_from_json(config)

df = pd.read_csv(clean_path)
df = df.sample(frac=1).reset_index(drop=True)
df.drop_duplicates(inplace=True)
df = df.dropna()
df = df[df['Stock'].isin(stocks_from_json)]
df['Date'] = pd.to_datetime(df['Date'])

USE_ONLY_TEST_DATA=False
stocks = df['Stock'].unique()
today = pd.to_datetime("today").normalize()
test_start   = today - pd.Timedelta(days=45)
test_end = today
if USE_ONLY_TEST_DATA:
    df= df[(df["Date"] >= test_start) & (df["Date"] < test_end)]

super_model = SuperModel(config_path, models_dir, scalers_dir)
MAIN_TICKER = 'LLY'
#feature_set = extract_features_of_stock(config,MAIN_TICKER)
#probabilities = super_model.predict_proba(MAIN_TICKER, df[feature_set])


#X_test = df[feature_set]
#y_true = df['to_buy_1d']
y_pred_from_model = super_model.predict_multiple(df)
df["prediction"] = y_pred_from_model
df["prediction_prob"] = super_model.predict_proba_multiple(df).apply(lambda x: x[1])
#df[df['Stock']=='JNJ'][['Date','to_buy_1d','prediction']].to_csv(f"debugging_JNJ_.csv", index=True)
# Ensure the 'Date' column is datetime type
df['Time'] = df['Date'].dt.strftime('%H:%M')

results = []

#VISUALIZE INITIAL 1s ratio :
# Assuming df and stocks are already defined
stock_names_ = []
percentages_ = []
for stock in stocks:
    df_stock = df[df['Stock'] == stock].copy()
    init_percentage_of_1s = df_stock['to_buy_1d'].mean() * 100
    stock_names_.append(stock)
    percentages_.append(init_percentage_of_1s)
    print(f"{stock}: \"init_percentage_of_1s\": {init_percentage_of_1s:.2f}, %")
plt.figure(figsize=(12, 6))
bars = plt.bar(stock_names_, percentages_)
# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.2f}%', ha='center', va='bottom')
plt.title('Initial Percentage of Buy Signals (1s) by Stock')
plt.xlabel('Stock')
plt.ylabel('Percentage of Buy Signals (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
# Add a horizontal line for average
avg = np.mean(percentages_)
plt.axhline(y=avg, color='r', linestyle='--', label=f'Average: {avg:.2f}%')
plt.legend()
plt.show()

#VISUALIZE PRECISION BY TIME OF DAY:
results = []
plt.figure(figsize=(10, 6))
best_stocks_by_time=[]
for stock in stocks:
    df_stock = df[df['Stock'] == stock].copy()  # use copy to avoid SettingWithCopyWarning
    results = []
    for time, group in df_stock.groupby('Time'):
        # Calculate metrics; only using precision here (adjust calculate_all_metrics as needed)
        precision, _, _, _, _ = calculate_all_metrics(group['to_buy_1d'], group['prediction'])
        results.append({'Time': time, 'Precision': precision})
    results_df = pd.DataFrame(results)
    min_precision = results_df['Precision'].min()
    print(f"for stock {stock} the min precision by time is {min_precision:.2f}")
    if min_precision > 0.7:
        best_stocks_by_time.append(stock)
    results_df['Time_dt'] = pd.to_datetime(results_df['Time'], format='%H:%M')
    results_df = results_df.sort_values('Time_dt')
    plt.plot(results_df['Time'], results_df['Precision'], marker='o', linestyle='-', label=stock)
print("by time : Stocks with min precision > 0.68:", best_stocks_by_time)
#"min_precision_by_time": 0.75,
plt.xlabel('Day of Week')
plt.ylabel('Precision')
plt.title('Precision of Predictions by Time for Each Stock')
plt.legend(title='Stock')
plt.grid(True)
plt.tight_layout()
plt.show()

#VISUALIZE PRECISION BY DAY OF WEEK:
results = []
df['Day'] = pd.to_datetime(df['Date']).dt.day_name().str[:3]
day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
plt.figure(figsize=(10, 6))
filtered_stocks = []
for stock in stocks:
    df_stock = df[df['Stock'] == stock].copy()
    results = []
    for day, group in df_stock.groupby('Day'):
        precision, _, _, _, _ = calculate_all_metrics(group['to_buy_1d'], group['prediction'])
        results.append({'Day': day, 'Precision': precision})
    results_df = pd.DataFrame(results)
    min_precision = results_df['Precision'].min()
    if min_precision > 0.7:
        filtered_stocks.append(stock)
    results_df['Day'] = pd.Categorical(results_df['Day'], categories=day_order, ordered=True)
    results_df = results_df.sort_values('Day')
    plt.plot(results_df['Day'], results_df['Precision'], marker='o', linestyle='-', label=stock)
print("by day of week: Stocks with min precision > 0.68:", filtered_stocks)
plt.xlabel('Day of Week')
plt.ylabel('Precision')
plt.title('Precision of Predictions by Day of Week for Each Stock')
plt.legend(title='Stock')
plt.grid(True)
plt.tight_layout()
plt.show()

#VISULAIZE PRECISION BY MONTH:
# Create a dictionary to store monthly precision for each ticker
ticker_precisions = {}

# Calculate precision for each ticker
for ticker in df['Stock'].unique():
    ticker_df = df[df['Stock'] == ticker]
    if ticker== 'TSN':
        ticker_df.to_csv("TSN_df.csv", index=True)
    monthly_precision = {}
    
    # Group by month and calculate precision
    for month, group in ticker_df.groupby(ticker_df['Date'].dt.to_period('M').astype(str)):
        if ticker == 'TSN' and month.startswith('2025-01'):  # Adjust year as needed for January
            print("actual for TSN (January):")
            for date, value in zip(group['Date'], group['to_buy_1d']):
                print(f"{date.strftime('%Y-%m-%d')}: {value}")
            
            print("\nmy predictions:")
            for date, pred in zip(group['Date'], group['prediction']):
                print(f"{date.strftime('%Y-%m-%d')}: {pred}")
            
            # Count 0s predicted as 1s (false positives)
            false_positives = sum((group['to_buy_1d'] == 0) & (group['prediction'] == 1))
            print(f"\nNumber of 0s predicted as 1s (false positives): {false_positives}")
        precision, _, _, _, _ = calculate_all_metrics(group['to_buy_1d'], group['prediction'])
        monthly_precision[month] = precision
    
    ticker_precisions[ticker] = monthly_precision
fig = plot_monthly_precision_all_tickers(df,ticker_precisions)
plt.show()

df_single_stock = df[df['Stock'].isin([MAIN_TICKER])]
df_single_stock['Month'] = df_single_stock['Date'].dt.to_period('M').astype(str)
monthly_precision = {}
for month, group in df_single_stock.groupby('Month'):
    precision, _, _, _, _ = calculate_all_metrics(group['to_buy_1d'], group['prediction'])
    monthly_precision[month] = precision
months = sorted(monthly_precision.keys())
precisions = [monthly_precision[m] for m in months]
plt.figure(figsize=(10,6))
plt.plot(months, precisions, marker='o', linestyle='-')
plt.xlabel("Month")
plt.ylabel("Precision")
plt.title(f"Monthly Prediction Precision for {MAIN_TICKER}")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


#VISUALIZE HOW THE THRESHOLD IMPACT THE PREICISON

plt.figure(figsize=(12, 8))
thresholds = np.linspace(0, 1, 50)

for stock in stocks:
    df_stock = df[df['Stock'] == stock]
    y_true = df_stock['to_buy_1d']
    #y_scores = df_stock['prediction_prob']
    y_scores = df_stock['prediction_prob']
    precisions = []
    max_precision=-1
    best_threshold=-1
    for t in thresholds:
        # Generate predictions based on threshold
        y_pred = (y_scores > t).astype(int)
        
        # Compute precision (set zero_division=0 to return 0 when no positive predictions)
        prec = precision_score(y_true, y_pred, zero_division=0)
        precisions.append(prec)
        if prec >= max_precision:
            max_precision = prec
            best_threshold = t
    #get the threshold from where the curve start decreasing
    print(f" for the stock {stock} \"best_probability_threshold\": {best_threshold:.2f}, with precision {max_precision}")
    plt.plot(thresholds, precisions, marker='.', linestyle='-', label=f'{stock}')
#"best_probability_threshold": 0.94,
# Add labels and title
plt.xlabel('Probability Threshold')
plt.ylabel('Precision')
plt.title('Effect of Probability Threshold on Precision for Multiple Stocks')
plt.legend(loc='best')
plt.grid(True)

# Set y-axis limits to focus on relevant range
plt.ylim([0, 1])

plt.show()
"""
precisions = []
for t in thresholds:
    # Generate predictions: prediction is 1 if probability > threshold, else 0
    y_pred = (df_single_stock['prediction_prob'] > t).astype(int)
    
    # Compute precision.
    # Setting zero_division=0 ensures that if no positive predictions are made,
    # precision_score returns 0 instead of raising an error.
    prec = precision_score(df_single_stock['to_buy_1d'], y_pred, zero_division=0)
    precisions.append(prec)

# Plot the precision vs threshold
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions, marker='o', linestyle='-')
plt.xlabel('Probability Threshold')
plt.ylabel('Precision')
plt.title('Effect of Probability Threshold on Precision')
plt.grid(True)
plt.show()

"""
####ROC AUC curve#####
# Create a new figure with a specific size (optional)
plt.figure(figsize=(6, 3))
long_text = ("We will now show you the ROC curve. What is ROC curve ? "
             "for each probability threshold, we take the corresponding TPR and FPR . "
             "so with  a single threhold we generate a point( TPR,FPR)."
            "then all those points will form a curve."
            "AUC is just the area under that curve (between 0 an 1) , 1 is the ideal value"
            "because 1 means that we have a threshold that satisfies TPR =1 and FPR =0"
            "but in our case TPR is not very important so we value the points with low FPR more"
            "so AUC is maybe not that important in our case") #https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
# Wrap the text at a chosen width (e.g., 60 characters per line)
wrapped_text = "\n".join(textwrap.wrap(long_text, width=60))
# Create a larger figure to better display the text
plt.figure(figsize=(8, 6))
# Optionally, adjust the subplot parameters to have more space
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.text(0.5, 0.5, wrapped_text, ha='center', va='center', fontsize=12)
plt.axis('off')
plt.show()

# PLOT THE ROC CURVE FOR EACH STOCK

plt.figure(figsize=(12, 8))
# For each stock, calculate and plot ROC curve
for stock in stocks:
    df_stock = df[df['Stock'] == stock]
    y_true = df_stock['to_buy_1d']
    #y_scores = df_stock['prediction_prob']
    y_scores = df_stock['prediction_prob']
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_value = roc_auc_score(y_true, y_scores)
    # Plot the ROC curve for this stock
    plt.plot(fpr, tpr, label=f'{stock} (AUC = {auc_value:.2f})', linewidth=2)

# Add reference line for random classifier
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
# Add labels and title
plt.xlabel('False Positive Rate (proportion of real 0s predicted as 1s)')
plt.ylabel('True Positive Rate (how much of real 1s are predicted as 1s (=recall))')
plt.title('ROC Curves for Multiple Stocks')
plt.legend(loc='lower right')
plt.grid(True)
# Add equal aspect ratio to make the plot square
plt.axis('square')
# Limit axes to [0,1]
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()



y_true = df_single_stock['to_buy_1d']
y_scores = df_single_stock['prediction_prob']

# Compute the false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# Calculate the Area Under the Curve (AUC)
auc_value = roc_auc_score(y_true, y_scores)

# Plot the ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_value:.2f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')  # Diagonal line for reference
plt.xlabel('False Positive Rate (proportion of real 0s predicted as 1s)')
plt.ylabel('True Positive Rate (how much of real 1s are predicted as 1s (=recall))')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
import os
import json
from collections import Counter
import glob
import re
import pandas as pd
import traceback

def safe_div(a, b):
    try:
        return float(a) / float(b)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    
root_dir = "C:/Users/aziz8/Documents/TRADINGBOT"
# Counter to track feature occurrences
feature_counter = Counter()

# Find all folders that start with old_stuff but are not old_stuff_analysis
"""folders = [f for f in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, f)) 
            and f.startswith("old_stuff") 
            and f != "old_stuff_analysis"]"""

folders = [f for f in os.listdir(root_dir)
          if os.path.isdir(os.path.join(root_dir, f))
          and f.startswith("old_stuff")
          and f != "old_stuff_analysis"
          and 1 <= int(f.replace("old_stuff", "")) <= 200]

print(f"Found {len(folders)} folders to process.")
csv_data = []
features_set = set()
FD_1_counter= 0
# Process each folder
for folder in folders:
    folder_path = os.path.join(root_dir, folder)
    models_file = os.path.join(folder_path, "models_to_use.json")
    print("Processing folder:", folder_path)
    if os.path.exists(models_file):
        try:
            with open(models_file, 'r') as f:
                models_data = json.load(f)
            for ticker, model_infos in models_data.items():
                if not isinstance(model_infos, list):
                    print("type of model_infos is : ", type(model_infos))
                    model_infos = [model_infos]  # Ensure it's a list
                else:
                    print("type of model_infos is : ", type(model_infos))
                for model_info in model_infos:
                    row = {"Ticker": ticker}

                    # Extracting "best_model" string
                    row["Model"] = (
                        model_info.get("best_model") or model_info.get("model_file", "")
                    ).split("_")[0]
                    if row["Model"] == "":
                        print(f"Warning: No model found for {ticker} in {models_file}.")
                        continue
                    # Extracting numeric values
                    row["Precision"] = model_info.get("precision", None)
                    row["Specificity"] = model_info.get("specificity", None)
                    row["Recall"] = model_info.get("recall", None)
                    row["good_model"] = model_info.get("good_model", None)
                    if row["good_model"] is None:
                        print(f"Warning: No good_model found for {ticker} in {models_file}.")
                    row["precision_after_cutoff"]= model_info.get("precision_after_cutoff", None)
                    if row["precision_after_cutoff"] is None:
                        print(f"Warning: No precision_after_cutoff found for {ticker} in {models_file}.")
                    row["min_precision_by_time"] = model_info.get("min_precision_by_time", None)
                    row["best_probability_threshold"] = model_info.get("best_probability_threshold", None)
                    row["init_percentage_of_1s"] = model_info.get("init_percentage_of_1s", None)
                    row["testing_length_in_days"]= model_info.get("testing_length_in_days", None)
                    row["std_precision_by_time"]= model_info.get("std_precision_by_time", None)
                    row["min_precision_by_month"]= model_info.get("min_precision_by_month", None)
                    row["std_precision_by_month"]= model_info.get("std_precision_by_month", None)
                    row["training_precision"]= model_info.get("training_precision", None)
                    row["file_created_after_20_03"]= model_info.get("file_created_after_20_03", 0)
                    row["lookahead_in_hours"]= model_info.get("lookahead_in_hours", 0)
                    #row["number_of_red_slopes_in_test"]= model_info.get("number_of_red_slopes_in_test", None)
                    #row["number_of_red_slopes_in_train"]= model_info.get("number_of_red_slopes_in_train", None)
                    
                    training_length = model_info.get("training_length", None)
                    testing_length = model_info.get("testing_length", None)
                    ratio = None 
                    if training_length is not None and testing_length is not None:
                        denominator = training_length + testing_length
                        if denominator > 0:
                            ratio = testing_length / denominator
                        else:
                            ratio = None  

                    row["training_length"]= training_length
                    row["testing_length"]= testing_length
                    row["testing_len_ratio"] = ratio
                    if row["Model"].startswith("RF"):
                        row["Model_type"]="RF"
                    elif row["Model"].startswith("XGB"):
                        row["Model_type"]="XGB"
                    elif row["Model"].startswith("DT"):
                        row["Model_type"]="DT"
                    elif row["Model"].startswith("BG"):
                        row["Model_type"]="BG"
                    else:
                        row["Model_type"]="Other"
                    
                    row["red_slopes_ratio_test"]  = safe_div(
                        model_info.get("number_of_red_slopes_in_test"),
                        row.get("testing_length")
                    )
                    row["red_slopes_ratio_train"] = safe_div(
                        model_info.get("number_of_red_slopes_in_train"),
                        row.get("training_length")
                    )
                    row["red_ratio_test2"]  = safe_div(
                        model_info.get("test_nbr_of_red2"),
                        row.get("testing_length")
                    )
                    row["red_ratio_train2"] = safe_div(
                        model_info.get("train_nbr_of_red2"),
                        row.get("training_length")
                    )
                    row["train_redness_index"] = model_info.get("train_redness_index", None)
                    row["test_redness_index"] = model_info.get("test_redness_index", None)
                    # Extracting features as separate columns
                    features = model_info.get("subset", [])
                    if features == []:
                        print(f"Warning: No features found for {ticker} in {models_file}.")
                    for feature in features:
                        row[feature] = 1  # Mark feature presence
                        features_set.add(feature)

                    row["features_length"] = len(features)
                    # Ensure all feature columns are present with 0 if not included
                    csv_data.append(row)
        except Exception as e:
            print(f"Error processing {models_file}: {str(e)}")
            traceback.print_exc()


# Convert to DataFrame
df = pd.DataFrame(csv_data)

print("FD_1 value counts:")
print(df['FD_1'].value_counts(dropna=False))

nan_forbidden_in_these_cols = ["Ticker", "Model","Model_type", "Precision", "Specificity", "Recall", "good_model"]
none_percentages = df.isna().mean() * 100

df.dropna(subset=nan_forbidden_in_these_cols, inplace=True)

none_percentages2 = df.drop(columns=features_set).isna().mean() * 100
print(none_percentages2)

for column in df.columns:
    if column not in features_set:
        percentage = df[column].isna().mean() * 100
        print(f"{column}: {percentage:.2f}%")
        
        if percentage >1 and column not in nan_forbidden_in_these_cols:
            print(f"\nColumn '{column}' has {percentage:.2f}% missing values.")
            action = input("Enter 'c' to drop the column, 'r' to drop rows with NaN in this column, or any other key to skip: ")
            
            if action.lower() == 'c':
                df = df.drop(columns=[column])
                print(f"Column '{column}' has been dropped from the DataFrame.")
            elif action.lower() == 'r':
                df = df.dropna(subset=[column])
                print(f"Rows with missing values in column '{column}' have been dropped from the DataFrame.")
            else:
                print(f"No action taken for column '{column}'.")

#delete row that have None in one of these columns : Ticker,Best Model,Precision,Specificity,Recall,good_model
#df.dropna(subset=nan_forbidden_in_these_cols, inplace=True)

# This will only fill NA values with 0 for columns in features_set
df[list(features_set)] = df[list(features_set)].fillna(0)

none_percentages = df.isna().mean() * 100
print("Percentage of None values in each column AFTER DROPPING NA:")
for column, percentage in none_percentages.items():
    if column not in features_set:
        print(f"{column}: {percentage:.2f}%")
csv_output_path = os.path.join(root_dir, "old_stuff_analysis/all_models_to_csv.csv")
df.to_csv(csv_output_path, index=False)
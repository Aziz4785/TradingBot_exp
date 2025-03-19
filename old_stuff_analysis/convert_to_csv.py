import os
import json
from collections import Counter
import glob
import re
import pandas as pd

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
          and 1 <= int(f.replace("old_stuff", "")) <= 9]

print(f"Found {len(folders)} folders to process.")
csv_data = []
features_set = set()
# Process each folder
for folder in folders:
    folder_path = os.path.join(root_dir, folder)
    models_file = os.path.join(folder_path, "models_to_use.json")
    if os.path.exists(models_file):
        try:
            with open(models_file, 'r') as f:
                models_data = json.load(f)
            for ticker, model_info in models_data.items():
                row = {"Ticker": ticker}

                # Extracting "best_model" string
                row["Model"] = model_info.get("best_model", "").split("_")[0]

                # Extracting numeric values
                row["Precision"] = model_info.get("precision", None)
                row["Specificity"] = model_info.get("specificity", None)
                row["Recall"] = model_info.get("recall", None)
                row["good_model"] = model_info.get("good_model", None)
                row["min_precision_by_time"] = model_info.get("min_precision_by_time", None)
                row["best_probability_threshold"] = model_info.get("best_probability_threshold", None)
                row["init_percentage_of_1s"] = model_info.get("init_percentage_of_1s", None)
                if row["Model"].startswith("RF"):
                    row["Model_type"]="RF"
                elif row["Model"].startswith("XGB"):
                    row["Model_type"]="XGB"
                elif row["Model"].startswith("DT"):
                    row["Model_type"]="DT"
                else:
                    row["Model_type"]="Other"
                
                # Extracting features as separate columns
                features = model_info.get("subset", [])
                for feature in features:
                    row[feature] = 1  # Mark feature presence
                    features_set.add(feature)

                row["features_length"] = len(features)
                # Ensure all feature columns are present with 0 if not included
                csv_data.append(row)
        except Exception as e:
            print(f"Error processing {models_file}: {str(e)}")


# Convert to DataFrame
df = pd.DataFrame(csv_data)

none_percentages = df.isna().mean() * 100
print("Percentage of None values in each column BEFORE DROPPING NA:")
for column, percentage in none_percentages.items():
    if column not in features_set:
        print(f"{column}: {percentage:.2f}%")
#delete row that have None in one of these columns : Ticker,Best Model,Precision,Specificity,Recall,good_model
df.dropna(subset=["Ticker", "Model","Model_type", "Precision", "Specificity", "Recall", "good_model"], inplace=True)

# This will only fill NA values with 0 for columns in features_set
df[list(features_set)] = df[list(features_set)].fillna(0)

none_percentages = df.isna().mean() * 100
print("Percentage of None values in each column AFTER DROPPING NA:")
for column, percentage in none_percentages.items():
    if column not in features_set:
        print(f"{column}: {percentage:.2f}%")
csv_output_path = os.path.join(root_dir, "old_stuff_analysis/all_models_to_csv.csv")
df.to_csv(csv_output_path, index=False)
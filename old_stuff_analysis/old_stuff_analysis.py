import os
import json
from collections import Counter
import glob
import re

root_dir = "C:/Users/aziz8/Documents/TRADINGBOT"
folders = [f for f in os.listdir(root_dir)
          if os.path.isdir(os.path.join(root_dir, f))
          and f.startswith("old_stuff")
          and f != "old_stuff_analysis"
          and 6 <= int(f.replace("old_stuff", "")) <= 60]

print("folders", folders)
#old_stuff1 -> old_stuff5  (15 dernier jours de fevrier)
#old_stuff6 -> old_stuff  (MARS 2025)
def find_all_feature_sets():
    """Get all feature sets from models_to_use.json files in old_stuff folders"""
    # List to store all feature sets
    all_feature_sets = []
    
    # Find all folders that start with old_stuff but are not old_stuff_analysis
    """folders = [f for f in os.listdir(root_dir) 
               if os.path.isdir(os.path.join(root_dir, f)) 
               and f.startswith("old_stuff") 
               and f != "old_stuff_analysis"]"""

    # Process each folder
    for folder in folders:
        folder_path = os.path.join(root_dir, folder)
        models_file = os.path.join(folder_path, "models_to_use.json")
        
        # Check if models_to_use.json exists in this folder
        if os.path.exists(models_file):
            try:
                with open(models_file, 'r') as f:
                    models_data = json.load(f)
                
                # Extract feature sets from each ticker's model
                for ticker, model_info in models_data.items():
                    if "subset" in model_info:
                        # Add the feature set (as a frozenset for hashability)
                        all_feature_sets.append(frozenset(model_info["subset"]))
            except Exception as e:
                print(f"Error processing {models_file}: {str(e)}")
    
    return all_feature_sets

def find_and_rank_features():
    # Counter to track feature occurrences
    feature_counter = Counter()
    
    # Find all folders that start with old_stuff but are not old_stuff_analysis
    """folders = [f for f in os.listdir(root_dir) 
               if os.path.isdir(os.path.join(root_dir, f)) 
               and f.startswith("old_stuff") 
               and f != "old_stuff_analysis"]"""
    
    print(f"Found {len(folders)} folders to process.")
    
    # Process each folder
    for folder in folders:
        folder_path = os.path.join(root_dir, folder)
        models_file = os.path.join(folder_path, "models_to_use.json")
        
        # Check if models_to_use.json exists in this folder
        if os.path.exists(models_file):
            try:
                with open(models_file, 'r') as f:
                    models_data = json.load(f)
                
                # Extract features from each ticker's model
                for ticker, model_info in models_data.items():
                    if "subset" in model_info:
                        features = model_info["subset"]
                        # Update counter with these features
                        feature_counter.update(features)
                
                print(f"Processed {models_file}")
            except json.JSONDecodeError:
                print(f"Error: Could not parse JSON in {models_file}")
            except Exception as e:
                print(f"Error processing {models_file}: {str(e)}")
        else:
            print(f"No models_to_use.json found in {folder_path}")
    
    # Rank features by count
    ranked_features = feature_counter.most_common()
    
    # Display results
    print("\n--- Features Ranked by Frequency ---")
    for feature, count in ranked_features:
        print(f"{feature}: {count}")
    
    return ranked_features

def check_missing_feature_sets(feature_sets_to_check):
    """Check which feature sets from the list are not present in any file"""
    # Get all feature sets from the files
    existing_feature_sets = find_all_feature_sets()
    
    # Convert each feature set to check to a frozenset for comparison
    feature_sets_to_check_frozen = [frozenset(fs) for fs in feature_sets_to_check]
    
    # Find missing feature sets
    missing_feature_sets = []
    
    for i, fs in enumerate(feature_sets_to_check_frozen):
        if fs not in existing_feature_sets:
            # Return the original list format for better readability
            missing_feature_sets.append(feature_sets_to_check[i])
    
    return missing_feature_sets

def count_model_types():
    """Count the occurrences of different model types (RF, XGB, DT) in all models_to_use.json files"""
    
    # Counter for model types
    model_counter = Counter()
    model_counter_good_models_only = Counter()
    subModel_counter_good_models_only  = Counter()
    # Full model name counter (to see specific versions)
    full_model_counter = Counter()
    
    print(f"Found {len(folders)} folders to process.")
    
    # Process each folder
    for folder in folders:
        folder_path = os.path.join(root_dir, folder)
        models_file = os.path.join(folder_path, "models_to_use.json")
        
        # Check if models_to_use.json exists in this folder
        if os.path.exists(models_file):
            try:
                with open(models_file, 'r') as f:
                    models_data = json.load(f)
                
                # Extract model names from each ticker's model
                for ticker, model_info in models_data.items():
                    if "best_model" in model_info:
                        goodmodel=False
                        if "good_model" in model_info and model_info["good_model"]==1:
                            goodmodel = True
                        model_name = model_info["best_model"]
                        
                        # Store the full model name
                        full_model_counter[model_name] += 1
                        
                        if goodmodel:
                            submodel = model_name.split('_')[0]
                            subModel_counter_good_models_only[submodel] += 1
                        # Extract model type (RF, XGB, DT)
                        if model_name.startswith("RF"):
                            model_counter["RF"] += 1
                            if goodmodel:
                                model_counter_good_models_only["RF"] += 1
                        elif model_name.startswith("XGB"):
                            model_counter["XGB"] += 1
                            if goodmodel:
                                model_counter_good_models_only["XGB"] += 1
                        elif model_name.startswith("DT"):
                            model_counter["DT"] += 1
                            if goodmodel:
                                model_counter_good_models_only["DT"] += 1
                        elif model_name.startswith("BG"):
                            model_counter["BG"] += 1
                            if goodmodel:
                                model_counter_good_models_only["BG"] += 1
                        else:
                            model_counter["Other"] += 1
                            if goodmodel:
                                model_counter_good_models_only["Other"] += 1
                
                print(f"Processed {models_file}")
            except Exception as e:
                print(f"Error processing {models_file}: {str(e)}")
        else:
            print(f"No models_to_use.json found in {folder_path}")
    
    # Display results
    print("\n--- Model Types Ranked by Frequency ---")
    for model_type, count in model_counter.most_common():
        print(f"{model_type}: {count} ({count/sum(model_counter.values())*100:.2f}%)")
    
    print("\n--- Model Types Ranked by Frequency (only for good_model ==1---")
    for model_type, count in model_counter_good_models_only.most_common():
        print(f"{model_type}: {count} ({count/sum(model_counter_good_models_only.values())*100:.2f}%)")

    print("\n--- Model Sub Types Ranked by Frequency (only for good_model ==1---")
    for model_type, count in subModel_counter_good_models_only.most_common():
        print(f"{model_type}: {count} ({count/sum(subModel_counter_good_models_only.values())*100:.2f}%)")

    print("\n--- Top 10 Specific Models ---")
    for model_name, count in full_model_counter.most_common(10):
        print(f"{model_name}: {count}")
    
    # Analyze model structure patterns (e.g., RF5_TICKER_24 format)
    pattern_counter = Counter()
    for model_name in full_model_counter:
        match = re.match(r'([A-Z]+)(\d+)_([A-Z]+)_(\d+)', model_name)
        if match:
            model_type, version, ticker, subset_id = match.groups()
            pattern_counter[f"{model_type}{version}"] += 1
    
    print("\n--- Model Versions ---")
    for pattern, count in pattern_counter.most_common():
        print(f"{pattern}: {count} ({count/sum(pattern_counter.values())*100:.2f}%)")
    
    return model_counter, full_model_counter, pattern_counter

if __name__ == "__main__":
    find_and_rank_features()
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
    count_model_types()

    missing_sets = check_missing_feature_sets(feature_subsets)
    
    if missing_sets:
        print("The following feature sets were not found in any models_to_use.json file:")
        for i, fs in enumerate(missing_sets):
            print(f"{i+1}. {fs}")
    else:
        print("All feature sets were found in at least one models_to_use.json file.")
    
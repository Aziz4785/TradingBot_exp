import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import random
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.base import clone

def evaluate_model_with_random_splits(model, X, y, n_splits=3, test_size=0.25):
    scores = []

    for _ in range(n_splits):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=None  # random split each time
        )
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        score = model_clone.score(X_test, y_test)
        scores.append(score)

    mean_score = sum(scores) / len(scores)
    return mean_score

features_to_test = [
    [],
    ['Model_RF8', 'Model_RF7', 'high_quad_q_rel_class', 'training_length'],
    ['Recall', 'Close_to_Close_1_day_ago_class', 'red_slopes_ratio_train'],
    ['Model_DT9', 'Recall', 'red_slopes_ratio_train', 'PM_max_to_min_ratio', 'red_slopes_ratio_test'],
    ['FD_3', 'Close_class', 'PM_max_to_Close_ratio_class', 'Model_RF8', 'AH_max_1dayago_to_Close_class'],
    ['Model_DT9', 'Recall', 'red_slopes_ratio_train', 'PM_max_to_min_ratio', 'red_slopes_ratio_test','FD_3', 'Close_class', 'PM_max_to_Close_ratio_class', 'Model_RF8', 'AH_max_1dayago_to_Close_class'],
    ['Recall', 'Model_BG2', 'date_after_1124', 'red_slopes_ratio_train', 'PM_max_to_min_ratio', 'Close_to_prevDayHigh_class'],
    ['Recall', 'red_slopes_ratio_train', 'Model_RF5', 'Model_RF9', 'features_length', 'return_2d_class', 'Model_DT4'],
    ['date_after_1124', 'Close_to_EMA_48_class', 'PM_max_to_Close_ratio_class', 'STD_10_class', 'init_percentage_of_1s', 'PM_range_to_close_ratio_class', 'Model_RF5', 'PM_volume_max_class', 'PM_min_to_Close_ratio_class', 'Model_DT10', 'min_precision_by_time', 'prev2DayClose_to_prevDayClose_ratio_class', 'Precision', 'Model_DT4', 'Model_type_RF', 'Model_DT8'],
    ['Model_RF8', 'dayOpen_to_prev2DayOpen_ratio_class'],
    ['red_slopes_ratio_test', 'Model_RF8'], #0.7479
    ['Model_RF8', 'AH_max_1dayago_vs_prevDayClose_class', 'return_2d_class', 'close_to_High10_class'],
    ['red_slopes_ratio_test', 'std_precision_by_time', 'coef_p_vol_rel_class', 'training_precision'],
    ['STD_10_class', 'red_slopes_ratio_test', 'PM_min_to_Close_ratio_class', 'Model_RF8', 'PM_min_to_prevDayClose_ratio_class'], #0.7564
    ['PM_min_to_prevDayOpen_ratio_class', 'PM_max_to_Close_ratio_class', 'high_quad_p_rel_class', 'FD_2', 'PM_time_diff_class'] ,
    ['Close_to_open_ratio_class', 'std_precision_by_time', 'training_length', 'training_precision', 'Recall', 'PM_min_to_Close_ratio_class'], #0.7650
    ['Close_to_prevDayHigh_class', 'Model_RF5', 'PM_max_to_PM_min_ratio_class', 'Recall', 'testing_len_ratio', 'testing_length', 'prev2DayClose_to_prevDayClose_ratio_class'],
    ['date_after_1124', 'training_length', 'FD_1', 'red_slopes_ratio_test', 'Recall', 'red_slopes_ratio_train'],
    ['testing_length_in_days', 'PM_min_to_open_ratio_class', 'dayOpen_to_prevDayOpen_ratio_class', 'Close_to_Close_1_day_ago_class', 'Model_XGB9', 'red_slopes_ratio_train', 'Recall', 'init_percentage_of_1s'],
    ['Model_type_XGB', 'testing_length_in_days', 'lookahead_in_hours', 'testing_len_ratio', 'slope_a_vol_rel_class', 'Open_1_day_ago_to_Close_1_day_ago_ratio_class', 'time_in_minutes', 'Model_BG1', 'Recall'],
    ['Model_DT9', 'dayOpen_to_prev2DayOpen_ratio_class'],
    ['Model_DT9', 'Close_to_Close_1_day_ago_class', 'dayOpen_to_prev2DayOpen_ratio_class'],
    ['Model_RF9', 'open_to_prev_close_class', 'PM_max_to_prevDayOpen_ratio_class', 'Volume3', 'dayOpen_to_prev2DayOpen_ratio_class'],
    ['Model_type_RF', 'Close_to_EMA_48_class', 'init_percentage_of_1s', 'dayOpen_to_prev2DayOpen_ratio_class'],
    ['Model_DT3', 'Specificity', 'Model_RF9', 'training_precision', 'Model_DT9'],
    ['Volume_class', 'PM_volume_sum_class', 'AH_max_1dayago_to_Close_class', 'Model_DT9', 'dayOpen_to_prev2DayOpen_ratio_class'] ,
    ['init_percentage_of_1s', 'testing_len_ratio', 'Model_DT10', 'Model_RF1', 'Model_type_RF', 'dayOpen_to_prev2DayOpen_ratio_class'],
    ['Model_XGB7', 'testing_len_ratio', 'Model_RF7', 'Model_RF9', 'date_after_0924', 'return_2d_class', 'dayOpen_to_prev2DayOpen_ratio_class'],
    ['FD_2', 'PM_time_diff_class', 'Model_RF4', 'Model_DT9', 'init_percentage_of_1s', 'high_quad_p_rel_class', 'dayOpen_to_prev2DayOpen_ratio_class'],
    
    ['red_slopes_ratio_train', 'high_slope_rel_class', 'training_length', 'PM_min_to_open_ratio_class', 'std_precision_by_time'] ,
    ['Open_1_day_ago_to_Close_1_day_ago_ratio_class', 'red_slopes_ratio_train', 'features_length', 'PM_range_to_open_ratio_class', 'Model_RF8', 'high_quad_p_rel_class', 'dayOpen_to_prev2DayOpen_ratio_class'],
    ['training_precision', 'features_length', 'lookahead_in_hours', 'init_percentage_of_1s', 'training_length', 'Recall', 'PM_min_to_open_ratio_class', 'AH_max_1dayago_vs_prevDayClose_class', 'PM_max_to_prevDayOpen_ratio_class'],
    ['Recall', 'Close_to_open_ratio_class', 'testing_len_ratio', 'Close_to_prevDayHigh_class', 'std_precision_by_time', 'Close_to_prevDayClose_class', 'training_length', 'features_length', 'AH_max_1dayago_to_Close_class', 'PM_min_to_prevDayOpen_ratio_class', 'PM_min_to_Close_ratio_class', 'PM_min_to_prevDayClose_ratio_class'],
    ['Close_to_Close_1_day_ago_class', 'PM_min_to_Close_ratio_class', 'Model_type_RF', 'AH_max_1dayago_vs_PM_max_class', 'init_percentage_of_1s', 'testing_length_in_days', 'Model_DT9', 'testing_len_ratio', 'Recall', 'PM_max_to_dayOpen_ratio_class', 'Close_to_prevDayHigh_class', 'prev2DayClose_to_prevDayClose_ratio_class', 'AH_max_1dayago_to_Close_class'],
    ['min_precision_by_time', 'AH_max_1dayago_vs_PM_max_class', 'std_precision_by_time', 'dayOpen_to_prevDayOpen_ratio_class', 'testing_length_in_days', 'PM_max_to_min_ratio'],
    #my features:
    ['Model_BG1','Model_BG2','Model_DT1','Model_DT10','Model_DT3','Model_DT4','Model_DT6','Model_DT8','Model_DT9','Model_RF1','Model_RF3','Model_RF4','Model_RF5','Model_RF6','Model_RF7','Model_RF8','Model_RF9','Model_XGB1','Model_XGB7','Model_XGB9'],
    ['init_percentage_of_1s','Model_type_DT','Model_type_RF','Model_type_XGB','red_slopes_ratio_test','red_slopes_ratio_train','testing_len_ratio'],
    ['features_length','Model_type_DT','Model_type_RF','init_percentage_of_1s','training_precision','Precision'],
    ['training_length','training_precision'],
    ['min_precision_by_time', 'date_after_0125', 'std_precision_by_time', 'testing_length_in_days'],
    ['Model_RF8', 'PM_max_to_prevDayOpen_ratio_class', 'min_precision_by_month', 'Model_RF6', 'AH_max_1dayago_vs_prevDayClose_class', 'dayOpen_to_prevDayOpen_ratio_class'],

]

df = pd.read_csv("old_stuff_analysis/balanced_targets_with0good.csv")
df.drop(columns=["Ticker"], inplace=True, errors='ignore')
target_column = 'good_model'
df.loc[df[target_column] == 0, target_column] = 1
df.loc[df[target_column] == -1, target_column] = 0
#print the value counts for FD_1 column:
print(df['FD_1'].value_counts(dropna=False))

#print columns that have the same value for at least 90% or rows (and print tha value):

threshold_percentage = 0.97
total_rows = len(df)
threshold_count = total_rows * threshold_percentage
columns_to_remove = []
for col in df.columns:
    # Calculate value counts (includes NaNs if they exist and are frequent)
    # Use dropna=False if you want NaN to be treated as a potential dominant value
    counts = df[col].value_counts(dropna=False)

    # Check if there are any values in the column at all
    if not counts.empty:
        # Get the count of the most frequent value
        most_frequent_count = counts.iloc[0]
        # Get the most frequent value itself
        dominant_value = counts.index[0]

        # Check if the most frequent count meets the threshold
        if most_frequent_count >= threshold_count:
            # Optional: Handle NaN display nicely
            display_value = 'NaN' if pd.isna(dominant_value) else dominant_value
            percentage = (most_frequent_count / total_rows) * 100
            print(f"Column '{col}' has '{display_value}' for {percentage:.1f}% of rows ({most_frequent_count}/{total_rows}).")
            columns_to_remove.append(col)
df.drop(columns=columns_to_remove, inplace=True)
print(f"Removed columns: {columns_to_remove}")

X = df.drop(columns=[target_column])
y = df[target_column]
features = X.columns.tolist()

# Define models
models = {
    "RandomForest": RandomForestClassifier(),
    "RandomForest_depth5": RandomForestClassifier(max_depth=5),
    "rf3": RandomForestClassifier(n_estimators=200,max_depth=3),
    "rf4": RandomForestClassifier(n_estimators=300,max_depth=10, min_samples_leaf=5,class_weight="balanced",max_features="sqrt",random_state=42),
    "rf5": RandomForestClassifier(n_estimators=300,max_depth=None,max_features="sqrt"),
    "rf6": RandomForestClassifier(n_estimators=200,max_depth=20,max_features=0.3,min_samples_leaf= 1,min_samples_split= 7),
    "rf7": RandomForestClassifier(n_estimators=500,max_depth=None,max_features="sqrt",min_samples_leaf= 2,min_samples_split= 7),
    "rf8": RandomForestClassifier(n_estimators=200,max_depth=20,max_features="sqrt",min_samples_leaf= 1,min_samples_split= 7),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "LogisticRegression_C0.1": LogisticRegression(C=0.1, max_iter=1000),
    "lr3": LogisticRegression(C=0.1, penalty="l2", solver="liblinear"),
    "SVC": make_pipeline(StandardScaler(), SVC()),  # Scale for SVM
    "GradientBoosting": GradientBoostingClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "DecisionTree_minSamples10": DecisionTreeClassifier(min_samples_split=6),
    "dt3": DecisionTreeClassifier(max_depth=10, min_samples_split=10, min_samples_leaf=1, max_features=0.3, splitter="best"),

}   

# Storage for best models
best_models = {}

# For each model
for name, model in models.items():
    print(f"\nTrying model: {name}")
    best_score = 0
    best_features = None
    best_model = None

    for selected_features in features_to_test:
        #selected_features = random.sample(features, i)
        if len(selected_features) == 0:
            selected_features = X.columns.tolist()  # Use all features if none selected 
            X_subset = X
        else:
            # Ensure selected features are in the DataFrame
            selected_features = [feature for feature in selected_features if feature in X.columns]
            X_subset = X[selected_features]

        try:
            mean_score = evaluate_model_with_random_splits(model, X_subset, y)
            #scores = cross_val_score(model, X_subset, y, cv=5, scoring='accuracy') #The data is split into 5 equal parts 
            #mean_score = scores.mean()
            #print(f"  Features: {selected_features} | CV Score: {mean_score:.4f}")
            if mean_score > best_score:
                best_score = mean_score
                best_features = selected_features
                print(f"  Features: {selected_features} | CV Score: {mean_score:.4f}")

        except Exception as e:
            print(f"  Skipped due to error: {e}")

    # Now retrain the model on full data using best features
    if best_features is None:
        continue  # No good features found
    if len(best_features) == 0:
        X_final = X
    else:
        X_final = X[best_features]

    trained_model = clone(model)  # Reinitialize a fresh model
    trained_model.fit(X_final, y)  # Train on full data

    best_models[name] = {
        "score": best_score,
        "features": best_features,
        "model": trained_model  # Save the trained model!
    }

# Print the best models and their performance
# Save all the models
print("\nBest models and feature subsets:")
for name, info in best_models.items():
    if info['score'] > 0.75:
        print()
        print()
        print(f"{name}: Score = {info['score']:.4f}, Features = {info['features']}")

        # Save the model
        model_filename = f"meta_model/{name}_best_model.pkl"
        joblib.dump(info["model"], model_filename)
        print(f"Saved {name} model to {model_filename}")

        # Save the features as a JSON file
        features_filename = f"meta_model/{name}_features.json"
        with open(features_filename, 'w') as f:
            json.dump(info["features"], f, indent=4)
        print(f"Saved {name} features to {features_filename}")
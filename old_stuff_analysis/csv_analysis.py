from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv(f"old_stuff_analysis/all_models_to_csv.csv")
df = df.drop_duplicates()
target_column = "good_model"

df = pd.get_dummies(df, columns=["Model_type", "Model","Ticker"], prefix=["Model_type", "Model","Ticker"])

feature_columns = [col for col in df.columns if col not in ["Ticker", target_column]]
df = df.dropna(subset=[target_column])  # Drop rows where target is NaN
df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors='coerce')
print("length of df before dropping none  : ",len(df))
df = df.dropna()  # Drop remaining NaNs
print("length of df after dropping none  : ",len(df))

X = df[feature_columns]
print("length of X : ",len(X))
print("colomns of X : ",X.columns)
y = df[target_column]

# Train Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=3)  # Limit depth for better interpretability
clf.fit(X, y)

# Plot the decision tree
plt.figure(figsize=(20, 10))
tree.plot_tree(clf, feature_names=feature_columns, class_names=["-1", "0", "1"], filled=True, rounded=True, fontsize=12)
plt.title("Decision Tree")
plt.tight_layout() 
plt.show()

#IF WE CONSIDER 0s as BAD
df_bad = df.copy() 
df_bad["good_model"] = df_bad["good_model"].replace(0, -1)
X_bad = df_bad[feature_columns]
y_bad = df_bad[target_column]
clf_bad = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5)
clf_bad.fit(X_bad, y_bad)
plt.figure(figsize=(20, 10))
tree.plot_tree(clf_bad, feature_names=feature_columns, class_names=["-1", "1"], filled=True, rounded=True)
plt.title("Decision Tree (0s as BAD)")
plt.show()

# IF WE CONSIDER 0s as GOOD
df2_good = df.copy()
n_pos = (df2_good[target_column] == 1).sum()
n_neg = (df2_good[target_column] == -1).sum()
n_zero = (df2_good[target_column] == 0).sum()
print("n_pos = ",n_pos)
print("n_neg = ",n_neg)
print("n_zero = ",n_zero)
# How many zeros we want to keep
n_zero_to_keep = n_neg - n_pos
n_zero_to_drop = n_zero - n_zero_to_keep
print("n_zero_to_keep = ",n_zero_to_keep)
print("n_zero_to_drop = ",n_zero_to_drop)
# Drop the first n_zero_to_drop rows where target == 0
df_zeros_to_drop = df2_good[df2_good[target_column] == 0].iloc[:n_zero_to_drop]
df2_good = df2_good.drop(df_zeros_to_drop.index)
# Convert remaining 0s to 1
df2_good.loc[df2_good[target_column] == 0, target_column] = 1
df2_good.to_csv('balanced_targets_with0good.csv', index=False)

df_good = df.copy()  # Make a copy for this approach
df_good["good_model"] = df_good["good_model"].replace(0, 1)
X_good = df_good[feature_columns]
y_good = df_good[target_column]
clf_good = DecisionTreeClassifier(max_depth=3, min_samples_leaf=8)
clf_good.fit(X_good, y_good)
plt.figure(figsize=(20, 10))
tree.plot_tree(clf_good, feature_names=feature_columns, class_names=["-1", "1"], filled=True, rounded=True)
plt.title("Decision Tree (0s as GOOD)")
plt.show()

# IF WE ONLY CONSIDER EXISTING 1s AND -1s (removing 0s completely)
df_filtered = df.copy()
# Filter out rows where the target column equals 0
df_filtered = df_filtered[df_filtered[target_column] != 0]
df_filtered.to_csv("old_stuff_analysis/csv_without_0.csv", index=False)
X_filtered = df_filtered[feature_columns]
y_filtered = df_filtered[target_column]
clf_filtered = DecisionTreeClassifier(max_depth=2, min_samples_leaf=4)
clf_filtered.fit(X_filtered, y_filtered)
plt.figure(figsize=(20, 10))
tree.plot_tree(clf_filtered, feature_names=feature_columns, class_names=["-1", "1"], filled=True, rounded=True)
plt.title("Decision Tree (0s removed)")
plt.show()

import random
from sklearn.metrics import precision_score, accuracy_score
from sklearn.tree import plot_tree
best_precision = 0
best_features_precision = None
best_model = None 
for sample_size in range(1,12):
    for i in range(8500):  
        if i % 1000 == 0:
            print(f"{i} -> {best_precision}")
        selected_features = random.sample(list(feature_columns), sample_size)
        X_selected_train = df_filtered[selected_features]
        # Create a new instance for each iteration
        model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=7)
        model.fit(X_selected_train, y_filtered)
        y_pred = model.predict(X_selected_train)
        precision = precision_score(y_filtered, y_pred, zero_division=0)
        
        if precision > best_precision:
            best_precision = precision
            best_features_precision = selected_features
            best_model = model  # Save the model that achieved this precision

print("Best precision: ", best_precision)
print("Best features: ", best_features_precision)

# Display the tree using the best model and its features
plt.figure(figsize=(20, 10))
plot_tree(best_model, feature_names=best_features_precision, class_names=["-1", "1"], filled=True, rounded=True, fontsize=12)
plt.title("Decision Tree (0s removed) with best features")
plt.tight_layout() 
plt.show()
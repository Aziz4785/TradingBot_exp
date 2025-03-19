from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv(f"old_stuff_analysis/all_models_to_csv.csv")
df = df.drop_duplicates()
target_column = "good_model"

df = pd.get_dummies(df, columns=["Model_type", "Model"], prefix=["Model_type", "Model"])

feature_columns = [col for col in df.columns if col not in ["Ticker", target_column]]
df = df.dropna(subset=[target_column])  # Drop rows where target is NaN
df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors='coerce')
df = df.dropna()  # Drop remaining NaNs

X = df[feature_columns]
print("length of X : ",len(X))
print("colomns of X : ",X.columns)
y = df[target_column]

# Train Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=1)  # Limit depth for better interpretability
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
clf_bad = DecisionTreeClassifier(max_depth=2, min_samples_leaf=3)
clf_bad.fit(X_bad, y_bad)
plt.figure(figsize=(20, 10))
tree.plot_tree(clf_bad, feature_names=feature_columns, class_names=["-1", "1"], filled=True, rounded=True)
plt.title("Decision Tree (0s as BAD)")
plt.show()

# IF WE CONSIDER 0s as GOOD
df_good = df.copy()  # Make a copy for this approach
df_good["good_model"] = df_good["good_model"].replace(0, 1)
X_good = df_good[feature_columns]
y_good = df_good[target_column]
clf_good = DecisionTreeClassifier(max_depth=2, min_samples_leaf=3)
clf_good.fit(X_good, y_good)
plt.figure(figsize=(20, 10))
tree.plot_tree(clf_good, feature_names=feature_columns, class_names=["-1", "1"], filled=True, rounded=True)
plt.title("Decision Tree (0s as GOOD)")
plt.show()

# IF WE ONLY CONSIDER EXISTING 1s AND -1s (removing 0s completely)
df_filtered = df.copy()
# Filter out rows where the target column equals 0
df_filtered = df_filtered[df_filtered[target_column] != 0]
X_filtered = df_filtered[feature_columns]
y_filtered = df_filtered[target_column]
clf_filtered = DecisionTreeClassifier(max_depth=2, min_samples_leaf=3)
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
best_model = None  # Store the best model
for i in range(2500):  # Or however many iterations you need
    if i % 600 == 0:
        print(f"{i} -> {best_precision}")
    selected_features = random.sample(list(feature_columns), 3)
    X_selected_train = df_filtered[selected_features]
    # Create a new instance for each iteration
    model = DecisionTreeClassifier(max_depth=1, min_samples_leaf=5)
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
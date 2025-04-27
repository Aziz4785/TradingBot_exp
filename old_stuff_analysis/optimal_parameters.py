from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("old_stuff_analysis/balanced_targets_with0good.csv")
df.drop(columns=["Ticker"], inplace=True, errors='ignore')
target_column = 'good_model'
df.loc[df[target_column] == 0, target_column] = 1
#transform -1 to 0
df.loc[df[target_column] == -1, target_column] = 0
X = df.drop(columns=[target_column])
y = df[target_column]
features = X.columns.tolist()


#RAndom Forest Classifier
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None,3, 10, 20, 30],
    'min_samples_split': [2, 5, 7, 10],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': ['sqrt', 'log2', 0.3, 0.1]
}
#DECISION TREE CLASSIFIER
"""param_grid = {
    'max_depth': [None, 3, 5, 10, 15, 20, 25, 30],
    'min_samples_split': [2, 5, 7, 10],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': ['sqrt', 'log2', 0.3, 0.1, 0.5],
    'splitter': ['best', 'random']
}
#logistic regression
param_grid = {
    'C': [0.01, 0.1, 1, 4, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}"""

#cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#use regular kfold : 
cv = KFold(n_splits=5, shuffle=True)

model = RandomForestClassifier(random_state=42)
#model = DecisionTreeClassifier(random_state=42)
#model = LogisticRegression(max_iter=1000, random_state=42)
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',  # or 'roc_auc', 'f1', etc.
    n_jobs=-1,  # Use all available CPUs
    verbose=2
)

grid_search.fit(X, y)

print("Best parameters found:")
print(grid_search.best_params_)

print("\nBest cross-validation score:")
print(grid_search.best_score_)

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X)
print(classification_report(y, y_pred))
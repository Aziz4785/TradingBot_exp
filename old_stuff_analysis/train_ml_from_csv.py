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

# Load your CSV
df = pd.read_csv("old_stuff_analysis/balanced_targets_with0good.csv")

# Separate features and target
target_column = 'good_model'
df.loc[df[target_column] == 0, target_column] = 1
X = df.drop(columns=[target_column])
y = df[target_column]
features = X.columns.tolist()

# Define models
models = {
    "RandomForest": RandomForestClassifier(),
    "RandomForest_depth5": RandomForestClassifier(max_depth=5),
    "rf3": RandomForestClassifier(n_estimators=200,max_depth=3),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "LogisticRegression_C0.1": LogisticRegression(C=0.1, max_iter=1000),
    "SVC": make_pipeline(StandardScaler(), SVC()),  # Scale for SVM
    "GradientBoosting": GradientBoostingClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "DecisionTree_minSamples10": DecisionTreeClassifier(min_samples_split=6),
}

# Storage for best models
best_models = {}

# For each model
for name, model in models.items():
    print(f"\nTrying model: {name}")
    best_score = 0
    best_features = None
    best_model = None

    for i in range(1, 680): 
        #slect random number n from 1 to 20
        #select n random features from features
        n = random.randint(1, min(20, len(features)))  
        selected_features = random.sample(features, n)  
        #selected_features = random.sample(features, i)
        X_subset = X[selected_features]

        try:
            scores = cross_val_score(model, X_subset, y, cv=5, scoring='accuracy') #The data is split into 5 equal parts 
            mean_score = scores.mean()
            #print(f"  Features: {selected_features} | CV Score: {mean_score:.4f}")
            if mean_score<0.75:
                continue
            if mean_score > best_score:
                best_score = mean_score
                best_features = selected_features
                best_model = model
                print(f"  Features: {selected_features} | CV Score: {mean_score:.4f}")

        except Exception as e:
            print(f"  Skipped due to error: {e}")

    best_models[name] = {
        "score": best_score,
        "features": best_features,
        "model": best_model
    }

# Print the best models and their performance
print("\nBest models and feature subsets:")
for name, info in best_models.items():
    if info['score']>0.5:
        print(f"{name}: Score = {info['score']:.4f}, Features = {info['features']}")
        filename = f"meta_model/{name}_best_model.pkl"
        joblib.dump({
            "model": info["model"],
            "features": info["features"]
        }, filename)
        features_path = f"meta_model/{name}_features.json"
        with open(features_path, "w") as f:
            json.dump(info["features"], f)

        print(f"Saved {name} model to {filename}")
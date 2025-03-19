import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
# Load your CSV
df = pd.read_csv("INTRADAY/best_scores.csv")
df = df[df["final_val_acc"]>0.65]

#print highest :
top_10 = df.nlargest(10, 'final_test_acc')
print(top_10)


# Compute the correlation matrix
corr_matrix = df.corr()
target_corr = corr_matrix["final_test_acc"].drop("final_test_acc")
most_correlated_column = target_corr.abs().idxmax()
correlation_value = target_corr[most_correlated_column]
print(f"Most correlated column: {most_correlated_column} with a correlation of {correlation_value:.3f}")

#decisiontree:
# Separate features and target
df['Min_AB'] = df[['final_val_acc', 'final_test_acc']].min(axis=1)
X = df.drop(["final_test_acc","final_val_acc",'Min_AB'], axis=1)
y = df["Min_AB"]

tree = DecisionTreeRegressor(max_depth=2,min_samples_leaf=10)
tree.fit(X, y)
importances = tree.feature_importances_
# Create a DataFrame for easier viewing
feature_importances = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)
print(feature_importances)
plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=X.columns, filled=True, rounded=True)
plt.show()



#LINEAR REGRESSION : 

X = df.drop(["final_test_acc","final_val_acc",'Min_AB','depth'], axis=1)
y = df["Min_AB"]

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import r2_score
# Compute the correlation matrix
corr_matrix = X.corr()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
most_correlated_pair = upper_triangle.abs().stack().idxmax()
highest_correlation = upper_triangle.abs().stack().max()
print(f"The most correlated columns in X are: {most_correlated_pair} with correlation {highest_correlation:.2f}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LinearRegression()
model.fit(X_scaled, y)
coefficients = model.coef_
intercept = model.intercept_
print("Intercept (w.r.t. scaled features):", intercept)
print("Coefficients (w.r.t. scaled features):", coefficients)
feature_names = X.columns
for name, coef in zip(feature_names, coefficients):
    print(f"{name}: {coef}")
score = model.score(X_scaled, y)
print("R^2 score (training):", score)
# Now you have a linear equation of the form:
# final_test_acc = intercept + (coef_1 * x1) + (coef_2 * x2) + ...

from sklearn.preprocessing import PolynomialFeatures

# Suppose X_scaled is your scaled feature matrix
# We transform it to include polynomial terms of degree=2
poly = PolynomialFeatures(degree=3, interaction_only=False , include_bias=True)
X_poly = poly.fit_transform(X_scaled)
model_poly = LinearRegression()
model_poly.fit(X_poly, y)
score_poly = model_poly.score(X_poly, y)
print("R^2 score (training) with polynomial features:", score_poly)
if(score_poly<0.5):
    print(" Score is too low, please try with a polynomial with higher degree")


# Get the coefficients and intercept
coef = model_poly.coef_
intercept = model_poly.intercept_
feature_names = poly.get_feature_names_out(input_features=['x' + str(i) for i in range(X_scaled.shape[1])])
terms = [f"{coef[i]:.4f} * {feature_names[i]}" for i in range(len(coef))]
equation = " + ".join(terms)
equation = f"y = {intercept:.4f} + " + equation
print(equation)

#regression without cross term
# Create polynomial features for each column separately
degree = 4
X_poly_list = []

for i in range(X_scaled.shape[1]):
    # Extract single column
    X_col = X_scaled[:, i:i+1]
    
    # Create polynomial features for this column only
    poly_col = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly_col = poly_col.fit_transform(X_col)
    
    # Store the transformed column
    X_poly_list.append(X_poly_col)

# Add a bias column (all 1s)
X_poly_list.append(np.ones((X_scaled.shape[0], 1)))

# Combine all polynomial features
X_poly = np.hstack(X_poly_list)

# Fit the model
model_poly = LinearRegression(fit_intercept=False)  # No need for intercept as we included a bias column
model_poly.fit(X_poly, y)
score_poly = model_poly.score(X_poly, y)
print("R^2 score (training) with polynomial features:", score_poly)

if(score_poly < 0.5):
    print("Score is too low, please try with a polynomial with higher degree")

# Get the coefficients
coef = model_poly.coef_

# Create feature names manually
feature_names = []
for i in range(X_scaled.shape[1]):
    col_name = f'x{i}'
    for j in range(1, degree+1):
        feature_names.append(f'{col_name}^{j}')
feature_names.append('bias')

# Create the equation
terms = [f"{coef[i]:.4f} * {feature_names[i]}" for i in range(len(coef))]
equation = " + ".join(terms)
equation = "y = " + equation
print(equation)
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import stats
"""
to run : py -m experiment1.experiment3
"""
df = pd.read_csv('old_stuff_analysis/all_models_to_csv.csv')
df = df.drop_duplicates()
target_column = "good_model"

#df = pd.get_dummies(df, columns=["Model_type", "Model"], prefix=["Model_type", "Model"])
df.drop(columns=["Model_type", "Model"], inplace=True)
feature_columns = [col for col in df.columns if col not in ["Ticker", target_column]]
df = df.dropna(subset=[target_column])  # Drop rows where target is NaN
df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors='coerce')
print("length of df before dropping none  : ",len(df))
df = df.dropna()  # Drop remaining NaNs
print("length of df after dropping none  : ",len(df))

LEN_ORIGINAL_DF= len(df)

print("length of df : ",len(df))
percentage_target_1 = (df[target_column].value_counts(normalize=True).get(1, 0)) * 100
print(f"Percentage of rows where target == 1: {percentage_target_1:.2f}%")

def find_best_fit(vectorY):
    """
    Find whether the data fits better to a linear (y=ax+b) or quadratic (y=ax^2+bx+c) equation
    """
    # Create x vector
    vectorX = np.array(range(len(vectorY))).reshape(-1, 1)
    vectorY = np.array(vectorY)
    
    # Fit linear regression (y = ax + b)
    linear_reg = LinearRegression()
    linear_reg.fit(vectorX, vectorY)
    linear_pred = linear_reg.predict(vectorX)
    linear_score = r2_score(vectorY, linear_pred)
    n = len(vectorY)


    # Fit quadratic regression (y = ax² + bx + c)
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(vectorX)
    quad_reg = LinearRegression()
    quad_reg.fit(X_poly, vectorY)
    quad_pred = quad_reg.predict(X_poly)
    quad_score = r2_score(vectorY, quad_pred)
    
    if max(quad_score,linear_score)<=0.7:
        return max(quad_score,linear_score),"",0
    # Compare scores and return result
    if linear_score > quad_score:
        return linear_score,{
            'best_fit': 'linear',
            'score': linear_score,
            'equation': f'y = {linear_reg.coef_[0]}x + {linear_reg.intercept_}'
        },[linear_reg.coef_[0]]
    else:
        return quad_score,{
            'best_fit': 'quadratic',
            'score': quad_score,
            'equation': f'y = {quad_reg.coef_[2]}x² + {quad_reg.coef_[1]}x + {quad_reg.intercept_}'
        },[quad_reg.coef_[2],quad_reg.coef_[1]]

def add_pos_neg_columns(df,columns_to_skip):
    
    # Create a copy of the dataframe to avoid modifying the original
    df_modified = df.copy()
    
    for column in df.columns:
        if column not in columns_to_skip:
            # Check if column contains any negative values
            if (df[column] < 0).any():
                # Create positive values column (>=0)
                df_modified[f'{column}_pos'] = df[column].where(df[column] >= 0)
                
                # Create negative values column (<0)
                df_modified[f'{column}_neg'] = df[column].where(df[column] < 0)
    
    return df_modified
 
def print_percentiles(df):
    # Skip specified columns
    columns_to_skip = ['to_buy', 'date', 'stock', 'Ticker', target_column,'Model','Model_type']
    useful_columns= set()
    if len(df)<=50:
        print("the df is small cannot proceed...")
        return
    for column in df.columns:
        if column not in columns_to_skip:
            unique_vals = set(df[column].dropna().unique())
            # Check if the column is binary (only contains 0 and 1)
            if unique_vals == {0, 1}:
                print(f"Column: {column}")
                 # Filter the rows where the binary column is 1 
                subset = df[df[column] == 1]
                # Calculate the percentage of rows where TARGET is 1 (ignoring -1 and 0)
                percentage_target_ones = (subset[target_column] == 1).mean() * 100
                print(f"If we keep only rows where {column} is 1, the percentage of 1s in TARGET is {percentage_target_ones:.2f}%")
            if df[column].nunique() < 4:
                continue
            print(f"{column}:")
            #print(f"  10th percentile: {p15:.2f}")
            #print(f"  90th percentile: {p85:.2f}")

            # Filter values between p15 and p85
            #filtered_values = df[column][(df[column] >= p15) & (df[column] <= p85)]
            filtered_values = df[column]
            # Calculate segments using percentiles
            segments = [0, 33.33,66.66, 100]
            #segments = [0, 25, 50, 75, 100]
            #segments = [0, 20, 40, 60, 80, 100 ]
            #segments = [0, 25, 50, 75, 100]
            #segments = [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100]
            #segments = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            segment_values = [filtered_values.quantile(p/100) for p in segments]

            # Create list of valid segments (where boundaries are different)
            valid_segments = []
            valid_segment_indices = []
            for i in range(len(segments)-1):
                if segment_values[i] != segment_values[i+1]:
                    valid_segments.append((segment_values[i], segment_values[i+1]))
                    valid_segment_indices.append(i)
            
            if valid_segments:
                if len(valid_segments)>0 and len(filtered_values)/len(valid_segments)<2: #number of values per segment
                    print("too few values to perform analysis")
                    print("valid segments : ",valid_segments)
                    print("filtered_values : ",filtered_values)
                    continue
                
                #print("\n  Segments boundaries:")
                #for idx, (left, right) in enumerate(valid_segments, 1):
                    #print(f"    Segment {idx}: {left:.2f} to {right:.2f}")
                
                vectorY = []
                #print("\n  Segments statistics:")
                max_ratio=0
                min_ratio =100
                for idx, (left, right) in enumerate(valid_segments, 1):
                    # Get mask for values in this segment
                    segment_mask = (df[column] >= left) & (df[column] <= right)
                    
                    # Count total elements in segment
                    total_in_segment = segment_mask.sum()
                    
                    # Count elements with target=1 in segment
                    target_in_segment = ((df[target_column] == 1) & segment_mask).sum()
                    
                    # Calculate ratio (handle division by zero)
                    ratio = (target_in_segment / total_in_segment) if total_in_segment > 0 else 0
                    max_ratio = max(max_ratio,ratio)
                    min_ratio= min(min_ratio,ratio)
                    vectorY.append(ratio)
                    # print(f"    Segment {idx}:")
                    # print(f"      Total elements: {total_in_segment}")
                    # print(f"      Elements with target=1: {target_in_segment}")
                    # print(f"      Target ratio: {ratio:.2%}")

                

                score,result,coeffs = find_best_fit(vectorY)
                #vectorY is the vector of ratios
                #valid_segments is the vector of segments
                almost_monotone=True
                if score >0.9:
                    if result['best_fit']=='linear':
                        almost_monotone=True
                    elif result['best_fit']=='quadratic':
                        #print(f"{column}:")
                        a = coeffs[0]
                        b = coeffs[1]
                        optimum = -b/(2*a)
                        #print("optimum : ",optimum)
                        if optimum>0 and optimum<len(vectorY)-1:
                            if len(vectorY)%2==1:
                                middle_segment_lower_bound = (len(vectorY)-1)/2 -1
                                middle_segment_upper_bound =middle_segment_lower_bound+2
                            else:
                                middle_segment_lower_bound = min(2,len(vectorY)/2-1)
                                middle_segment_upper_bound =max(len(vectorY)-3,2)
                            
                            if optimum>=middle_segment_lower_bound and optimum<=middle_segment_upper_bound:
                                almost_monotone=False
                            #print("middle_segment_lower_bound : ",middle_segment_lower_bound)
                            #print("middle_segment_upper_bound : ",middle_segment_upper_bound)
                            


                if score >0.8:
                    print()
                    print(f"{column}:")
                    print(score)
                    useful_columns.add(column)
                    if len(valid_segments)>0 and len(filtered_values)/len(valid_segments)<250:
                        print("(leaf node)")
                    print(f"we divide the values of {column} into equal sized segments:")
                    for idx, (left, right) in enumerate(valid_segments, 1):
                        print(f"    Segment {idx}: {left:.2f} to {right:.2f}")
                    print("then for each segment we calculate the proportion of target==1 : ")
                    print(vectorY)
                    print(result['equation'])
                    

            else:
                print("\n  No valid segments found (all boundaries are equal)")
    print("columns to consider : ")
    print(useful_columns)

print_percentiles(df)

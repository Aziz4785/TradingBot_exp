from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,HistGradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from xgboost import XGBClassifier
import json
from sklearn.preprocessing import LabelEncoder
from utils import * 
from SuperModel import * 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def neutralize(
    df: pd.DataFrame,
    neutralizers: np.ndarray,
    proportion: float = 1.0,
) -> pd.DataFrame:
    """Neutralize each column of a given DataFrame by each feature in a given
    neutralizers DataFrame. Neutralization uses least-squares regression to
    find the orthogonal projection of each column onto the neutralizers, then
    subtracts the result from the original predictions.

    Arguments:
        df: pd.DataFrame - the data with columns to neutralize
        neutralizers: pd.DataFrame - the neutralizer data with features as columns
        proportion: float - the degree to which neutralization occurs

    Returns:
        pd.DataFrame - the neutralized data
    """
    assert not neutralizers.isna().any().any(), "Neutralizers contain NaNs"
    assert len(df.index) == len(neutralizers.index), "Indices don't match"
    assert (df.index == neutralizers.index).all(), "Indices don't match"
    df[df.columns[df.std() == 0]] = np.nan
    df_arr = df.values
    neutralizer_arr = neutralizers.values
    neutralizer_arr = np.hstack(
        # add a column of 1s to the neutralizer array in case neutralizer_arr is a single column
        (neutralizer_arr, np.array([1] * len(neutralizer_arr)).reshape(-1, 1))
    )
    inverse_neutralizers = np.linalg.pinv(neutralizer_arr, rcond=1e-6)
    adjustments = proportion * neutralizer_arr.dot(inverse_neutralizers.dot(df_arr))
    neutral = df_arr - adjustments
    return pd.DataFrame(neutral, index=df.index, columns=df.columns)

df = pd.read_csv('old_stuff2/clean.csv')
df = df.sample(frac=1).reset_index(drop=True)
df.drop_duplicates(inplace=True)
df = df.dropna()
MAIN_TICKER= 'AAPL'
TARGET_COLUMN = 'to_buy_1d'
df = df[df['Stock']==MAIN_TICKER]
df['Date'] = pd.to_datetime(df['Date'])
#df['era'] = df['Date'].dt.date 
models_dir  = "old_stuff2/allmodels"
scalers_dir = "old_stuff2/allscalers"
config_path = "old_stuff2/models_to_use.json"
with open(config_path, 'r') as file:
    config = json.load(file)

feature_set = extract_features_of_stock(config,MAIN_TICKER)
loaded_bin_dict_json = load_bins("old_stuff2/bins_json.json")
super_model = SuperModel(config_path, models_dir, scalers_dir)
print("1")
today = pd.to_datetime("today").normalize()
test_start   = today - pd.Timedelta(days=45)
test_end = today

df_validation= df[(df["Date"] >= test_start) & (df["Date"] < test_end)]
probabilities = super_model.predict_proba(MAIN_TICKER, df_validation[feature_set])
df_validation["prediction_prob"] = probabilities[:, 1]

print("2")
BIGGER_ERA = False #set to false
GROUPPING = True #set to true
if BIGGER_ERA:
    df_validation['week'] = df_validation['Date'].dt.to_period('W').dt.start_time
    # or do something like to_period('M') for monthly
    df_validation["neutralized"] = (
        df_validation.groupby("week", group_keys=True)
        .apply(lambda d: neutralize(
            d[["prediction_prob"]],
            d[feature_set],
            proportion=1.0
        ))
        .reset_index(level=0, drop=True)
    )
else:
    
    if GROUPPING:
        df_validation["neutralized"] = (
            df_validation.groupby("Date", group_keys=True)
            .apply(lambda d: neutralize(
                d[["prediction_prob"]],  # your predicted probabilities
                d[feature_set],    # the same features you want to neutralize against
                proportion=0.12
            ))
            .reset_index(level=0, drop=True)
        )
    else:
        df_validation["neutralized"] = neutralize(
            df_validation[["prediction_prob"]],
            df_validation[feature_set],
            proportion=0.1
        )

def compute_corr(sub_df):
    # Compute correlation between each feature and target in this era
    return sub_df[feature_set].corrwith(sub_df[TARGET_COLUMN])
df_validation = df_validation.sort_values(by="Date")
#df_validation["era"] = df_validation["Date"].dt.date  
df_validation["era"] = df_validation["Date"].dt.to_period("W").astype(str)
per_era_corr = df_validation.groupby("era").apply(compute_corr)
print(per_era_corr.head(20))
print(per_era_corr.tail(20))

cumsum_corr = per_era_corr.cumsum()
plt.figure(figsize=(15, 5))
for col in cumsum_corr.columns:
    plt.plot(cumsum_corr.index, cumsum_corr[col], label=col)
plt.title("Cumulative Absolute Value CORR of Features and the Target")
plt.xlabel("Era ")
plt.legend()
plt.show()
exit()
# Create prediction_res column
df_validation['prediction_res'] = (df_validation['prediction_prob'] > 0.5).astype(int)

# Create neutralized_res column
df_validation['neutralized_res'] = (df_validation['neutralized'] > 0.5).astype(int)

y_pred_from_model = super_model.predict_multiple(df_validation)
y_pred_from_proba = df_validation['prediction_res']
y_pred_neutr = df_validation['neutralized_res']
y_test = df_validation[TARGET_COLUMN]
precision_mdl,recall_mdl,specificity_mdl,f05_mdl,mcc_mdl = calculate_all_metrics(y_test,y_pred_from_model)
precision_prb,recall_prb,specificity_prb,f05_prb,mcc_prb = calculate_all_metrics(y_test,y_pred_from_proba)
precision_neutr,recall_neutr,specificity_neutr,f05_neutr,mcc_neutr = calculate_all_metrics(y_test,y_pred_neutr)

print("precision , recall , specificity (model): ")
print(f"{precision_mdl}  , {recall_mdl}  , {specificity_mdl}")
print("precision , recall , specificity (proba) (should be the same as from model): ")
print(f"{precision_prb}  , {recall_prb}  , {specificity_prb}")
print("precision , recall , specificity (neutralized): ")
print(f"{precision_neutr}  , {recall_neutr} , {specificity_neutr}")

df_validation[['Stock','Date','Close','to_buy_1d','prediction_prob','neutralized','prediction_res','neutralized_res']].to_csv('after_neutralization.csv', encoding='utf-8', index=False)
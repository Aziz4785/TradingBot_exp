import json
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, fbeta_score, matthews_corrcoef, confusion_matrix, accuracy_score, precision_recall_curve
import pandas as pd
import numpy as np

class SuperModel:
    def __init__(self, config_path, models_dir, scalers_dir):
        """
        Initialize the SuperModel.
        
        Parameters:
            config_path (str): Path to the JSON configuration file.
            models_dir (str): Directory where all the model pickle files are stored.
        """
        self.config = self._load_config(config_path)
        self.models_dir = models_dir
        self.loaded_models = self._load_models()
        self.scalers_dir = scalers_dir
        self.loaded_scalers = self._load_scalers()

    def _load_config(self, config_path):
        """Load the JSON configuration file."""
        with open(config_path, "r") as f:
            config = json.load(f)
        return config

    def _load_scalers(self):
        loaded_scalers = {}
        for stock, info in self.config.items():
            features_id = info["subset_id"]
            scaler_file = f"scaler_{features_id}_{stock}.pkl"
            scaler_path = os.path.join(self.scalers_dir, scaler_file)
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"scaler file {scaler_path} not found for stock {stock}.")
            
            with open(scaler_path, "rb") as mf:
                loaded_scalers[stock] = pickle.load(mf)
        return loaded_scalers
    
    def _load_models(self):
        """Load all models from the models directory based on the configuration."""
        loaded_models = {}
        for stock, info in self.config.items():
            model_name = info["best_model"]
            model_file = model_name + ".pkl"
            model_path = os.path.join(self.models_dir, model_file)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file {model_path} not found for stock {stock}.")
            
            with open(model_path, "rb") as mf:
                loaded_models[stock] = pickle.load(mf)
        return loaded_models

    def predict(self, stock, input_data):
        """
        Predict using the model corresponding to the given stock.
        
        Parameters:
            stock (str): Stock symbol (e.g., "MSFT").
            input_data (dict): Dictionary containing feature values for prediction.
            
        Returns:
            The prediction result from the corresponding model.
            
        Raises:
            ValueError: If the stock is not found in the configuration or if any required features are missing.
        """
        if stock not in self.config:
            return 0  # Or consider raising an exception

        # Get the list of required features from the configuration.
        required_features = self.config[stock]["subset"]
        
        # Build the input list and check for missing features.
        model_input = []
        missing_features = []
        for feature in required_features:
            if feature not in input_data or input_data[feature] is None:
                missing_features.append(feature)
            else:
                model_input.append(input_data[feature])
                
        if missing_features:
            raise ValueError(f"Missing features for {stock}: {missing_features}")
        
        # Convert the input list to a 2D array.
        
        model_input_array = np.array(model_input).reshape(1, -1)
        
        # If the scaler was fitted using a DataFrame with column names, create a DataFrame here:
        input_df = pd.DataFrame(model_input_array, columns=required_features)
        
        # Retrieve the scaler and transform the input.
        scaler = self.loaded_scalers[stock]
        model_input_scaled = scaler.transform(input_df)
        
        # Retrieve the model and predict.
        model = self.loaded_models[stock]
        prediction = model.predict(model_input_scaled)
        
        return prediction[0]

    def predict_multiple(self, df):
        predictions = []
        for index, row in df.iterrows():
            # Extract the stock symbol from the row.
            if "Stock" not in row:
                raise ValueError(f"Row {index} is missing the 'stock' column.")
            stock = row["Stock"]
            
            # Convert the row to a dictionary.
            # If you want to remove the 'stock' key from the input_data, you can do so.
            input_data = row.to_dict()
            #input_data.pop("stock", None)  # Remove the 'stock' key if present
            
            # Call the existing predict method.
            pred = self.predict(stock, input_data)
            predictions.append(pred)
        
        return predictions

    def predict_proba(self, stock, df):
        """
        Compute prediction probabilities (for models that support predict_proba)
        for a given stock and a DataFrame `df` of feature values.
        
        Parameters:
            stock (str): Stock symbol (e.g., "AMZN").
            df (pd.DataFrame): DataFrame containing rows of feature values.
                               Must contain all features in self.config[stock]["subset"].
        
        Returns:
            np.ndarray: The predicted probabilities for each row (shape: [n_samples, n_classes]).
        
        Raises:
            ValueError: If the stock is not found or required features are missing.
            AttributeError: If the loaded model does not support predict_proba.
        """
        if stock not in self.config:
            raise ValueError(f"Stock {stock} not found in configuration.")

        # Extract required features
        required_features = self.config[stock]["subset"]
        
        # Check for missing columns
        missing_cols = [feat for feat in required_features if feat not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in input DataFrame for {stock}: {missing_cols}")
        
        # Reorder and select only required features
        X = df[required_features].copy()
        
        # Scale
        scaler = self.loaded_scalers[stock]
        X_scaled = scaler.transform(X)
        
        # Predict probabilities
        model = self.loaded_models[stock]
        
        # Make sure the model actually supports predict_proba
        if not hasattr(model, "predict_proba"):
            raise AttributeError(
                f"The model for stock {stock} does not support predict_proba."
            )
        
        probabilities = model.predict_proba(X_scaled)
        return probabilities
    
    def predict_proba_multiple(self, df):
        """
        Compute prediction probabilities for all rows in the DataFrame.
        It groups the rows by 'Stock' and calls predict_proba for each stock.

        Parameters:
            df (pd.DataFrame): DataFrame containing a "Stock" column and all required features.

        Returns:
            pd.Series: A Series with the prediction probabilities for each row.
                    Each entry is the array of probabilities returned by predict_proba.
        """
        # Create an empty Series to store probabilities, preserving the original index.
        probabilities = pd.Series(index=df.index, dtype=object)
        
        # Group the DataFrame by the stock symbol
        for stock, group in df.groupby("Stock"):
            # Call your existing predict_proba for the group of rows corresponding to the stock.
            proba = self.predict_proba(stock, group)
            # Assign the probabilities (converted to list if needed) back to the corresponding indices.
            probabilities.loc[group.index] = list(proba)
        
        return probabilities
    
def calculate_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    
    precision = precision_score(y_test, y_pred) #Measures how many predicted "1"s are correct.
    recall = recall_score(y_test, y_pred) #Measures how many actual "1"s are captured
    # F0.5 Score (places more weight on precision)
    f05 = fbeta_score(y_test, y_pred, beta=0.5) #Harmonic mean of precision and recall
    mcc = matthews_corrcoef(y_test, y_pred)

    return accuracy,specificity,precision,recall,f05,mcc
# Example usage:
if __name__ == "__main__":
    # Paths to the configuration file and models directory.
    config_path = "config.json"
    models_dir = "allmodels"
    scalers_dir= "allscalers"
    # Instantiate the SuperModel
    super_model = SuperModel(config_path, models_dir,scalers_dir)
    
    # Example input data for stock MSFT. Make sure to include all features required by MSFT.
    input_data_example = {
        "PM_max_to_dayOpen_ratio_class": 0.5,
        "PM_min_to_Close_ratio_class": 0.3,
        "PM_max_to_Close_ratio_class": 0.45,
        "dayOpen_to_prev2DayOpen_ratio_class": 0.6,
        "day_of_week": 3,
        "AH_max_1dayago_vs_prevDayClose_class": 0.2,
        "PM_range_to_close_ratio_class": 0.55,
        "PM_max_to_PM_min_ratio_class": 1.2,
        "AH_max_1dayago_vs_PM_max_class": 0.75,
        "PM_min_to_prevDayClose_ratio_class": 0.4,
        # ... add any additional features if needed.
    }
    
    try:
        prediction = super_model.predict("MSFT", input_data_example)
        print("Prediction for MSFT:", prediction)
    except Exception as e:
        print("Error during prediction:", e)

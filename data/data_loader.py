# data/data_loader.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from app.config import DATA_PATH, FEATURES, TARGET, TEST_SIZE, RANDOM_STATE

def load_data(file_path='data/patient_data.csv'):
    """
    Load data from CSV file
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully with {len(df)} rows")  # Debug print
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """
    Preprocess the data and split into features and target
    Returns:
        X_train, X_test, y_train, y_test, feature_names, scaler
    """
    print("Starting preprocessing...")  # Debug print
    
    if df is None:
        print("DataFrame is None")  # Debug print
        return None, None, None, None, None, None
    
    try:
        # Separate features and target
        X = df.drop('readmitted', axis=1)
        y = df['readmitted']
        
        # Get feature names
        feature_names = X.columns.tolist()
        print(f"Features: {feature_names}")  # Debug print
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_names)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("Preprocessing completed successfully")  # Debug print
        
        # Explicitly create the return tuple
        result = (X_train, X_test, y_train, y_test, feature_names, scaler)
        print(f"Number of return values: {len(result)}")  # Debug print
        return result
    
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None, None, None, None, None, None
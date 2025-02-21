# models/model.py
from sklearn.ensemble import RandomForestClassifier
import logging
from app.config import MODEL_PARAMS
import pandas as pd

def train_model(X_train, y_train):
    """Train the Random Forest model"""
    try:
        model = RandomForestClassifier(**MODEL_PARAMS)
        # Ensure X_train is a DataFrame
        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("X_train must be a pandas DataFrame with feature names")
        model.fit(X_train, y_train)
        logging.info("Model trained successfully")
        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        return None

def predict(model, X):
    """Make predictions using trained model"""
    try:
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame with feature names")
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        return predictions, probabilities
    except Exception as e:
        logging.error(f"Error making predictions: {e}")
        return None, None
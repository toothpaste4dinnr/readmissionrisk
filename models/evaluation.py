# models/evaluation.py
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import logging

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate model performance metrics"""
    try:
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc_roc': roc_auc_score(y_true, y_pred_proba)
        }
        return metrics
    except Exception as e:
        logging.error(f"Error calculating metrics: {e}")
        return None
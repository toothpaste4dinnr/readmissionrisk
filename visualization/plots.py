# visualization/plots.py
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import logging
import pandas as pd
import numpy as np

def plot_feature_importance(model, X):
    """Create feature importance plot"""
    try:
        importance = model.feature_importances_
        feat_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        })
        feat_importance = feat_importance.sort_values('importance', ascending=True)
        
        fig = px.bar(
            feat_importance,
            x='importance',
            y='feature',
            title='Feature Importance',
            orientation='h'
        )
        
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            title_x=0.5,
            xaxis_title="Relative Importance",
            yaxis_title="Features"
        )
        
        return fig
    except Exception as e:
        logging.error(f"Error plotting feature importance: {e}")
        return None

def plot_confusion_matrix(y_true, y_pred):
    """Create confusion matrix plot"""
    try:
        cm = confusion_matrix(y_true, y_pred)
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual"),
            x=['Not Readmitted', 'Readmitted'],
            y=['Not Readmitted', 'Readmitted'],
            title="Confusion Matrix"
        )
        return fig
    except Exception as e:
        logging.error(f"Error plotting confusion matrix: {e}")
        return None

def generate_shap_plots(model, X_test_df):
    """Generate SHAP plots"""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_df)
        
        # Create figure with even smaller size
        plt.figure(figsize=(6, 4))
        
        # Generate summary plot with smaller size and font
        shap.summary_plot(
            shap_values,
            X_test_df,
            show=False,
            plot_size=(6, 4),
            max_display=10,  # Limit number of features shown
            plot_type="bar"  # Use bar plot for more compact display
        )
        
        # Adjust layout and fonts
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.xlabel("SHAP value (impact on model output)", fontsize=8)
        
        # Adjust layout to prevent cutoff
        plt.tight_layout()
        
        # Get the current figure
        fig = plt.gcf()
        
        return fig
    except Exception as e:
        logging.error(f"Error generating SHAP plots: {e}")
        return None
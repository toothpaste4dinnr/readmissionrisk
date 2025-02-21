import streamlit as st
import numpy as np
import pandas as pd
import logging
from data.data_loader import load_data, preprocess_data
from visualization.plots import generate_shap_plots, plot_feature_importance
from models.model import train_model, predict
from models.evaluation import calculate_metrics

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

def is_valid_number(value):
    """Validate if input string is a valid number"""
    try:
        float(value)
        return True
    except ValueError:
        return False

def main():
    st.set_page_config(
        page_title="30-Day Readmission Risk Predictor",
        layout="wide"
    )
    
    setup_logging()
    
    st.title("30-Day Mental Health Hospital Readmission Risk Predictor")
    
    st.write("""
    This application predicts the risk of patient readmission within 30 days
    of discharge from a mental health hospital using machine learning.
    """)
    
    try:
        # Load and preprocess data
        data = load_data()
        if data is None:
            st.error("Error loading data")
            return
            
        X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data(data)
        if X_train is None:
            st.error("Error preprocessing data")
            return
        
        # Ensure X_train and X_test are DataFrames with feature names
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        
        # Train model
        model = train_model(X_train_df, y_train)
        if model is None:
            st.error("Error training model")
            return
            
        # Make predictions
        y_pred, y_pred_proba = predict(model, X_test_df)
        if y_pred is None:
            st.error("Error making predictions")
            return
            
        # Calculate and display metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        if metrics:
            st.header("Model Performance Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.2f}")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.2f}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.2f}")
            with col4:
                st.metric("F1 Score", f"{metrics['f1']:.2f}")
            with col5:
                st.metric("AUC-ROC", f"{metrics['auc_roc']:.2f}")
        
        # Define categorical features and their options
        categorical_features = {
            'age': ['18-25', '26-35', '36-45', '46-55', '56-65', '65+'],
            'length_of_stay': ['1-3 days', '4-7 days', '8-14 days', '15+ days']
        }

        # Define numerical features and their ranges
        numerical_features = {
            'previous_admissions': {'min': 0, 'max': 100},
            'num_procedures': {'min': 0, 'max': 50},
            'num_medications': {'min': 0, 'max': 100},
            'num_diagnoses': {'min': 0, 'max': 50},
            'num_lab_procedures': {'min': 0, 'max': 200},
            'num_outpatient': {'min': 0, 'max': 100},
            'num_emergency': {'min': 0, 'max': 100},
            'num_inpatient': {'min': 0, 'max': 100}
        }

        # Interactive Prediction
        st.header("Predict Readmission Risk")
        st.write("Enter patient information to predict readmission risk")
        
        # Create columns for input fields
        col1, col2 = st.columns(2)
        
        # Store all input values
        input_values = {}
        
        # First column of inputs
        with col1:
            # Handle categorical features
            for feature in categorical_features:
                input_values[feature] = st.selectbox(
                    f"{feature.replace('_', ' ').title()}",
                    options=categorical_features[feature],
                    key=f"select_{feature}"
                )
            
            # Handle first half of numerical features
            numerical_features_list = list(numerical_features.keys())
            for feature in numerical_features_list[:len(numerical_features_list)//2]:
                input_text = st.text_input(
                    f"{feature.replace('_', ' ').title()}",
                    value="0",
                    key=f"input_{feature}"
                )
                
                if not is_valid_number(input_text):
                    st.error(f"Please enter a valid number for {feature}")
                    input_values[feature] = 0
                else:
                    value = float(input_text)
                    if value < numerical_features[feature]['min'] or value > numerical_features[feature]['max']:
                        st.warning(f"Value should be between {numerical_features[feature]['min']} and {numerical_features[feature]['max']}")
                    input_values[feature] = value

        # Second column of inputs
        with col2:
            # Handle second half of numerical features
            for feature in numerical_features_list[len(numerical_features_list)//2:]:
                input_text = st.text_input(
                    f"{feature.replace('_', ' ').title()}",
                    value="0",
                    key=f"input_{feature}"
                )
                
                if not is_valid_number(input_text):
                    st.error(f"Please enter a valid number for {feature}")
                    input_values[feature] = 0
                else:
                    value = float(input_text)
                    if value < numerical_features[feature]['min'] or value > numerical_features[feature]['max']:
                        st.warning(f"Value should be between {numerical_features[feature]['min']} and {numerical_features[feature]['max']}")
                    input_values[feature] = value

        # Prediction button
        if st.button("Predict"):
            # Convert categorical inputs to numerical
            input_data = input_values.copy()
            
            # Process age categories
            age_mapping = {'18-25': 21.5, '26-35': 30.5, '36-45': 40.5, 
                          '46-55': 50.5, '56-65': 60.5, '65+': 70}
            input_data['age'] = age_mapping[input_data['age']]
            
            # Process length of stay categories
            los_mapping = {'1-3 days': 2, '4-7 days': 5.5, '8-14 days': 11, 
                          '15+ days': 15}
            input_data['length_of_stay'] = los_mapping[input_data['length_of_stay']]
            
            # Create DataFrame and make prediction
            input_df = pd.DataFrame([input_data], columns=feature_names)
            input_scaled = pd.DataFrame(
                scaler.transform(input_df),
                columns=feature_names
            )
            
            _, probabilities = predict(model, input_scaled)
            
            if probabilities is not None:
                prediction = probabilities[0]
                st.write(f"Predicted probability of readmission: {prediction:.2%}")
                
                risk_category = (
                    "High" if prediction > 0.7
                    else "Medium" if prediction > 0.3
                    else "Low"
                )
                st.write(f"Risk Category: {risk_category}")
        
        # Model Insights Section
        st.header("Model Insights")
        
        # Feature Importance Plot
        st.subheader("Feature Importance")
        st.write("""
        This plot shows the relative importance of each feature in making predictions.
        Higher values indicate more important features.
        """)
        fig_importance = plot_feature_importance(model, X_train_df)
        if fig_importance:
            st.plotly_chart(fig_importance)
        
        # SHAP Analysis
        st.subheader("SHAP Analysis")
        st.write("""
        SHAP (SHapley Additive exPlanations) values show how each feature
        contributes to predictions. Features in red increase the prediction,
        while features in blue decrease it.
        """)
        
        # Create and display SHAP plot with custom width
        fig_shap = generate_shap_plots(model, X_test_df)
        if fig_shap:
            # Use a container with custom width
            with st.container():
                st.pyplot(fig_shap, use_container_width=False)
        else:
            st.error("Error generating SHAP plot")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 
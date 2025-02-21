# README.md
# Mental Health Readmission Risk Predictor

This application predicts 30-day readmission risk for mental health hospital patients using machine learning.

## Setup
1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install requirements: `pip install -r requirements.txt`
5. Place your patient data CSV in the `data` folder
6. Run the app: `streamlit run app/main.py`

## Project Structure
- `app/`: Main Streamlit application
- `data/`: Data loading and preprocessing
- `models/`: ML model implementation
- `visualization/`: Plotting and visualization
- `utils/`: Utility functions

## Features
- Machine learning-based readmission risk prediction
- Comprehensive model metrics
- Interactive visualizations
- Model explainability using SHAP
- Feature importance analysis

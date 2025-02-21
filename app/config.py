# app/config.py
import os

# Data settings
DATA_PATH = os.path.join('data', 'patient_data.csv')
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Model settings
MODEL_PARAMS = {
    'n_estimators': 100,
    'random_state': RANDOM_STATE
}

# Feature settings
FEATURES = [
    'age',
    'length_of_stay',
    'previous_admissions',
    'diagnosis',
    'medication_count',
    'insurance_type',
    'discharge_disposition'
]

TARGET = 'readmitted_30_days'
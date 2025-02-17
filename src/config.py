# IMPORT LIBRARIES
import os
import pathlib

# FILE PATHS
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "raw_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "lg_model.pkl")

# MODEL HYPERPARAMETERS
MODEL_PARAMS = {
    'penalty': ['elasticnet'],
    'solver': ['saga'],
    'l1_ratio': [0.1, 0.5, 0.7, 1.0],
    'C': [0.01, 0.1, 1.0, 10.0],
    'max_iter': [1000]
}

# DATA COLUMNS
FEATURE_COLUMNS = [
    'Daily Internet Usage', 'Daily Time Spent on Site', 'Area Income',
    'Age', 'Male', 'Country'
]
TARGET_COLUMN = "Clicked on Ad"

# FINAL FEATURE NAMES (AFTER TRANSFORMATIONS)
FINAL_FEATURE_NAMES = [
    "Daily Internet Usage", "Daily Time Spent on Site", "Age", 
    "Male", "Area Income Square", "North America", "Asia"  
]
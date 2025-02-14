import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from src.preprocessing import scale_features
from src.model_handling import load_model
import src.config as config

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

def get_user_input():
    """Collect user input for predictions."""
    print("Enter the following details:")

    daily_internet_usage = float(input("Daily Internet Usage: "))
    daily_time_spent = float(input("Daily Time Spent on Site: "))
    area_income = float(input("Area Income: "))
    age = int(input("Age: "))
    male = int(input("Male (1 for Yes, 0 for No): "))
    north_america = int(input("North America (1 for Yes, 0 for No): "))
    asia = int(input("Asia (1 for Yes, 0 for No): "))

    # Apply feature transformation (ensure this matches training preprocessing)
    area_income_square = area_income ** 2

    # Ensure feature order matches training data
    features = pd.DataFrame([[daily_internet_usage, daily_time_spent, age, male, area_income_square, north_america, asia]], 
                            columns=config.FINAL_FEATURE_NAMES)
    return features

def make_prediction():
    """Load model and scaler, then make a prediction."""
    model, scaler = load_model()
    features = get_user_input()

    # Scale features using the saved scaler (avoiding warning by keeping feature names)
    features_scaled = scaler.transform(features)

    prob_class_0 = round(float(model.predict_proba(features_scaled)[0][0] * 100), 2)
    prob_class_1 = round(float(model.predict_proba(features_scaled)[0][1] * 100), 2)

    print(f"\033[1m- Probability of No Click:\033[0m {prob_class_0}%")
    print(f"\033[1m- Probability of Click:\033[0m {prob_class_1}%")

if __name__ == "__main__":
    make_prediction()


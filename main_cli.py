# IMPORT LIBRARIES
import os
import sys
from pathlib import Path

# SETTING UP PROJECT ROOT AND IMPORTS
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from src.training_cli import run_pipeline
from src.model_handling import load_model
from src.predict_cli import get_user_input

# MAIN FUNCTION TO RUN THE PIPELINE AND MAKE PREDICTIONS
def main():
    print("Running training pipeline...")
    run_pipeline()
    print("Model training complete! Model saved.")
    
    print("\nNow, let's make predictions with new data.")
    
    # LOAD TRAINED MODEL AND SCALER
    model, scaler = load_model()
    
    # GET USER INPUT FOR PREDICTIONS
    features = get_user_input()
    
    # SCALE FEATURES USING THE SAVED SCALER (WITHOUT FITTING AGAIN)
    features_scaled = scaler.transform(features)
    
    # MAKE PREDICTIONS
    prob_class_0 = round(float(model.predict_proba(features_scaled)[0][0] * 100), 2)
    prob_class_1 = round(float(model.predict_proba(features_scaled)[0][1] * 100), 2)
    
    # DISPLAY PREDICTION RESULTS
    print(f"\033[1m- Probability of No Click:\033[0m {prob_class_0}%")
    print(f"\033[1m- Probability of Click:\033[0m {prob_class_1}%")

# ENTRY POINT FOR SCRIPT EXECUTION
if __name__ == "__main__":
    main()





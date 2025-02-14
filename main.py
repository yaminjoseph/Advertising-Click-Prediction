import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from src.pipeline import run_pipeline
from src.model_handling import load_model
from src.predict import get_user_input

def main():
    print("Running training pipeline...")
    run_pipeline()
    print("Model training complete! Model saved.")
    
    print("\nNow, let's make predictions with new data.")
    
    # Load the trained model and scaler
    model, scaler = load_model()
    
    # Get user input
    features = get_user_input()
    
    # Use the saved scaler (without fitting again)
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prob_class_0 = round(float(model.predict_proba(features_scaled)[0][0] * 100), 2)
    prob_class_1 = round(float(model.predict_proba(features_scaled)[0][1] * 100), 2)
    
    print(f"\033[1m- Probability of No Click:\033[0m {prob_class_0}%")
    print(f"\033[1m- Probability of Click:\033[0m {prob_class_1}%")

if __name__ == "__main__":
    main()




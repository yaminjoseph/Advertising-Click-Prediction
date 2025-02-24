import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

# Import training pipeline functions
from src.config import TARGET_COLUMN
from src.data_handling import load_data
from src.model_handling import train_model
from src.preprocessing import feature_engineering, scale_features

# Define paths
models_directory = './models'
model_path = os.path.join(models_directory, 'lg_model.pkl')

# Ensure the models directory exists
os.makedirs(models_directory, exist_ok=True)

# Run training pipeline if model doesn't exist
if not os.path.exists(model_path):
    print("Model not found. Running training pipeline...")
    
    # Load Data
    df = load_data()
    
    # Feature Engineering
    df = feature_engineering(df)

    # Split Features & Target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].values

    # Model Training
    model, scaler = train_model(X, y)

    # Save Model
    joblib.dump({'model': model, 'scaler': scaler}, model_path)
    print("Model Training Complete! Model Saved.")

# Load Flask App
app = Flask(__name__, template_folder='html_templates', static_folder='html_static')

# Load Model & Scaler
model, scaler = None, None
if os.path.exists(model_path):
    try:
        loaded = joblib.load(model_path)
        if isinstance(loaded, dict) and 'model' in loaded and 'scaler' in loaded:
            model, scaler = loaded['model'], loaded['scaler']
    except Exception as e:
        print(f"Error loading model or scaler: {e}")

# Frontend Route
@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            request_data = {k: float(v) for k, v in request.form.items()}
            data = pd.DataFrame([request_data])

            # Feature engineering
            data['Area Income Square'] = data['Area Income Square'] ** 2
            data = data[["Daily Internet Usage", "Daily Time Spent on Site", "Age", 
                         "Male", "Area Income Square", "North America", "Asia"]]

            # Scale & Predict
            if scaler and model:
                data_scaled = scaler.transform(data)
                prob_class_0 = round(float(model.predict_proba(data_scaled)[0][0] * 100), 2)
                prob_class_1 = round(float(model.predict_proba(data_scaled)[0][1] * 100), 2)
                result = f"Probability of No Click: {prob_class_0}% | Probability of Click: {prob_class_1}%"
            else:
                result = "Model or scaler not loaded properly."

            return render_template('index.html', prediction=result)

        except Exception as e:
            return render_template('index.html', prediction=f"Error during prediction: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)

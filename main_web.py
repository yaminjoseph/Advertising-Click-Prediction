import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

# Load App
app = Flask(__name__, template_folder='html_templates')

# Load Model and Scaler
models_directory = './models'
model_path = os.path.join(models_directory, 'lg_model.pkl')

model, scaler = None, None

if os.path.exists(model_path):
    try:
        loaded = joblib.load(model_path)
        if isinstance(loaded, tuple) and len(loaded) == 2:
            model, scaler = loaded
        elif isinstance(loaded, dict) and 'model' in loaded and 'scaler' in loaded:
            model = loaded['model']
            scaler = loaded['scaler']
    except Exception as e:
        print(f"Error loading model or scaler: {e}")

# Front End
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

            # Scale and predict
            if scaler and model:
                data_scaled = scaler.transform(data)
                prediction = model.predict(data_scaled)
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







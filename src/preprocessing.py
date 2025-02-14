import os
import sys
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler


PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

# Define country categories
NORTH_AMERICA_COUNTRIES = [
        "Antigua and Barbuda", "Bahamas", "Barbados", "Belize", "Canada", 
        "Costa Rica", "Cuba", "Dominica", "Dominican Republic", "El Salvador", 
        "Grenada", "Guatemala", "Haiti", "Honduras", "Jamaica", "Mexico", 
        "Nicaragua", "Panama", "Saint Kitts and Nevis", "Saint Lucia", 
        "Saint Vincent and the Grenadines", "Trinidad and Tobago", "United States of America",
        "United States Minor Outlying Islands", "Puerto Rico", "Guam", "Northern Mariana Islands",
        "American Samoa", "Cayman Islands", "Bermuda", "Isle of Man", "Saint Pierre and Miquelon"
    ]

ASIAN_COUNTRIES = [
        "Afghanistan", "Armenia", "Azerbaijan", "Bahrain", "Bangladesh", "Bhutan", 
        "Brunei Darussalam", "Cambodia", "China", "Georgia", "India", "Indonesia", "Iran", 
        "Iraq", "Israel", "Japan", "Jordan", "Kazakhstan", "Kuwait", "Kyrgyz Republic", 
        "Laos", "Lebanon", "Malaysia", "Maldives", "Mongolia", "Myanmar", 
        "Nepal", "North Korea", "Oman", "Palestinian Territory", "Qatar", "Saudi Arabia", 
        "Singapore", "South Korea", "Sri Lanka", "Syria", "Taiwan", "Tajikistan", 
        "Thailand", "Timor-Leste", "Turkey", "Turkmenistan", 
        "United Arab Emirates", "Uzbekistan", "Vietnam", "Yemen", "Hong Kong", "Macao","Korea","Lao People's Democratic Republic",
        'Philippines','Pakistan'
    ]

def feature_engineering(df):
    """Apply feature transformations."""
    df["Area Income Square"] = np.power(df['Area Income'], 2)
    df['North America'] = df["Country"].isin(NORTH_AMERICA_COUNTRIES).astype(int)
    df['Asia'] = df["Country"].isin(ASIAN_COUNTRIES).astype(int)
    
    # Drop unnecessary columns
    df.drop(columns=["Country", "Area Income"], inplace=True)

    return df

def scale_features(X, scaler=None):
    """Scale features using the saved StandardScaler."""
    if scaler is None:
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
            scaler = model_data['scaler']
    
    return scaler.transform(X)
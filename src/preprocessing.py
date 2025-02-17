# IMPORT LIBRARIES
import os
import sys
import pickle
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


# DEFINE PACKAGE ROOT AND SYSTEM PATH
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

# DEFINE COUNTRY CATEGORIES
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
    "United Arab Emirates", "Uzbekistan", "Vietnam", "Yemen", "Hong Kong", "Macao", "Korea", "Lao People's Democratic Republic",
    "Philippines", "Pakistan"
]

# FEATURE ENGINEERING FUNCTION
def feature_engineering(df):
    """Apply feature transformations to the dataframe.

    - Create a new feature 'Area Income Square' as the square of 'Area Income'.
    - Create binary indicators for whether a country is in North America or Asia.
    - Drop unnecessary columns 'Country' and 'Area Income'.
    
    Args:
        df (pd.DataFrame): Input dataframe containing 'Area Income' and 'Country'.
    
    Returns:
        pd.DataFrame: Transformed dataframe with new features.
    """
    df["Area Income Square"] = np.power(df['Area Income'], 2)
    df['North America'] = df["Country"].isin(NORTH_AMERICA_COUNTRIES).astype(int)
    df['Asia'] = df["Country"].isin(ASIAN_COUNTRIES).astype(int)

    # Drop unnecessary columns
    df.drop(columns=["Country", "Area Income"], inplace=True)

    return df

# FEATURE SCALING FUNCTION
def scale_features(X, scaler=None):
    """Scale features using a pre-trained StandardScaler.

    - If no scaler is provided, load it from the saved model file.
    - Apply the scaler to transform the feature set.
    
    Args:
        X (pd.DataFrame or np.ndarray): Feature set to be scaled.
        scaler (StandardScaler, optional): Preloaded scaler object. Defaults to None.

    Returns:
        np.ndarray: Scaled feature set.
    """
    if scaler is None:
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
            scaler = model_data['scaler']

    return scaler.transform(X)
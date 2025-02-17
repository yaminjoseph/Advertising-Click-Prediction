# IMPORT LIBRARIES
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from src.config import MODEL_PATH, MODEL_PARAMS

# TRAINING THE MODEL AND SAVING THE SCALER
def train_model(X, y):
    """
    Train a Logistic Regression model using GridSearchCV and save the scaler.
    
    Parameters:
    - X: Feature matrix (numpy array or pandas DataFrame).
    - y: Target vector (numpy array or pandas Series).
    
    Returns:
    - best_model: Trained Logistic Regression model with the best hyperparameters.
    """
    
    # Initialize and fit the scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize Logistic Regression model
    base_model = LogisticRegression()
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=base_model, 
        param_grid=MODEL_PARAMS, 
        cv=5, 
        scoring='accuracy', 
        n_jobs=-1, 
        verbose=2
    )
    grid_search.fit(X_scaled, y)

    # Extract the best model
    best_model = grid_search.best_estimator_

    # Save the best model and scaler to disk
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': best_model, 'scaler': scaler}, f)
    
    return best_model

# LOADING THE TRAINED MODEL AND SCALER
def load_model():
    """
    Load the trained Logistic Regression model and scaler from disk.
    
    Returns:
    - model: Trained Logistic Regression model.
    - scaler: StandardScaler used during training.
    """
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['scaler']

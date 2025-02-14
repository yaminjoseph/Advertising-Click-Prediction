import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from src.config import MODEL_PATH, MODEL_PARAMS

def train_model(X, y):
    """Train logistic regression model with GridSearchCV and save the scaler."""
    
    # Initialize and fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    base_model = LogisticRegression()
    grid_search = GridSearchCV(estimator=base_model, param_grid=MODEL_PARAMS, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_scaled, y)

    best_model = grid_search.best_estimator_

    # Save model and scaler together
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': best_model, 'scaler': scaler}, f)
    
    return best_model

def load_model():
    """Load trained model and scaler from disk."""
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['scaler']




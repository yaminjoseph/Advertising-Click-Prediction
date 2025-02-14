from src.data_handling import load_data
from src.preprocessing import feature_engineering, scale_features
from src.model_handling import train_model
from src.config import TARGET_COLUMN

def run_pipeline():
    """Run the full machine learning pipeline."""
    df = load_data()
    df = feature_engineering(df)

    # Split features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].values

    # No scaling here, let the model handle it and save the scaler
    model = train_model(X, y)

    print("Model training complete!")

if __name__ == "__main__":
    run_pipeline()

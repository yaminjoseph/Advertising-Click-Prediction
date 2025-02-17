# IMPORT LIBRARIES
from src.config import TARGET_COLUMN
from src.data_handling import load_data
from src.model_handling import train_model
from src.preprocessing import feature_engineering, scale_features

# PIPELINE 
def run_pipeline():
    """
    RUN THE FULL MACHINE LEARNING PIPELINE

    This function orchestrates the end-to-end machine learning pipeline:
    1. LOAD DATA: Load the dataset using the load_data function.
    2. FEATURE ENGINEERING: Apply feature transformations to prepare the data.
    3. SPLIT FEATURES AND TARGET: Separate the independent variables (X) and the target variable (y).
    4. MODEL TRAINING: Train the model on the prepared data.
    5. OUTPUT: Indicate the completion of model training.
    """
    # Output
    print("Running Training Pipeline...")

    # Load Data
    df = load_data()
    
    # Feature Eng.
    df = feature_engineering(df)

    # Split Features & Target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].values

    # Model Trainig 
    model = train_model(X, y)

    # Output
    print("Model Training Complete! Model Saved.")

if __name__ == "__main__":
    run_pipeline()


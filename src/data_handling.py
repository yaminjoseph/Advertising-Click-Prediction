import pandas as pd
from src.config import DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN

def load_data():
    """Load dataset from the specified path."""
    df = pd.read_csv(DATA_PATH)

    # Select relevant columns
    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

    return df
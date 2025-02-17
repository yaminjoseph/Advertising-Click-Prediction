# IMPORT LIBRARIES
import pandas as pd
from src.config import DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN

def load_data():
    """
    Load Dataset from the Specified Path.

    - Reads the dataset from the path defined in DATA_PATH.
    - Selects relevant columns specified in FEATURE_COLUMNS and TARGET_COLUMN.
    - Returns a DataFrame containing the selected columns.
    """
    # LOAD DATA FROM CSV FILE
    df = pd.read_csv(DATA_PATH)

    # SELECT RELEVANT COLUMNS
    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

    return df
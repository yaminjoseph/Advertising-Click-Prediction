# IMPORT LIBRARIES
import os
import sys
from pathlib import Path
from src.predict_cli import make_prediction
from src.training_cli import run_pipeline

# SETTING UP PROJECT ROOT AND IMPORTS
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

# MAIN FUNCTION TO RUN THE PIPELINE AND MAKE PREDICTIONS
def main():
    # LOAD PIPELINE
    run_pipeline()
    
    # MAKE PREDICTIONS
    make_prediction()

# ENTRY POINT FOR SCRIPT EXECUTION
if __name__ == "__main__":
    main()





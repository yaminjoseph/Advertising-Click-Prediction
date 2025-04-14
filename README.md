# Ad Click Prediction - Setup and Usage Guide

## Environment Setup
### 1. Create and Activate Virtual Environment
```bash
conda create -n pred-env python=3.10.16  # Create a new environment
conda activate pred-env  # Activate the environment
```

### 2. Navigate to Project Directory
```bash
cd ../Advertising-Click-Prediction  # Change to source directory
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt  # Install required libraries
```

---

## Running the Model
### Option 1: Running via Terminal
#### Train and Predict
```bash
python -m src.training_cli  # Run Model Training
python -m src.predict_cli  # Run Prediction
```
#### Run Training & Prediction Together
```bash
python main_cli.py  # Execute training and prediction in one step
```

### Option 2: Running via Flask Web Application
```bash
python main_web.py  # Start Flask-based Prediction API (No Training)
```

### Option 3: Running via Docker
#### Build and Run Docker Container
```bash
docker build -t ad-click-predictor .  # Build Docker image
docker images  # List available Docker images
docker run -d -p 5000:80 ad-click-predictor  # Run container in detached mode
http://localhost:5000 # Run in Web Browser
```

---

## Deactivating and Deleting the Environment
### Option 1 & 2: Conda Environment Cleanup
```bash
conda deactivate  # Deactivate environment
conda env remove --name pred-env  # Remove environment
```

### Option 3: Docker Container Cleanup
#### Stop and Remove Container
```bash
docker stop <container_id>  # Stop running container
docker ps -a  # List all containers
docker rm <container_id>  # Remove container
```
#### Remove Docker Images and Prune Unused Resources
```bash
docker container prune  # Remove all stopped containers
docker images  # List available images
docker rmi <image_id>  # Remove specific image
docker image prune  # Remove unused images
```

## Project Structure
```bash
ad_click_prediction/
│── data/               # Data directory
│   ├── exploration/    # Data exploration dataset storage
│   ├── raw/            # Raw dataset storage
│── html_static/        # Static assets (CSS, JS, etc.)
│   ├── style.css       # Styling for web interface
│── html_templates/     # HTML templates for Flask app
│   ├── index.html      # Main web interface template
│── models/             # Trained models storage
│   ├── lg_model.pkl    # model file
│── notebooks/          # Jupyter notebooks
│   ├── eda/            # Exploratory Data Analysis notebooks
│   ├── eda_src/        # EDA scripts and utilities
│   ├── experiments/    # Model Experiments
│── src/                # Source code directory
│   ├── config.py       # Configuration settings
│   ├── data_handling.py # Data processing functions
│   ├── model_handling.py # Model training and saving
│   ├── predict_cli.py  # CLI for making predictions
│   ├── preprocessing.py # Preprocessing functions
│   ├── training_cli.py # CLI for model training
│── .gitignore          # Git ignore file
│── Dockerfile          # Docker configuration
│── main_cli.py         # Script for CLI-based training & prediction
│── main_docker.py      # Docker entry point script
│── main_web.py         # Flask web app script
│── README.md           # Documentation
│── requirements.txt    # Environments 
```

---

## Additional Notes
- Ensure you have **Conda, Python (>=3.10.16), and Docker** installed before setup.
- The Flask API runs on port **5000** by default. Adjust the port if needed.
- Use `docker ps` to check active containers and `docker logs <container_id>` for debugging.

For any issues, refer to the documentation or raise an issue in the repository.











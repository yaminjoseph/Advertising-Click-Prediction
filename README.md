conda create -n pred-env python=3.10.16 # install env
conda activate pred-env # activate env
cd ../ad_click_prediction # cd source
pip install -r requirements.txt # Install Libraries
python -m src.training_cli # Run Model Training
python -m src.predict_cli # Run Prediction
python main_cli.py # Run on Terminal Model Training & Prediction
conda deactivate # deactivate env
conda env remove --name pred-env # remove env

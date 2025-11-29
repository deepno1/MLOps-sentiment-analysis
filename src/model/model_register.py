import numpy as np
import pandas as pd
import pickle
import json
import logging
import mlflow
import dagshub
import os
from src.logger import logging
from src.exception import CustomException
import sys
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

mlflow.set_tracking_uri('https://dagshub.com/deepno1/MLOps-sentiment-analysis.mlflow')
dagshub.init(repo_owner='deepno1', repo_name='MLOps-sentiment-analysis', mlflow=True)

def load_model_info(file_path):
    try:
        with open(file_path,'r') as file:
            model_info = json.load(file)
            logging.debug('Model info loaded from %s', file_path)
            return model_info
    except Exception as e:
        raise CustomException(e,sys)
    
def register_model(model_info):
    try:
        model_run_url = 'runs:/{}/{}'.format(model_info['run_id'],model_info['model_path'])
        model_version = mlflow.register_model(model_run_url,'My_model')
        
        mlflow_client = mlflow.tracking.MlflowClient()
        mlflow_client.transition_model_version_stage(
            name = model_version.name,
            version = model_version.version,
            stage = 'Staging'
        )
        
        logging.debug('Model {} version {} registered and transition to Staging.'.format(model_version.name,model_version.version))
    except Exception as e:
        raise CustomException(e,sys)

def main():
    try:
        model_info = load_model_info('reports\experiment_info.json')
        register_model(model_info)
    except Exception as e:
        raise CustomException(e,sys)
    
if __name__ == '__main__':
    main()
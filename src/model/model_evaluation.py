import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import mlflow.sklearn
import dagshub
import os
from src.logger import logging
from src.exception import CustomException
import sys

mlflow.set_tracking_uri('https://dagshub.com/deepno1/MLOps-sentiment-analysis.mlflow')
dagshub.init(repo_owner='deepno1', repo_name='MLOps-sentiment-analysis', mlflow=True)

def load_model(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info('Model loaded from %s', file_path)
        return model
    except Exception as e:
        raise CustomException(e,sys)

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(clf, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred = clf.predict(x_test)
        y_pred_proba = clf.predict_proba(x_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logging.info('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        raise CustomException(e,sys)

def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        raise CustomException(e,sys)

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file)
        logging.info('Model info saved to %s', file_path)
    except Exception as e:
        raise CustomException(e,sys)

def main():
    mlflow.set_experiment("my-dvc-pipeline")
    with mlflow.start_run() as run:
        try:
            clf = load_model('./models/model.pkl')
            test_data = load_data('./data/processed/test_bow.csv')
            
            x_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            metrics = evaluate_model(clf, x_test, y_test)
            save_metrics(metrics, 'reports/metrics.json')
            mlflow.log_metrics(metrics)
            mlflow.log_artifact('reports/metrics.json')
            
            params = clf.get_params()
            mlflow.log_params(params)
            
            input_example = x_test[:10]
            mlflow.sklearn.log_model(clf, "model", input_example = input_example)
            
            save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')
            
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == '__main__':
    main()
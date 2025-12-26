import numpy as np
import pandas as pd
import pickle
from sklearn.naive_bayes import MultinomialNB
import sys
from src.exception import CustomException
from src.logger import logging
import yaml

def load_params(params_path : str) -> dict:
    try:
        with open(params_path,'r') as file:
            params = yaml.safe_load(file)
            logging.debug('Parameters retrieved from %s', params_path)
            return params
    except Exception as e:
        raise CustomException(e,sys)

def load_data(path : str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logging.info('Data loaded from %s', path)

        return df
    except Exception as e:
        raise CustomException(e,sys)
    
def train_model(x_train, y_train,**model_params):
    try :
        model = MultinomialNB(**model_params)
        model.fit(x_train,y_train)
        logging.info('Model training completed with params {}'.format(model.get_params()))

        return model
    except Exception as e:
        raise CustomException(e,sys)
    
def save_model(obj,file_path : str):
    try:
        with open(file_path,'wb') as file:
            pickle.dump(obj,file)
        logging.info('Model saved to %s', file_path)
    except Exception as e:
        raise CustomException(e,sys)

def main():
    try:
        params = load_params('params.yaml')

        model_params = {
                           'alpha' : params['model_building']['alpha']
                       }

        train_data = load_data('./data/processed/train_bow.csv')
        x_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(x_train, y_train,**model_params)
        
        save_model(clf, 'models/model.pkl')
    except Exception as e:
        raise CustomException(e,sys)

if __name__ == '__main__':
    main()
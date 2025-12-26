import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from src.logger import logging
import pickle
from src.exception import CustomException
import sys
import yaml

def load_params(params_path : str) -> dict:
    try:
        with open(params_path,'r') as file:
            params = yaml.safe_load(file)
            logging.debug('Parameters retrieved from %s', params_path)
            return params
    except Exception as e:
        raise CustomException(e,sys)

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logging.info('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        raise CustomException(e,sys)

def apply_bow(train_data: pd.DataFrame, test_data: pd.DataFrame, **vec_params : dict):
    try:
        logging.info("Applying BOW...")
        vectorizer = CountVectorizer(**vec_params)

        X_train = train_data['review'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['review'].values
        y_test = test_data['sentiment'].values

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        with open('models/vectorizer.pkl', 'wb') as file:
            pickle.dump(vectorizer, file)
        logging.info('Bag of Words applied and data transformed')

        return train_df, test_df
    except Exception as e:
        raise CustomException(e,sys)

def save_data(df: pd.DataFrame, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logging.info('Data saved to %s', file_path)
    except Exception as e:
        raise CustomException(e,sys)

def main():
    try:
        params = load_params('params.yaml')

        vec_params = {  'max_features' : params['feature_engineering']['max_features'],
                        'max_df' : params['feature_engineering']['max_df'],
                        'min_df' : params['feature_engineering']['min_df'],
                        'ngram_range' : tuple(params['feature_engineering']['ngram_range'])
                     }

        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        train_df, test_df = apply_bow(train_data, test_data, **vec_params)

        save_data(train_df, os.path.join("./data", "processed", "train_bow.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_bow.csv"))
    except Exception as e:
        raise CustomException(e,sys)

if __name__ == '__main__':
    main()
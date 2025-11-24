import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.connections.s3_connection import s3_operation
from dotenv import load_dotenv

def load_params(params_path: str) -> dict:

    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise

def load_data(data_url: str) -> pd.DataFrame:
    
    try:
        df = pd.read_csv(data_url)
        logging.info('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    
    try:
        logging.info("pre-processing...")
        final_df = df[df['sentiment'].isin(['positive', 'negative'])]
        final_df['sentiment'] = final_df['sentiment'].map({'positive' : 1,'negative' : 0})
        logging.info('Data preprocessing completed')
        return final_df
    except KeyError as e:
        logging.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logging.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        TEST_SIZE = 0.25
        FILE_NAME = 'IMDB.csv'
        BUCKET_NAME = ''

        load_dotenv()
        AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
        AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
        
        #df = load_data(data_url='https://raw.githubusercontent.com/deepno1/MLOps-sentiment-analysis/refs/heads/main/notebooks/IMDB.csv')

        s3_connection = s3_operation(bucket_name = BUCKET_NAME, aws_access_key = AWS_ACCESS_KEY, aws_secret_key = AWS_SECRET_KEY)
        df = s3_connection.fetch_data_from_bucket(file_name = FILE_NAME) 
        final_df = preprocess_data(df)

        train_data, test_data = train_test_split(final_df, test_size = TEST_SIZE, random_state = 43)
        save_data(train_data, test_data, data_path='./data')
        
    except Exception as e:
        logging.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
import boto3
import pandas as pd
from src.logger import logging
from io import StringIO
import sys
from src.exception import CustomException

class s3_operation:

    def __init__(self,bucket_name : str, aws_access_key : str, aws_secret_key : str, region_name = 'us-east-1') -> None :
        try:
            self.bucket_name = bucket_name
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id = aws_access_key,
                aws_secret_access_key = aws_secret_key,
                region_name = region_name
            )
            logging.info("Data Ingestion from S3 bucket initialized")

        except Exception as e:
            raise CustomException(e,sys)
        
    def fetch_data_from_bucket(self, file_name : str) -> pd.DataFrame:
        try:
            logging.info(f"Fetching file '{file_name}' from S3 bucket '{self.bucket_name}'...")
            file_obj = self.s3_client.get_object(Bucket = self.bucket_name, Key = file_name)
            df = pd.read_csv(StringIO(file_obj['Body'].read().decode('utf-8')))
            logging.info(f"Successfully fetched and loaded '{file_name}' from S3 that has {len(df)} records.")
            return df
        except Exception as e:
            raise CustomException(e,sys)


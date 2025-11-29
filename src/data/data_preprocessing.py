import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.logger import logging
from src.exception import CustomException
import sys
from bs4 import BeautifulSoup
import html

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

def preprocess_dataframe(df : pd.DataFrame, col : str) -> pd.DataFrame:
    try:
        def clean_text(text : str) -> str:
            text = str(text)
            text = BeautifulSoup(text,'html.parser').get_text()
            text = re.sub(r'https?://\S+|www\.\S+','',text)
            text = text.lower()
            text = re.sub(r'\d+','',text)
            text = re.sub(r'[^\w\s]',' ',text)
            text = re.sub(r'\s+',' ',text)
            text = str(text).strip()

            return text
        
        def remove_stopword(text : str) -> str:
            stopword = set(stopwords.words('english'))
            text = [word for word in str(text).split() if word not in stopword]
            text = ' '.join(text)

            return text
        
        def text_lemmatize(text):
            pos_text = nltk.pos_tag(str(text).split())

            lemma = WordNetLemmatizer()
            text = [lemma.lemmatize(word[0]) for word in pos_text]
            text = " ".join(text)

            return text
        
        df[col] = df[col].apply(clean_text)
        df[col] = df[col].apply(remove_stopword)
        df[col] = df[col].apply(text_lemmatize)

        df = df.dropna(subset=[col])
        logging.info("Data pre-processing completed")
        return df

    except Exception as e:
        raise CustomException(e,sys)
    
def main():
    try:
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logging.info('data loaded properly')

        train_processed_data = preprocess_dataframe(train_data, 'review')
        test_processed_data = preprocess_dataframe(test_data, 'review')

        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        logging.info('Processed data saved to %s', data_path)

    except Exception as e:
        logging.error('Failed to complete the data transformation process: %s', e)
        raise CustomException(e,sys)

if __name__ == '__main__':
    main()
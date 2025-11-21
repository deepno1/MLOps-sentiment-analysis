import mlflow
import pandas as pd
import mlflow.sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from bs4 import BeautifulSoup
import html
import time
import nltk
import dagshub
import scipy.sparse

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

CONFIG = {
    "data_path": "notebooks\data.csv",
    "test_size": 0.25,
    "mlflow_tracking_uri": "https://dagshub.com/deepno1/MLOps-sentiment-analysis.mlflow",
    "experiment_name": "Bow vs TfIdf"
}

def clean_text(text):

    text = BeautifulSoup(text,'html').get_text()
    text = re.sub(r'https?://\S+|www\.\S+','',text)
    text = text.lower()
    text = re.sub(r'\d+','',text)
    text = re.sub(r'[^\w\s]',' ',text)
    test = re.sub(r'\s+',' ',text)
    text = str(text).strip()

    return text

nltk.download('stopwords')
def remove_stopword(text):

    stopword = set(stopwords.words('english'))
    text = [word for word in str(text).split() if word not in stopword]
    text = " ".join(text)

    return text

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
def text_lemmatize(text):

    pos_text = nltk.pos_tag(str(text).split())

    lemma = WordNetLemmatizer()
    text = [lemma.lemmatize(word[0]) for word in pos_text]
    text = " ".join(text)

    return text

def normalize_text(df):
    
    df['review'] = df['review'].apply(clean_text)
    df['review'] = df['review'].apply(remove_stopword)
    df['review'] = df['review'].apply(text_lemmatize)

    return df

def load_data(path):

    df = pd.read_csv(path)
    df = normalize_text(df)
    df = df[df['sentiment'].isin(['positive','negative'])] 
    df['sentiment'] = df['sentiment'].map({'positive' : 1,'negative' : 0})

    return df

VECTORIZERS = {
    'BoW': CountVectorizer(),
    'TF-IDF': TfidfVectorizer()
}

ALGORITHMS = {
    'LogisticRegression': LogisticRegression(),
    'MultinomialNB': MultinomialNB(),
    'XGBoost': XGBClassifier(),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier()
}

mlflow.set_tracking_uri(CONFIG['mlflow_tracking_uri'])
dagshub.init(repo_owner='deepno1', repo_name='MLOps-sentiment-analysis', mlflow=True)
mlflow.set_experiment(CONFIG['experiment_name'])

def train_eval_track(df):

    with mlflow.start_run() as parent:
        for algo_name,algo in ALGORITHMS.items():
            for vec_name, vectorizer in VECTORIZERS.items():
                with mlflow.start_run(nested = True) as child:
                    
                    x = vectorizer.fit_transform(df['review'])
                    y = df['sentiment']

                    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = CONFIG['test_size'], random_state = 42)

                    mlflow.log_param('model',algo_name)
                    mlflow.log_param('vectorizer',vec_name)
                    mlflow.log_param('test_size',CONFIG['test_size'])

                    model = algo
                    model.fit(x_train,y_train)

                    train_y_pred = model.predict(x_train)
                    test_y_pred = model.predict(x_test)

                    train_acc = accuracy_score(y_train,train_y_pred)
                    test_acc = accuracy_score(y_test,test_y_pred)

                    precision = precision_score(y_test,test_y_pred)
                    recall = recall_score(y_test,test_y_pred)
                    f_one = f1_score(y_test,test_y_pred)

                    metrics = {
                               'train_accuracy' : train_acc,
                               'test_accuracy' : test_acc,
                               'precision' : precision,
                               'recall' : recall,
                               'f1' : f_one     
                                }
                    mlflow.log_metrics(metrics)

                    input_example = x_test[:5] if not scipy.sparse.issparse(x_test) else x_test[:5].toarray()
                    mlflow.sklearn.log_model(model,'model',input_example=input_example)

                    print(f"\nAlgorithm: {algo_name}, Vectorizer: {vec_name}")
                    print(f"Metrics: {metrics}")

if __name__ == '__main__':

    df = load_data(CONFIG['data_path'])
    train_eval_track(df)

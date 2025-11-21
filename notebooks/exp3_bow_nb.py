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
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

CONFIG = {
    "data_path": "notebooks\IMDB.csv",
    "test_size": 0.25,
    "mlflow_tracking_uri": "https://dagshub.com/deepno1/MLOps-sentiment-analysis.mlflow",
    "experiment_name": "Bow & MultinomialNB"
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
    x = df['review']
    y = df['sentiment']

    return train_test_split(x, y, test_size = CONFIG['test_size'], random_state = 43)

pipeline = Pipeline([
                        ('vec',CountVectorizer()),
                        ('model',MultinomialNB())   
                    ])

# Exp on sample data (notebooks\data.csv)
#params = {
#            'vec__max_df' : [0.75, 0.85],
#            'vec__min_df' : [2, 5, 10],
#            'vec__max_features' : [1000, 5000],
#            'vec__ngram_range' : [(1,1), (1,2)],
#            'model__alpha' : [0.5, 1.0]   
#         }

# Final exp with original data (notebooks\IMDB.csv)
params = {
            'vec__max_df' : [0.75, 0.85],
            'vec__min_df' : [5],
            'vec__max_features' : [5000],
            'vec__ngram_range' : [(1,1)],
            'model__alpha' : [0.5]   
         }


mlflow.set_tracking_uri(CONFIG['mlflow_tracking_uri'])
dagshub.init(repo_owner='deepno1', repo_name='MLOps-sentiment-analysis', mlflow=True)
mlflow.set_experiment(CONFIG['experiment_name'])

def train_eval_track(x_train,x_test,y_train,y_test):

    with mlflow.start_run() as parent:
        gs = GridSearchCV(pipeline, params, cv = 5, scoring = 'f1', n_jobs = -1)
        gs.fit(x_train,y_train)

        for param,mean_test_score in zip(gs.cv_results_['params'],gs.cv_results_['mean_test_score']):
            with mlflow.start_run(run_name = 'NB with params: {}'.format(param), nested = True) as child:
                vec_params = {}
                model_params = {}

                for name,value in param.items():
                    if name.startswith('vec__'):
                        x = name.split('__')[1]
                        vec_params[x] = value 

                    elif name.startswith('model__'):
                        x = name.split('__')[1]
                        model_params[x] = value

                vectorizer = CountVectorizer(**vec_params)
                x_train_transformed = vectorizer.fit_transform(x_train)
                x_test_transformed = vectorizer.transform(x_test)

                model = MultinomialNB(**model_params)
                model.fit(x_train_transformed,y_train)

                train_y_pred = model.predict(x_train_transformed) 
                test_y_pred = model.predict(x_test_transformed) 

                train_acc = accuracy_score(y_train,train_y_pred)
                test_acc = accuracy_score(y_test,test_y_pred)
                precision = precision_score(y_test,test_y_pred)
                recall = recall_score(y_test,test_y_pred)
                f_one = f1_score(y_test,test_y_pred)

                mertics = {
                            'train_accuracy' : train_acc,
                            'test_accuracy' : test_acc,
                            'precision' : precision,
                            'recall' : recall,
                            'f1' : f_one,
                            'mean_test_score' : mean_test_score    
                          }
                
                mlflow.log_metrics(mertics)
                mlflow.log_params(param)

        mlflow.log_params(gs.best_params_)
        mlflow.log_metric('best_f1_score',gs.best_score_)
        mlflow.log_param('vectorizer','Bow')
        mlflow.log_param('model','MultinomialNB')
        input_example = x_test[:10].values
        mlflow.sklearn.log_model(gs.best_estimator_,'best_model',input_example = input_example)
        
        print(f"\nBest Params: {gs.best_params_} | Best F1 Score: {gs.best_score_:.4f}")


if __name__ == '__main__':

    x_train,x_test,y_train,y_test = load_data(CONFIG['data_path'])
    train_eval_track(x_train,x_test,y_train,y_test)

    #alpha 0.5,max_df = 0.75,0.85,'vec__max_features': 5000,'vec__min_df': 5,'vec__ngram_range': (1, 1)



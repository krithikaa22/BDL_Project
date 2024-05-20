# %%
import os
import requests
import tarfile
from io import BytesIO
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml import Pipeline
from pyspark.sql.functions import rand

import numpy as np 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from tensorflow.keras.metrics import AUC

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import mlflow

import warnings
warnings.filterwarnings('ignore')

# %%
def download_and_extract_dataset(url, extract_path):
    response = requests.get(url)
    if response.status_code == 200:
        with tarfile.open(fileobj=BytesIO(response.content), mode="r:gz") as tar:
            tar.extractall(path=extract_path)
    else:
        print("Failed to download the dataset. Please check the URL or your internet connection.")

# %%
def load_reviews(spark, path, label):
    df = spark.read.text(os.path.join(path, "*.txt"))
    df = df.withColumn("label", lit(label))
    return df

# %%
def preprocess_data(train_df, max_words=10000, max_len=100):
    # Tokenization and Stopwords Removal using PySpark
    tokenizer = Tokenizer(inputCol="value", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    pipeline = Pipeline(stages=[tokenizer, remover])
    train_df = pipeline.fit(train_df).transform(train_df)
    
    # Convert filtered words to list
    filtered_words_list = train_df.select("filtered_words").rdd.flatMap(lambda x: x).collect()
    filtered_words_str = [" ".join(words) for words in filtered_words_list]
    
    # Tokenization using Keras Tokenizer
    keras_tokenizer = KerasTokenizer(num_words=max_words)
    keras_tokenizer.fit_on_texts(filtered_words_str)
    sequences = keras_tokenizer.texts_to_sequences(filtered_words_str)

    X = pad_sequences(sequences, maxlen=max_len)
    y = np.array(train_df.select("label").collect())

    return X, y

# %%
parameter_configurations = [
    {'max_words': 10000, 'max_len': 100, 'embedding_dim': 128, 'lstm_units': 128, 'dropout': 0.2},
    {'max_words': 10000, 'max_len': 100, 'embedding_dim': 128, 'lstm_units': 256, 'dropout': 0.2},
    {'max_words': 10000, 'max_len': 100, 'embedding_dim': 128, 'lstm_units': 128, 'dropout': 0.1},
    {'max_words': 10000, 'max_len': 100, 'embedding_dim': 256, 'lstm_units': 128, 'dropout': 0.2},
    {'max_words': 10000, 'max_len': 150, 'embedding_dim': 128, 'lstm_units': 128, 'dropout': 0.2},
    {'max_words': 5000, 'max_len': 100, 'embedding_dim': 128, 'lstm_units': 128, 'dropout': 0.2},
    {'max_words': 10000, 'max_len': 100, 'embedding_dim': 128, 'lstm_units': 128, 'dropout': 0.2, 'activation': 'relu'},
    {'max_words': 10000, 'max_len': 100, 'embedding_dim': 128, 'lstm_units': 128, 'dropout': 0.2, 'optimizer': 'rmsprop'}
]


# %%
def train_model(params,x_train,x_val,y_train,y_val):
    mlflow.autolog()

    max_words = params['max_words']
    max_len = params['max_len']
    embedding_dim = params['embedding_dim']
    lstm_units = params['lstm_units']
    dropout = params['dropout']

    model = Sequential()
    model.add(Embedding(max_words, embedding_dim))
    model.add(SpatialDropout1D(dropout))
    model.add(LSTM(lstm_units, dropout=dropout, recurrent_dropout=dropout))
    model.add(Dense(1, activation='sigmoid'))

    es = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=2)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',AUC()])

    model.fit(x_train,y_train, epochs=2, validation_data=(x_val, y_val), batch_size = 64, callbacks=[es])

    eval_result = model.evaluate(x_val,y_val,batch_size=64)

    mlflow.end_run()

    return eval_result

# %%

logistic_params = [
    {'penalty': 'l2', 'C': 0.01, 'solver': 'lbfgs', 'max_iter': 100},
    {'penalty': 'l2', 'C': 0.1, 'solver': 'lbfgs', 'max_iter': 200},
    {'penalty': 'l2', 'C': 1.0, 'solver': 'lbfgs', 'max_iter': 300},
    {'penalty': 'l1', 'C': 0.01, 'solver': 'saga', 'max_iter': 100},
    {'penalty': 'l1', 'C': 0.1, 'solver': 'saga', 'max_iter': 200},
    {'penalty': 'l1', 'C': 1.0, 'solver': 'saga', 'max_iter': 300},
    {'penalty': 'elasticnet', 'C': 0.01, 'solver': 'saga', 'max_iter': 100, 'l1_ratio': 0.5},
    {'penalty': 'elasticnet', 'C': 0.1, 'solver': 'saga', 'max_iter': 200, 'l1_ratio': 0.5},
    {'penalty': 'elasticnet', 'C': 1.0, 'solver': 'saga', 'max_iter': 300, 'l1_ratio': 0.5},
    {'penalty': 'none', 'C': 1.0, 'solver': 'newton-cg', 'max_iter': 100},
    {'penalty': 'none', 'C': 1.0, 'solver': 'newton-cg', 'max_iter': 200},
    {'penalty': 'none', 'C': 1.0, 'solver': 'newton-cg', 'max_iter': 300}
]


def train_and_log_models(params_list, X, y):
    # Setting the MLflow experiment
    mlflow.set_experiment("Logistic_Regression_Experiments")

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    for params in params_list:
        with mlflow.start_run():
            # Create a logistic regression model with the specified parameters
            model = LogisticRegression(**params)

            # Train the model
            model.fit(X_train, y_train)

            # Predict on the validation set
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]  # Get probabilities for the positive class

            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred)
            roc_auc = roc_auc_score(y_val, y_proba)
            f1 = f1_score(y_val, y_pred)

            # Log parameters and metrics to MLflow
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("f1_score", f1)

            # Log the model
            mlflow.sklearn.log_model(model, "model")

            print(f"Logged model with params: {params} | Accuracy: {accuracy}, ROC AUC: {roc_auc}, F1 Score: {f1}")



# %%

dataset_url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
dataset_path = 'aclImdb'

# Download and extract the dataset
if not os.path.exists(dataset_path):
    print("Downloading and extracting dataset...")
    download_and_extract_dataset(dataset_url, '.')

# Initialize Spark session
spark = SparkSession.builder \
    .appName("IMDb Sentiment Analysis") \
    .getOrCreate()

# Load positive and negative reviews into Spark DataFrames
pos_train_df = load_reviews(spark, os.path.join(dataset_path, "train/pos"), 1)
neg_train_df = load_reviews(spark, os.path.join(dataset_path, "train/neg"), 0)

train_df = pos_train_df.union(neg_train_df)

train_df = train_df.orderBy(rand())



# %%
mlflow.set_tracking_uri(uri="http://localhost:8080")

# %%
X,y = preprocess_data(train_df)



# %%
train_and_log_models(logistic_params,X,y)



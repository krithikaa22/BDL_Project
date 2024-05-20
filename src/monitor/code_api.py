import imdb

from fastapi import FastAPI, UploadFile, File
from prometheus_client import Counter, Gauge, CollectorRegistry, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import start_http_server
import io
import psutil
import time

from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml import Pipeline

import numpy as np
import pandas as pd

import argparse
from starlette.responses import Response

import pickle

import uvicorn
import os

import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title='Movie Review Sentiment Analysis')
Instrumentator().instrument(app).expose(app, include_in_schema=False, should_gzip=True)

# Prometheus metrics
custom_registry = CollectorRegistry()
REQUEST_COUNTER = Counter('api_requests_total', 'Total number of API requests', registry=custom_registry)

RUN_TIME_GAUGE = Gauge('api_run_time_seconds', 'Running time of the API', registry=custom_registry)

MEMORY_USAGE_GAUGE = Gauge('api_memory_usage', 'Memory usage of the API process', registry=custom_registry)
CPU_USAGE_GAUGE = Gauge('api_cpu_usage_percent', 'CPU usage of the API process', registry=custom_registry)

NETWORK_BYTES_SENT_GAUGE = Gauge('api_network_bytes_sent', 'Network bytes sent by the API process', registry=custom_registry)
NETWORK_BYTES_RECV_GAUGE = Gauge('api_network_bytes_received', 'Network bytes received by the API process', registry=custom_registry)

os.environ['PYSPARK_PYTHON'] = 'C:/Python310/python.exe'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'C:/Python310/python.exe'


# Defining a function to parse command line arguments

def parse_arguments():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument("model_path", type=str, help="Path to the model file")
    args = parser.parse_args()
    return args.model_path

# Defining a function to load a keras model stored on the local machine

def load_model_from_disk(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    
    return model


# Defining a function to preprocess the reviews

def preprocess_data(train_df, max_words=10000, max_len=100):
    # Tokenization and Stopwords Removal using PySpark
    tokenizer = Tokenizer(inputCol="Review", outputCol="words")
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

    return X

def get_reviews_preprocess(movie_title):
    ia = imdb.IMDb()

    # Search for the movie by title
    movie_list = ia.search_movie(movie_title)

    if not movie_list:
        print("Movie not found.")
        return

    # Get the first movie from the search results (assuming it's the correct one)
    movie = movie_list[0]

    # Fetch movie information including reviews
    ia.update(movie, info=['reviews'])

    # Get movie reviews
    reviews = movie.get('reviews', [])

    reviews_list = []
    for review in reviews:
        review_text = review.get('content', '')
        reviews_list.append({'Movie Title': movie_title, 'Review': review_text})
    
    reviews_df = pd.DataFrame(reviews_list)

    spark = SparkSession.builder \
        .appName("MovieReviewsPreprocessing") \
        .getOrCreate()
    
    # Convert movie_reviews_df to Spark DataFrame
    spark_reviews_df = spark.createDataFrame(reviews_df)

    X = preprocess_data(spark_reviews_df)

    return X

def predict_sentiment(model,X):
    predictions = model.predict(X)

    if np.mean(predictions) >= 0.5:
        return 'Positive'
    
    return 'Negative'

def process_memory():
    return psutil.virtual_memory().used/(1024)

@app.post('/predict')
async def predict(text:str):

    start_time = time.time()                    # Start time of the API call
    memory_usage_start = process_memory()       # Memory usage before the API call

    # Update network I/O gauges
    network_io_counters = psutil.net_io_counters()

    reviews = get_reviews_preprocess(text)

    # Reading the command line argument that stores the path of the model
    path = parse_arguments()

    # Loading the model from the path mentioned in the command line argument
    model = load_model_from_disk(path)
    
    sentiment = predict_sentiment(model, reviews)

    cpu_percent = psutil.cpu_percent(interval=1)    # Get the CPU usage percentage
    memory_usage_end = process_memory()             # Get the memory usage after the API call

    CPU_USAGE_GAUGE.set(cpu_percent)                                            # Set the CPU usage gauge
    MEMORY_USAGE_GAUGE.set((np.abs(memory_usage_end-memory_usage_start)))       # Set the memory usage gauge
    NETWORK_BYTES_SENT_GAUGE.set(network_io_counters.bytes_sent)                # Set the network bytes sent gauge
    NETWORK_BYTES_RECV_GAUGE.set(network_io_counters.bytes_recv)                # Set the network bytes received gauge
    
    # Calculate API running time
    end_time = time.time()
    run_time = end_time - start_time
    
    # Record API usage metrics
    REQUEST_COUNTER.inc()         # Increment the request counter             
    RUN_TIME_GAUGE.set(run_time)                    # Set the running time gauge

    return {"Sentiment": sentiment}

@app.get('/metrics')
async def predict(text:str):
    metrics= generate_latest(custom_registry)
    return Response(content=metrics)

if __name__ == '__main__':
    start_http_server(8000)

    # Running the web-application defined earlier
    uvicorn.run("code_api:app",host='127.0.0.1', port=8000, reload=True)
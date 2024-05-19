import imdb

from fastapi import FastAPI, UploadFile, File
import io

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

import pickle

import uvicorn
import os

import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title='Movie Review Sentiment Analysis')

os.environ['PYSPARK_PYTHON'] = '/opt/anaconda3/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/opt/anaconda3/bin/python'


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
        return None

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

@app.post('/predict')
async def predict(text:str):
    reviews = get_reviews_preprocess(text)

    if reviews is None:
        return {'Not a valid movie title'}

    # Reading the command line argument that stores the path of the model
    path = parse_arguments()

    # Loading the model from the path mentioned in the command line argument
    model = load_model_from_disk(path)
    
    sentiment = predict_sentiment(model, reviews)

    return {"Sentiment": sentiment}

if __name__ == '__main__':
    # Running the web-application defined earlier
    uvicorn.run("code_api:app", host="0.0.0.0", port=8000, reload=True)

    
    
    


 
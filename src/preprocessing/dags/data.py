from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand
import mlflow
import json

from src.preprocessing.spark_mlflow import download_and_extract_dataset, load_reviews, preprocess_data, train_and_log_models
from airflow.models import Variable
import apache_beam as beam

# Initiate a beam pipeline
beam_pipeline_py_file = Variable.get('beam')

with DAG(
    dag_id= "preprocessing",
    schedule_interval="@daily",
    default_args={
            "owner": "BDL",
            "retries": 1,
            "start_date": datetime(2023, 5, 20),
        },
    catchup=False
) as preprocessing:
        

        def get_data():
            dataset_url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
            dataset_path = 'aclImdb'

            # Download and extract the dataset
            if not os.path.exists(dataset_path):
                print("Downloading and extracting dataset...")
                download_and_extract_dataset(dataset_url, '.')
        
        def setup_spark():
            dataset_path = 'aclImdb'
            # Initialize Spark session
            spark = SparkSession.builder \
                .appName("IMDb Sentiment Analysis") \
                .getOrCreate()

            # Load positive and negative reviews into Spark DataFrames
            pos_train_df = load_reviews(spark, os.path.join(dataset_path, "train/pos"), 1)
            neg_train_df = load_reviews(spark, os.path.join(dataset_path, "train/neg"), 0)

            train_df = pos_train_df.union(neg_train_df)

            train_df = train_df.orderBy(rand())

            file_path = 'raw.json'
            with open(file_path, 'w') as file:
                json.dump({"data": train_df}, file)

        def preprocess():
            file_path = "raw.json"

            # Read the data from the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)

            train_df = data['data']

            mlflow.set_tracking_uri(uri="http://localhost:8080")

            X,y = preprocess_data(train_df)

            file_path = "data.json"

            with open(file_path, 'w') as file:
                json.dump({"x": X, "y": y}, file)

        # Operator to extract data from the public API
        extract = PythonOperator(
            task_id="get_data",
            python_callable=get_data,
            provide_context=True,
            op_kwargs={"name":"first_task"},
        )
        # Operator to initiate spark session
        spark = PythonOperator(
            task_id="spark",
            python_callable=setup_spark,
            provide_context=True,
            op_kwargs={"name":"first_task"},
        )
        # Operator to preprocess the extracted data
        process = PythonOperator(
            task_id="preprocess_data",
            python_callable=preprocess,
            provide_context=True,
            op_kwargs={"name":"first_task"},
        )

        
with DAG(
    dag_id= "train_model",
    schedule_interval="@daily",
    default_args={
            "owner": "BDL",
            "retries": 1,
            "start_date": datetime(2023, 5, 20),
        },
    catchup=False
) as model:
        
        def logistic_model():
            file_path = "data.json"

            # Read the data from the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)

            X = data['x']
            y = data['y']

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

            # Train a logistic regression model
            train_and_log_models(logistic_params,X,y)

        # Operator to train model
        train = PythonOperator(
            task_id="train_model",
            python_callable=logistic_model,
            provide_context=True,
            op_kwargs={"name":"first_task"},
        )

extract >> spark >> process >> train
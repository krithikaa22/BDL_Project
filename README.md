IMDb Sentiment Analysis
========

This repository contains the code and necessary resources for deploying a logistic regression model for sentiment analysis on IMDb movie reviews. The model predicts whether a given review is positive or negative.


Project Structure
================

The project is organized as follows:

- `src/`: Contains the source code for getting data, training the model and monitoring performance.
    - `preprocessing/`: Airflow dag to start a Spark pipeline to preprocess data
    - `model\`: MLFlow to test and train the optimal model
    - `monitor\`: Monitor performance of the optimal model

Requirements
===========================

- Python 3.x
- Docker
- scikit-learn
- Pandas
- NumPy
- Airflow
- Spark
- MLFlow
- FastAPI
- Prometheus
- Grafana

Usage
=================================

Clone the repository and modify configurations. Use docker compose to spin up the services:
```
    docker-compose up
```
Alternatively:
1. ./src/preprocessing/sparl_mlflow.ipynb can be run to start a spark session to extract and preprocess a public IMDb dataset
2. ./src/model/model_selection.ipynb can be run to start a mlflow session to pick the model with the optimal hyperparameters
3. ./src/model/fast_api.py can be run to set up a FastAPI endpoint which will predict using the optimal hyperparameters
    - The endpoint can be tested using Postman at the url: http://localhost:8000/predict
4. ./src/monitor/code_api.py can be run with the optimal model file as parameter to run the endpoint with the metrics that can be monitored
5. To start the prometheus server, ensure prometheus is installed and use the configuration in ./prometheus to start the server using the command 
```
    ./prometheus --config.file=./prometheus/prometheus.yml
```
The dashboard can be accessed at http://localhost:9090
6. To start the grafana server, ensure grafana is installed and use the configuration in ./grafana to start the server using the command in the grafana folder
```
    ./bin/grafana-server
``` 
The dashboard can be access at http://localhost:3000
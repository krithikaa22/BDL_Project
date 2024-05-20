
# Use a smaller base image
FROM python:3.9.1-slim

# Set work directory
WORKDIR /usr/src/app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libssl-dev libffi-dev libpq-dev && \
    rm -rf /var/lib/apt/lists/*


USER airflow

# Copy and install Python dependencies

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install apache-beam
RUN pip install apache-beam[gcp]
RUN pip install google-api-python-client
ADD . /home/beam 
    
RUN pip install apache-airflow[gcp_api]

COPY requirements.txt /usr/src/app/
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# Copy project files
COPY . /usr/src/app/

FROM prom/prometheus

COPY ./prometheus/prometheus.yml /etc/prometheus.yml

FROM grafana/grafana

COPY ./grafana/defaults.ini /etc/defaults.ini
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libblas-dev \
    liblapack-dev \
    libgomp1 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install AutoGluon in one layer
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir autogluon==1.2.0


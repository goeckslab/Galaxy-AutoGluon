FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libblas-dev \
    liblapack-dev \
    libgomp1 \
    libopenblas-dev \
    unzip \
  && rm -rf /var/lib/apt/lists/*

# Pin setuptools <81 to avoid pkg_resources warnings
RUN pip install --no-cache-dir 'setuptools<81.0.0'

RUN pip install --no-cache-dir --upgrade pip wheel

RUN pip install --no-cache-dir autogluon==1.2.0

RUN pip install --no-cache-dir pyarrow==20.0.0


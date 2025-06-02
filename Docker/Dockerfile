FROM python:3.10-slim

# Install system dependencies including unzip
RUN apt-get update && apt-get install -y \
    build-essential \
    libblas-dev \
    liblapack-dev \
    libgomp1 \
    libopenblas-dev \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Pin setuptools <81 to avoid pkg_resources deprecation warnings
RUN pip install --no-cache-dir 'setuptools<81.0.0'

# Upgrade pip, setuptools, and wheel (setuptools remains pinned to <81)
RUN pip install --no-cache-dir --upgrade pip wheel

# Install AutoGluon (will use pinned setuptools)
RUN pip install --no-cache-dir autogluon==1.2.0

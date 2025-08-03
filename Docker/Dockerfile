FROM python:3.10-slim

# Install system dependencies, including those required for building matplotlib
RUN apt-get update && apt-get install -y \
    build-essential \
    libblas-dev \
    liblapack-dev \
    libgomp1 \
    libopenblas-dev \
    unzip \
    pkg-config \
    libfreetype6-dev \
    libpng-dev \
    libqhull-dev \
  && rm -rf /var/lib/apt/lists/*

# Pin setuptools <81 to avoid pkg_resources warnings
RUN pip install --no-cache-dir 'setuptools<81.0.0'

# Upgrade pip and wheel
RUN pip install --no-cache-dir --upgrade pip wheel

# Install AutoGluon
RUN pip install --no-cache-dir autogluon==1.2.0

# Install pyarrow
RUN pip install --no-cache-dir pyarrow==20.0.0

# Install matplotlib and seaborn for plotting in the script
RUN pip install --no-cache-dir matplotlib seaborn

FROM python:3.9-slim

# Install system dependencies for building Python packages and AutoGluon requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libjpeg62-turbo \
    libpng16-16 \
    zlib1g \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --no-cache-dir -U pip

# Install numpy first to avoid version conflicts
RUN pip install --no-cache-dir numpy==1.23.5

# Install remaining Python dependencies
RUN pip install --no-cache-dir \
    pandas \
    autogluon.multimodal \
    scikit-learn

# Create a non-root user and set up a home directory
RUN useradd -m -s /bin/bash nonrootuser

# Switch to the non-root user
USER nonrootuser

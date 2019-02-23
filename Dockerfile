FROM tensorflow/tensorflow:latest-gpu-py3
RUN apt-get update

RUN apt-get install -y \
    git \
    curl \
    libsm6 \
    libxext6 \
    libxrender-dev \
    netbase

WORKDIR /opt
COPY requirements.txt .
RUN pip install -r requirements.txt
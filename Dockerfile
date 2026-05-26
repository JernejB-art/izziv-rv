FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1

RUN pip install numpy
RUN pip install opencv-python-headless
RUN pip install matplotlib
RUN pip install mediapipe==0.10.9
RUN pip install scipy
RUN pip install scikit-learn

WORKDIR /workspace
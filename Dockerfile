FROM python:trixie

RUN pip install numpy
RUN pip install opencv-python-headless
RUN pip install matplotlib
RUN pip install mediapipe
RUN pip install scipy

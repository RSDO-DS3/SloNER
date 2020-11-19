# contains python 3.8, conda 4.9.1
FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

RUN mkdir /app

WORKDIR /app

COPY ./src /app/src
COPY ./requirements.txt /app/

RUN pip install -r requirements.txt


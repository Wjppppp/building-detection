FROM ubuntu:16.04
# FROM python:3.6.15

# install system libraries
RUN apt-get update &&\
        apt-get -y install \
        build-essential \
        libsqlite3-dev \
        zlib1g-dev \
        libgeos-dev \
        libgdal-dev \
        libgnutls28-dev \
        libcurl4-openssl-dev \
        protobuf-compiler \
        python-pil \
        python-lxml \
        python-tk

# install python 3.6 with ubuntu 16.04
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:jblgf0/python && \
    apt-get update -y  && \
    apt-get install -y python3.6 python3.6-dev python3-pip && \
    apt-get install -y git &&\
    # update pip
    apt -y remove --purge python3-pip && \
    apt-get -y install curl && \
    curl -O https://bootstrap.pypa.io/pip/3.6/get-pip.py && \
    python3.6 get-pip.py && \
    python3.6 -m pip --version

WORKDIR /app
COPY . .

RUN export PYCURL_SSL_LIBRARY=gnutls 
RUN pip install -r requirements.txt
RUN cd ohsome2label && \
    pip install --editable .

RUN export LC_ALL=C.UTF-8 && \
    export LANG=C.UTF-8

EXPOSE 7000:7000
# ENTRYPOINT ["python", "./test.py"]
ENTRYPOINT ["tail", "-f", "/dev/null"]

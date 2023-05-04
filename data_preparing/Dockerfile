FROM ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Munich

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
        python-tk \
        tzdata

# install python 3.6
RUN apt-get update && \
    apt-get install -y python3 && \
    apt-get install -y python3-pip

RUN export PYCURL_SSL_LIBRARY=gnutls 
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN pip install pycurl
RUN pip install pandas
RUN pip install tensorflow==1.14.0
RUN pip install Cython
RUN pip install matplotlib
RUN pip install jupyter

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /app
COPY . .

RUN cd label && pip install --editable .

RUN echo 'export LC_ALL=C.UTF-8' >> ~/.bashrc 

RUN cd label/TF_od/cocoapi/PythonAPI &&\
    make &&\
    cp -r pycocotools ../../models/research &&\
    cd ../../models/research &&\
    protoc object_detection/protos/*.proto --python_out=. &&\    
    echo 'export PYTHONPATH=$PYTHONPATH:/app/label/TF_od/models/research/slim' >> ~/.bashrc 

# EXPOSE 7000:7000

# CMD [ "bash" ]

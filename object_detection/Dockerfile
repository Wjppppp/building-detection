FROM tensorflow/tensorflow:2.2.0-gpu

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Munich

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
# Install apt dependencies
RUN apt-get update &&\
    apt-get install -y \
    tzdata \
    git \
    libhdf5-dev \
    python3-h5py \
    python3-opencv

RUN pip3 install Pillow scipy numpy
RUN pip3 install h5py opencv-python 
RUN pip3 install rasterio 
RUN pip3 install scikit-learn 
RUN pip3 install pandas 
RUN pip3 install tensorflow_addons

RUN ls -la /sbin/ | grep ldconfig
RUN /sbin/ldconfig

WORKDIR /app
COPY . .

ENTRYPOINT ["tail", "-f", "/dev/null"]
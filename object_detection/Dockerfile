FROM tensorflow/tensorflow:2.2.0-gpu

ARG DEBIAN_FRONTEND=noninteractive
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
# Install apt dependencies
RUN apt-get update && apt-get install -y \
    git \
    libhdf5-dev \
    python3-h5py \
    python3-opencv \
    Pillow scipy numpy \
    h5py opencv-python \
    rasterio \
    sklearn \
    pandas \
    tensorflow_addons

RUN ls -la /sbin/ | grep ldconfig
RUN /sbin/ldconfig

WORKDIR /app
COPY . .

ENTRYPOINT ["tail", "-f", "/dev/null"]
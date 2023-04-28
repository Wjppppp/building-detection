FROM tensorflow/tensorflow:2.5.0-gpu

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Munich

# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# Fix GPG error
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install apt dependencies
RUN apt-get update &&\
    apt-get install -y \
    tzdata \
    git \
    protobuf-compiler
    # libhdf5-dev \
    # python3-h5py \
    # python3-opencv

RUN python -m pip install --upgrade pip

RUN cd cocoapi/PythonAPI && make &&\
    cp -r pycocotools ../../models/research &&\
    cd ../../models/research &&\
    protoc object_detection/protos/*.proto --python_out=. &&\
    cp object_detection/packages/tf2/setup.py . &&\
    python -m pip install .


# RUN pip3 install h5py opencv-python 
# RUN pip3 install rasterio 
# RUN pip3 install scikit-learn 
# RUN pip3 install pandas 
# RUN pip3 install tensorflow_addons
# RUN pip3 install tf-models-official
RUN pip install cmake


WORKDIR /app
COPY . .

# CMD [ "bash" ]
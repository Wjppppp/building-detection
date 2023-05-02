## Installation

```shell
git clone https://github.com/tensorflow/models.git
```

```shell
# From the root of the git repository
docker build -f research/object_detection/dockerfiles/tf2/Dockerfile -t od .
docker run -it od
```

```shell
cd models/research
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python -m pip install --use-feature=2020-resolver .
```

```shell
# Test the installation.
python object_detection/builders/model_builder_tf2_test.py
```

```shell
# From within TensorFlow/models/research/

protoc object_detection/protos/*.proto --python_out=.

git clone https://github.com/cocodataset/cocoapi.git

cd cocoapi/PythonAPI && make

cp -r pycocotools <PATH_TO_TF>/TensorFlow/models/research/

# From within TensorFlow/models/research/

cp object_detection/packages/tf2/setup.py .

python -m pip install .
```

## Docker

#### Build

```shell
docker build -t tf_od:<TAG> .
# This may take more than 1.5 hour
```
If building failed, please refer to https://github.com/tensorflow/models/issues/10951

#### Run

```shell
docker run -it --gpus all --name tf2_od -p 8888:8888 --mount type=bind,source="$(pwd)",target=/app tf_od:<TAG>
```


? with tf2.5.0 docker images, tf==2.6.2 was installed


#### Run Jupyter notebook inside Docker container

```shell
jupyter notebook --ip=0.0.0.0 --no-browser --allow-root --debug
```


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
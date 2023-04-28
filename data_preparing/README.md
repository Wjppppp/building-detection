# Preparing training data sets with OSM data

This repository is heavily relied on ohsome2label (https://github.com/GIScience/ohsome2label).

## Build

### Docker

- Build an ohsome2label image
  
    ```shell
    docker build -t ohsome2label:<TAG> .
    ```

- Run a container

    ```shell
    docker run --name ohsome2label ohsome2label:<TAG>
    # bind mount
    docker run -it --gpus all --name ohsome2label --mount type=bind,source="$(pwd)"/label,target=/app ohsome2label:<TAG>
    ```

- Start/Stop a container

    ```shell
    docker start ohsome2label
    #or
    docker stop ohsome2label
    ```

- Remove a container

    ```shell
    docker rm ohsome2label
    ```

- Delete an image

    ```shell
    docker rmi ohsome2label:<TAG>
    ```

## Usage

After docker images built, use teh following line to start a docker container and enter the container:

```shell
docker exec -it ohsome2label sh
```

### Install ohsome2label package

When entering the container, install ohsome2label:

```shell
pip install --editable .
export LC_ALL=C.UTF-8
```

Use `ohsome2label --help` to check the help of ohsome2label.

### Configuration

You can set the config.yaml which is under folder `config`:

```yaml
project:
  name: kalola
  workspace: ./tanzania
  project_time: 2021-3-25
  task: object detection

osm:
  api: ohsome
  url: https://api.ohsome.org/v1/elements/geometry
  bboxes: [32.463226318359375, -5.032122934090069, 32.48931884765625, -5.019810836520875]
  tags:
    - {'label': 'building', 'key': 'building', 'value': ''}
  timestamp: 2020-10-20
  types: polygon

image:
  img_api: bing
  img_url: http://t0.tiles.virtualearth.net/tiles/a{q}.png?g=854&mkt=en-US&token={token}
  api_token : 'REPLACE BY YOURTOKEN'
  zoom: 18

```

*Note: To know more about the meaning of parameter, please refer to the homepage of [ohsome2label](https://github.com/GIScience/ohsome2label/tree/master/ohsome2label). Moreover, you need to provide user specific `api_token` (e.g., for Bing satellite image, [you can apply here](https://www.bingmapsportal.com/) )*

### Run ohsome2label

- Download OSM data:

  ```shell
  ohsome2label vector
  ```

- Generate labels:

  ```shell
  ohsome2label label
  ```

- Download satellite images

  ```shell
  ohsome2label image
  ```

- Preview the labels and satellite images

  ```shell
  ohsome2label visualize -t overlay
  ```

### Convert labeled images to training records

#### Install Tensorflow object detection API (1.14.0)

First, you need to install COCO API

``` shell
cd TF_od/cocoapi/PythonAPI
make
cp -r pycocotools <path_to_tensorflow_models>/models/research/
```

Then, you need to compile the protobuf

``` shell
# From models/research/
protoc object_detection/protos/*.proto --python_out=.
```

Then, add Libraries to PYTHONPATH

``` shell
# From models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

Tips: This command needs to run from every new terminal you start. If you wish to avoid running this manually, you can add it as a new line to the end of your ~/.bashrc file.



#### Test Tensorflow

``` shell
python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

#### Test Tensorflow Object Detection API

From folder models/research/, run:

``` shell
python object_detection/builders/model_builder_test.py
```

#### Convert to training records

```shell
python tf_record_from_coco.py --label_input=./tanzania --train_rd_path=./tanzania/train.record --valid_rd_path=./tanzania/valid.record
```

Eventually, you can find the training record and validation record under your workspace folder.
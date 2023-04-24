## Build

### Docker

- Build an image
  
    ```shell
    docker build -t ohsome2label:0.1 .
    ```

- Run a container

    ```shell
    docker run --name ohsome2label ohsome2label:0.1
    # bind mount
    docker run -d --name ohsome2label --mount type=bind,source="$(pwd)"/label,target=/app ohsome2label:0.1
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
    docker rmi ohsome2label:0.1
    ```

python3 setup.py install

python tf_record_from_coco.py --label_input=./tanzania --train_rd_path=./tanzania/train.record --valid_rd_path=./tanzania/valid.record
python3 object_detection/builders/model_builder_test.py

tf_slim
keras
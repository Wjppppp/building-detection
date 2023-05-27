# TensorFlow 2 Object Detection API Workspace

Please follow the `walkthrough.ipynb`

## Reference

https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html

### Running

#### Traning Bash

```shell
cd workspace
./train.sh
```

When facing the configuration of pipeline, please copy **pipeline_template.config** or https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#configure-the-training-pipeline.

#### Walkthrough

If you want to train the model step by step, please copy **walkthrough.ipynb** under your working folder, and follow it step by step. 

*Run Jupyter notebook inside Docker container*

```shell
jupyter notebook --ip=0.0.0.0 --no-browser --allow-root --debug
```

#### Detach from docker container

Ctrl+P followed by Ctrl+Q

#### TensorBoard

```shell
docker exec -it <container_name> bash
tensorboard --port 6006 --host 0.0.0.0  --logdir 'models/my_ssd_resnet50_v1_fpn'
```

Visit tensorboard via http://0.0.0.0:6006/
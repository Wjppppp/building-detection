#! /bin/bash

ohsome2label vector

ohsome2label label

ohsome2label image

ohsome2label visualize -t overlay

# get the project name from config
name=$(yq -r '.project.name' config/config.yaml)
echo $name

# install object detection api
cd TF_od/models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ../../../

# convert to .record
python tf_record_from_coco.py --label_input=./data/$name --train_rd_path=./data/$name/train.record --valid_rd_path=./data/$name/valid.record

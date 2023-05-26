# Description: Prepare tensorflow traning samples .record with ohsome2label.
# Author: Jiapan Wang
# Created Date: 05.05.2023

#! /bin/bash

for filename in $(ls config/)
do
  echo $filename
done;

read -p "Enter the config file you want to use: " CONFIGNAME

echo $CONFIGNAME

# ohsome2label --config config/$CONFIGNAME vector

# # start training
# echo "Do you have correct vector data now?"
# select yn in "Yes" "No"; do
#     case $yn in
#         Yes ) break;;
#         No ) exit;;
#     esac
# done


# ohsome2label --config config/$CONFIGNAME label

# ohsome2label --config config/$CONFIGNAME image

ohsome2label --config config/$CONFIGNAME visualize -t overlay

# get the project name from config
name=$(yq -r '.project.name' config/$CONFIGNAME)
echo $name

# install object detection api
cd TF_od/models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ../../../

# convert to .record
# start training
echo "Do you want to make tf records now?"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) break;;
        No ) exit;;
    esac
done
python tf_record_from_coco.py --label_input=./data/$name --train_rd_path=./data/$name/train.record --valid_rd_path=./data/$name/valid.record

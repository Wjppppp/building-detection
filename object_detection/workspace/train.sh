# Description: The process for training the building detection model with .record samples and exporting the trained model.
# Author: Jiapan Wang
# Created Date: 09.05.2023

#! /bin/bash
read -p "Enter a project name: " NAME

# prepare workspace
mkdir -p $NAME/annotations
mkdir -p $NAME/images
mkdir -p $NAME/pre-trained-models
mkdir -p $NAME/models
mkdir -p $NAME/exported-models

# create label map
LABEL_MAP=$NAME/annotations/label_map.pbtxt

if [ -f "$LABEL_MAP" ]; then
    echo "$LABEL_MAP exists."
else     
    touch $NAME/annotations/label_map.pbtxt
    echo -e "item {\n\tid: 1\n\tname: 'building'\n}" >> $NAME/annotations/label_map.pbtxt
    echo "$LABEL_MAP created."
fi

# prepare traning data
while [ ! -f "$NAME/annotations/train.record" ]
do
    echo "please prepare the traning record under $NAME/annotations/"
done
echo "traning record is ready"

# download pre-trained model
cd $NAME/pre-trained-models
if [ -f "ssd_resnet101_v1_fpn_640x640_coco17_tpu-8/pipeline.config" ]; then
    echo "ssd_resnet101_v1_fpn_640x640_coco17_tpu-8 exists."
else     
    curl http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz | tar -xz 
    echo "ssd_resnet101_v1_fpn_640x640_coco17_tpu-8 downloaded."
fi

# configure the training pipeline
if [ -f "../models/my_ssd_resnet101_v1_fpn/pipeline.config" ]; then
    echo "pipeline.config exists."
else  
    mkdir -p ../models/my_ssd_resnet101_v1_fpn
    cp ssd_resnet101_v1_fpn_640x640_coco17_tpu-8/pipeline.config ../models/my_ssd_resnet101_v1_fpn/pipeline.config 
fi
echo "Now please configurate the pipeline.config under $NAME/models/my_ssd_resnet101_v1_fpn"   

# cd to $NAME/ and copy training scripts
cd ..
cp ../model_main_tf2.py model_main_tf2.py
cp ../exporter_main_v2.py exporter_main_v2.py
ls

# start training
echo "Do you want to start training?"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) python model_main_tf2.py --model_dir=models/my_ssd_resnet101_v1_fpn --pipeline_config_path=models/my_ssd_resnet101_v1_fpn/pipeline.config; break;;
        No ) break;;
    esac
done

# evaluating the model
echo "Do you want to evaluate the trained model?"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) python model_main_tf2.py --model_dir=models/my_ssd_resnet101_v1_fpn --pipeline_config_path=models/my_ssd_resnet101_v1_fpn/pipeline.config --checkpoint_dir=models/my_ssd_resnet101_v1_fpn; break;;
        No ) break;;
    esac
done


# Export the trained model
echo "Do you want to export the trained model?"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) python exporter_main_v2.py --input_type image_tensor --pipeline_config_path models/my_ssd_resnet101_v1_fpn/pipeline.config --trained_checkpoint_dir models/my_ssd_resnet101_v1_fpn/ --output_directory exported-models/my_model; break;;
        No ) exit;;
    esac
done



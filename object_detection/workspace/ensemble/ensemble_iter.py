# =======================================================================================================================================================
# ensemble.py
# Author: Jiapan Wang
# Created Date: 02/06/2023
# Description: Ensemble predications generated from different object detection models.
# =======================================================================================================================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    
import warnings
import pathlib
import numpy as np
import json
import math
import geojson
from tqdm import tqdm
from ensemble_boxes import *
import time
from prediction_to_geojson import detection_to_geojson, merge_all_geojson_to_one

PREDICTION_DIR = './predictions/'
IMAGE_SIZE = 256

def normalize_box(box):

    (x1, y1, x2, y2) = box[0]/IMAGE_SIZE, box[1]/IMAGE_SIZE, box[2]/IMAGE_SIZE, box[3]/IMAGE_SIZE
    return [x1, y1, x2, y2]

def load_weights(weights_path):

    with open(weights_path,"r") as file:
        weights_dict_list = json.load(file)
    
    # print(type(weights_dict_list))

    return weights_dict_list

def ensemble(weights_path, weights_type):
    pred_dir = os.listdir(PREDICTION_DIR)
    pred_dir.pop(0) # pop prediction-base-model
    ensemble_dir = pred_dir.pop(0) # ensemble target directory

    print("ensemble dir: ", ensemble_dir)
    print("prediction dir: ", pred_dir)

    json_path = PREDICTION_DIR + pred_dir[0] + "/json"
    json_list = os.listdir(json_path) # geojson filename list

    # print("json dir: ", json_list)

    weights_dict_list = load_weights(weights_path)
    print(weights_dict_list[0])

    # ensemble for each tile
    for i, json_filename in enumerate(tqdm(json_list)):

        # bbox, score, label combined from multiple models
        boxes_list = []
        score_list = []
        label_list = []      

        task_id = os.path.splitext(os.path.basename(json_filename))[0]

        for pred_model in pred_dir:

            # bbox, score, label from single model
            bbox = []
            score = []
            label = []

            # predictions/model/json/task_id.geojson
            input_path = PREDICTION_DIR + pred_model + "/json/" + json_filename 
            # print("\npredictions from path: ", input_path)

            with open(input_path, 'r') as input_file:
                geojson_dict = geojson.load(input_file)
                # print("geojson: ", geojson_dict['features'])
            geojson_dict['features']

            for feature in geojson_dict['features']:
                label.append(feature['properties']['prediction_class'])
                score.append(feature['properties']['score'])
                bbox.append(normalize_box(feature['properties']['bbox']))

            # print("bbox: \n", bbox)
            # print("score: \n", score)
            # print("label: \n", label)

            boxes_list.append(bbox)
            score_list.append(score)
            label_list.append(label)

        #     # break
        # print("bbox list: \n", boxes_list)
        # print("score list: \n", score_list)
        # print("label list: \n", label_list)

        iou_thr = 0.5
        skip_box_thr = 0.0001
        conf_type = "box_and_model_avg"
        # sigma = 0.1


        ################################################################################### weights choice
        weight_dict = weights_dict_list[i]

        if weights_type == "average":
            weights = [1/8,1/8,1/8,1/8,1/8,1/8,1/8,1/8]
        elif weights_type == "distance":
            weights = weight_dict["inverse_distance_weights"]
        elif weights_type == "similarity":
            weights = weight_dict["image_similarity_weights"]
        elif weights_type == "attention":
            weights = weight_dict["attention_map_weights"]
        else:
            warnings.warn('Please select an appropriate weight type: "average" or "distance" or "similarity" or "attention"')
            exit()

        if weight_dict["tile_id"] == task_id:
            # print("weight_dict", weight_dict)
            print("weight list", weights)
        else:
            print("Didn't find weights!!!", task_id)

        boxes, scores, labels = weighted_boxes_fusion(boxes_list, score_list, label_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr, conf_type=conf_type)
        boxes = (np.array(boxes)*256).astype(np.int)  

        # print("ensemble bbox list: \n", boxes)
        # print("ensemble score list: \n", scores)
        # print("ensemble label list: \n", labels)
        # print("task_id", task_id)

        # task_id = os.path.splitext(os.path.basename(task_id))[0]
        # print("task_id", task_id)

        output_path = PREDICTION_ENSEMBLE_DIR + "json/"
        detection_to_geojson(task_id, boxes, labels, scores, output_path)

        # break

    input_dir = PREDICTION_ENSEMBLE_DIR + "json/"
    geojson_all = merge_all_geojson_to_one(input_dir)

    output_path = f"{PREDICTION_ENSEMBLE_DIR}merged_prediction.geojson"
    with open(output_path, 'w', encoding='utf-8') as output_file:
        geojson.dump(geojson_all, output_file)

    print("done")

    return

if __name__ == "__main__":

    print("hello, ensemble")
    start_time = time.time()
    weights_dir = os.listdir("./weights")

    for i, weight_json in enumerate(weights_dir):

        PREDICTION_ENSEMBLE_DIR = f'./random/prediction-ensemble-{i}/'   
        if not os.path.exists(PREDICTION_ENSEMBLE_DIR):
            os.makedirs(PREDICTION_ENSEMBLE_DIR) 

        weight_path = f"./weights/{weight_json}"

        ensemble(weight_path, "attention")

    print(f"time used: {time.time() - start_time}")

    

    

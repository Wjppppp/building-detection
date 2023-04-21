"""
Python script to obtain binary classification result from the object detection result geojson file.

Hao Li 
14.05.2018 
Heigit, Heidelberg 

Usage:
           python binary_od.py    --OD_result=prediction_result/OD_prediction.geojson
                                  --OSM_gt=prediction_result/index_tiles_GT.lst

"""

import os
from os import makedirs, path as op
import sys
import glob
import six.moves.urllib as urllib
import tensorflow as tf
import tarfile


from io import StringIO
import zipfile
import numpy as np
from collections import defaultdict

from PIL import ImageDraw, Image

import json
from geojson import Feature, FeatureCollection
from shapely.geometry import Polygon
import pandas as pd
import geopandas as gp

sys.path.append("..")

from utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('OD_result', '', 'Path to OD geojson file ')
flags.DEFINE_string('OSM_gt', '', 'Path to OSM gt file ')
FLAGS = flags.FLAGS


def probability(task_groupby, lists_osm_gt, threshold):
    # this is the probability of there is at least one object in the image,
    # threshlod setting should be automatically selected later
    tile_id = []
    tag = []
    prob = []
    d = {}

    for task in task_groupby:
        p = 1
        scores = task.score
        tile_id_i = task.task_id.unique()[0]
        score_min=50
      
        for score in scores:
           # here we only consider the score bigger than 50 as candidates.
           if score > score_min:
             s = (100-float(score))/100
             p = p*s
        total_score = 1-p
        if total_score >= threshold:
           tag_i = 1
        else:
           tag_i = 0
        for list_osm_gt in lists_osm_gt:
           list_osm_gt = list_osm_gt.split()
           if list_osm_gt[0] == tile_id_i:
              tag_osm = 1
              break
           else:
              tag_osm = 0
        prob.append(total_score)
        tile_id.append(tile_id_i)
        tag.append(tag_i)
        d_small = { 'prediction_id': tag_i , 'probability': total_score , 'osm_tag' : tag_osm, 'mapswipe_tag' : 0}
        d[tile_id_i] = d_small
        

    output_json = pd.DataFrame(d)
    
    filename = 'OD_prediction_binary' + '.json'
    output_dir = 'prediction_result'
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    
    
    out_path = os.path.join(output_dir, filename)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(output_json.to_json())

      
 

    
if __name__ =='__main__':
    path_to_od_result = op.join(os.getcwd(), FLAGS.OD_result)
    path_to_osm_gt = op.join(os.getcwd(), FLAGS.OSM_gt)
    # Path to object detection result with respect to each tile.
    df = gp.read_file(path_to_od_result)
    list_gt = open(path_to_osm_gt,'r')
    list_gt = list_gt.readlines()
    # Read the geojson file 
    prediction = pd.DataFrame()
    task_id = df['task_id']
    score = df['score']
    prediction ['task_id'] = task_id
    prediction ['score'] = score
    prediction_task = prediction.groupby(['task_id'])
    prediction_task = [prediction_task.get_group(x) for x in prediction_task.groups]
    threshold = 0.5
    probability(prediction_task, list_gt, threshold)
    print("{} tiles got the binary prediction tag, using threshold value: {} !".format(len(prediction_task)+1, threshold )  )
    

 

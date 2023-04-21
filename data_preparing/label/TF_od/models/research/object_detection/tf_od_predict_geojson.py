"""
This is adapted from Tensorflow (https://github.com/tensorflow/models/tree/master/research/object_detection);
Save this code under the directory `models/research/object_detection/`

To use, run:
python tf_od_predict.py --model_name=building_od_ssd \
                         --path_to_label=data/building_od.pbtxt \
                         --test_image_path=test_images
"""

import os
from os import makedirs, path as op
import sys
import glob
import six.moves.urllib as urllib
import tensorflow as tf
import tarfile
from tqdm import tqdm


from io import StringIO
import zipfile
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from PIL import ImageDraw, Image

import json
from geojson import Feature, FeatureCollection
from shapely.geometry import Polygon
import pandas as pd
import geopandas as gp

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

flags = tf.app.flags
flags.DEFINE_string('model_name', '', 'Path to frozen detection graph')
flags.DEFINE_string('path_to_label', '', 'Path to label file')
flags.DEFINE_string('test_image_path', '', 'Path to test imgs and output diractory')
FLAGS = flags.FLAGS

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
	
	
	
def OutputGeojson( task_id, prediction_id, score, bbox):
    task_id_Seri = pd.Series(task_id)
    prediction_id_Seri = pd.Series(prediction_id)
    score_Seri = pd.Series(score)
    bbox_GeoSeri = gp.GeoSeries([Polygon(box) for box in bbox])

    d = {'task_id':task_id_Seri, 'prediction_id': prediction_id_Seri, 'bbox': bbox_GeoSeri, 'score': score_Seri}
    keywordsDf = gp.GeoDataFrame(d, geometry='bbox')

    filename = 'OD_prediction' + '.geojson'

    output_dir = 'prediction_result'
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
      
    out_path = os.path.join(output_dir, filename)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(keywordsDf.to_json(ensure_ascii = False))


		

def tf_od_pred():
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
			
            #for all the prediction bbox without of filter, list of [N, 4]
            bbox = []
            # for the corresponding score of shape [N] or None.  If scores=None, then 
            # this function assumes that the boxes to be plotted are groundtruth
            score = []
            #for the task_id 
            task_id = []
            #for the prediction bbox id
            prediction_id = []
	
            for image_path in tqdm(test_imgs):
                image = Image.open(image_path)
                image_np = load_image_into_numpy_array(image)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                  [detection_boxes, detection_scores, detection_classes, num_detections],
                  feed_dict={image_tensor: image_np_expanded})
                # draw_bounding_box_on_image(image, boxes, )
                # Visualization of the results of a detection.
                vis_image = vis_util.visualize_boxes_and_labels_on_image_array(
                         image_np,
                         np.squeeze(boxes),
                         np.squeeze(classes).astype(np.int32),
                         np.squeeze(scores),
                         category_index,
                         use_normalized_coordinates=True,
                         line_thickness=1)
                print("{} boxes in {} image tile!".format(len(boxes), image_path))
                image_pil = Image.fromarray(np.uint8(vis_image)).convert('RGB')
                with tf.gfile.Open(image_path, 'w') as fid:
                     image_pil.save(fid, 'PNG')
                
                M = tf.image.non_max_suppression(np.squeeze(boxes), np.squeeze(scores), boxes.shape[1], iou_threshold=0.3)
                M = M.eval()
                task = os.path.splitext(os.path.basename(image_path))[0]
                scores = np.squeeze(((scores*100).transpose()).astype(np.int))
                scores_ls = scores.tolist()
                scores_ls = [scores_ls[i] for i in M]
                bboxe = (boxes*256).astype(np.int)
                bboxe = np.squeeze(bboxe)
                if bboxe.any():
                    bboxe_ls = bboxe.tolist()
                    bboxe_ls = [bboxe_ls[i] for i in M]

                for count, box in enumerate(bboxe_ls, start=0):
                    box = [max(0, min(255, x)) for x in box[:4]]
                    (left, right, top, bottom) = (box[0], box[2], box[1], box[3])
                    box_polygon=[(left, top), (left, bottom), (right, bottom), (right, top), (left, top)]
                    bbox.append(box_polygon)
                    #prediction bbox polygon
            
            
                    task_id.append(task)
			
                    #ID in total prediction
                    num_box = count+1
                    prediction_id.append(num_box )
            
                    #score of each box prediction
                    score.append(scores_ls[count])
        
            OutputGeojson( task_id, prediction_id, score, bbox)
   
           
            
      
            

            
            
        
		    
		
		   
		  
		   
		
		



if __name__ =='__main__':
    # load your own trained model inference graph. This inference graph was generated from
    # export_inference_graph.py under model directory, see `models/research/object_detection/`
    model_name = op.join(os.getcwd(), FLAGS.model_name)
    # Path to frozen detection graph.
    path_to_ckpt = op.join(model_name,  'frozen_inference_graph.pb')
    # Path to the label file
    path_to_label = op.join(os.getcwd(), FLAGS.path_to_label)
    #only train on buildings
    num_classes = 1
    #Directory to test images path
    test_image_path = op.join(os.getcwd(), FLAGS.test_image_path)
    test_imgs = glob.glob(test_image_path + "/*.png")

    ############
    #Load the frozen tensorflow model
    #############

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    ############
    #Load the label file
    #############
    label_map = label_map_util.load_labelmap(path_to_label)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    tf_od_pred()


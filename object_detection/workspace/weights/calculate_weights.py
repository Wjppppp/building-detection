# =======================================================================================================================================================
# calculate_weights.py
# Author: Jiapan Wang
# Created Date: 01/06/2023
# Description: Calculate the weight of distance and image correlation between the reference area and the target area.
# =======================================================================================================================================================

from math import sin, cos, acos, atan2, radians, pi
from sklearn.preprocessing import normalize
import numpy as np
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import random

IMAGE_DIR = './sample_images/'

def get_center_latlon(lon1, lat1, lon2, lat2):

    center_lat = abs(lat1 - lat2)/2 + min(lat1, lat2)
    center_lon = abs(lon1 - lon2)/2 + min(lon1, lon2)

    return center_lon, center_lat


def calculate_distance(lon1, lat1, lon2, lat2):
    """
    Calculate distance in meters between two latitude, longitude points.

    Law of cosines: d = acos( sin φ1 ⋅ sin φ2 + cos φ1 ⋅ cos φ2 ⋅ cos Δλ ) ⋅ R
    ACOS( SIN(lat1)*SIN(lat2) + COS(lat1)*COS(lat2)*COS(lon2-lon1) ) * 6371000
    """
    R = 6371.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    
    distance = acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos((lon2 - lon1))) * R * 1000
    # print("distance", distance) # m

    return distance

def distance_weights():
    c_lat = np.zeros(8)
    c_lon = np.zeros(8)
    distance = np.zeros(8)

    bbox_list = [
        [10.1808867479861433,5.6779222981920654,10.1854357002290943,5.6815842067768036],
        [10.2044794916899786,5.6874603256778036,10.2081243554005656,5.6910385317555923],
        [10.2369822404324680,5.6739091434510103,10.2427961658346156,5.6787338574077060],
        [10.1758264495190556,5.6589049871613826,10.1808537742712719,5.6624908025580210],
        [10.2405342704539137,5.6561765898082879,10.2458976227248382,5.6612797033758442],
        [10.1783557870918955,5.6422238384631669,10.1850979479314638,5.6475389476528628],
        [10.2071616243839944,5.6417252758946166,10.2115071541707554,5.6463194782201533],
        [10.2613935439247506,5.6314816109111181,10.2656648067598848,5.6353477536127228]
    ] 

    c_lon_target, c_lat_target = get_center_latlon(10.1926316905051912,5.6555060217455528,10.2352523789601566,5.6722064157053831)

    for i in range(0, 8):
        # print("bbox", bbox_list[i][0], bbox_list[i][1], bbox_list[i][2], bbox_list[i][3])
        c_lon[i], c_lat[i] = get_center_latlon(bbox_list[i][0], bbox_list[i][1], bbox_list[i][2], bbox_list[i][3])
        distance[i] = calculate_distance(c_lon[i], c_lat[i], c_lon_target, c_lat_target)
    

    # print("center:", c_lat, "\n", c_lon)
    # print("distance:", distance)
    # print("inverse distance", 1.0/distance)

    # distance weights
    print("Distance weights:")
    weights = normalize_weights(distance)

    # inverse distance weights
    print("Inverse distance weights:")
    inv_weights = normalize_weights(1.0/distance)

    return distance, inv_weights

# Load images
def load_images(path):
    
    # Get the list of all files and directories
    dir_list = os.listdir(path)
    # print("Files and directories in '", path, "' :")
    # prints all files
    # print(dir_list)
    
    filenames = dir_list
    image_paths = []
    for filename in filenames:
#         image_path = tf.keras.utils.get_file(fname=filename,
#                                             origin=path + filename,
#                                             untar=False)
        image_path = pathlib.Path(path+filename)
        image_paths.append(str(image_path))

    return image_paths

def calculate_image_similarity(img1, img2, channel):
    
    similarity = 0

    for i in range(channel):
        
        hist_similarity = cv2.compareHist(cv2.calcHist([img1], [i], None, [256], [0, 256]), cv2.calcHist([img2], [i], None, [256], [0, 256]), cv2.HISTCMP_CORREL)
        similarity += hist_similarity
        # print(i, "similarity", hist_similarity)
    #     hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    #     hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])

    # similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    # gray_similarity = cv2.compareHist(cv2.calcHist([img1_gray], [0], None, [256], [0, 256]), cv2.calcHist([img2_gray], [0], None, [256], [0, 256]), cv2.HISTCMP_CORREL)
    similarity = similarity / channel
    print("similarity", similarity)
    # print("gray_similarity", gray_similarity)
    return similarity

def image_similarity_weights():

    ref_dirs = os.listdir(IMAGE_DIR)
    
    # ref_paths = load_images(IMAGE_DIR)
    target_dir = ref_dirs.pop()
    print(target_dir)

    # target image path list
    target_image_path = IMAGE_DIR+target_dir+"/"
    target_image_paths = load_images(target_image_path)
    # print(target_image_paths)

    average_similarity = []

    for i, ref_dir in enumerate(ref_dirs):
        print("Image similarity between {} and {}".format(ref_dir, target_image_path))
        # reference image path list
        ref_image_paths = load_images(IMAGE_DIR+ref_dir+"/")
        # print("ref",ref_image_paths)
        length = len(ref_image_paths)
        # print(length)

        # pick random image samples from target area 
        target_image_samples = random.sample(target_image_paths, length)
        # print("samples", target_image_samples)

        similarity = 0

        for j, image_path in enumerate(ref_image_paths):

            # print(i, image_path)
            # print("target", target_image_samples[i])

            img1 = cv2.imread(image_path)
            img2 = cv2.imread(target_image_samples[j])

            similarity += calculate_image_similarity(img1, img2, 3)

        average_similarity.append(similarity/len(ref_image_paths))

        print("Average Similarity between {} and {} is {} \n".format(ref_dir, target_image_path, average_similarity[i]))     

        # img1 = cv2.imread('img5.png')
        # img2 = cv2.imread('img2.png')

    print("Average Similarity List", average_similarity)
    similarity_weights = normalize_weights(np.array(average_similarity))

    return average_similarity, similarity_weights

def normalize_weights(weight):

    norm_weights = normalize(weight[:,np.newaxis], axis=0, norm='l1').ravel()
    print("weights:", norm_weights, "\nsum of weights:",sum(norm_weights), "\n")

    return norm_weights



if __name__ == '__main__':

    print("image similarity weights:")
    similarity, similarity_weight = image_similarity_weights()

    print("distance weights:")
    distance, distance_weight = distance_weights()

    # # Writing to file
    # with open("weights.txt", "a") as file:
    #     # Writing data to a file
    #     file.write("image similarity: \n{}\n".format(similarity))
    #     file.write("image similarity weights: \n{}\n".format(similarity_weight.tolist()))
    #     file.write("distance: \n{}\n".format(distance.tolist()))
    #     file.write("distance weights: \n{}\n".format(distance_weight.tolist()))
    #     file.write("==================================================================================================================================\n")
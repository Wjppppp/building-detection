# =======================================================================================================================================================
# weights_uncertainty.py
# Author: Jiapan Wang
# Created Date: 04/08/2023
# Description: Evaluation weights uncertainty.
# =======================================================================================================================================================

from math import sin, cos, acos, atan2, radians, pi, atan, exp
from sklearn.preprocessing import normalize
import numpy as np
from PIL import Image
import time
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import random
import json
from tqdm import tqdm
from merge_image_patch import merge_images
from vit_representations import get_image_attention_weights, load_model, MODELS_ZIP

IMAGE_DIR = './sample_images/'
WORKSPACE = './uncertainty/'

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


def normalize_weights(weight):

    norm_weights = normalize(weight[:,np.newaxis], axis=0, norm='l1').ravel()
    print("weights:", norm_weights, "\nsum of weights:",sum(norm_weights), "\n")

    return norm_weights

def tile_to_ref_attention_weights(tile_id, vit_model):
    target_dir = os.listdir(IMAGE_DIR)
    target_dir.sort()
    target_dir = target_dir.pop()

    target_image_path = IMAGE_DIR + target_dir + "/" + tile_id + ".png"
    tile_image = cv2.imread(target_image_path)

    vit_sample_image_dir = './ViT_sample_images/'

    # replace center image patch
    tile_image_new_file = f"{vit_sample_image_dir}5.png"
    cv2.imwrite(tile_image_new_file, tile_image)

    # replace surrounding image patch
    for i in [1,2,3,4,6,7,8,9]:
        ref_image_dir = f"./sample_images/ref_0{i}/"
        ref_image_list = os.listdir(ref_image_dir)
        ref_image_path = random.choice(ref_image_list)
        ref_image_path = ref_image_dir + ref_image_path

        # print(f"random ref {i} image path: {ref_image_path}")
        # input("Press Enter to continue...")

        ref_image = cv2.imread(ref_image_path)

        ref_image_new = f"{vit_sample_image_dir}{i}.png"
        cv2.imwrite(ref_image_new, ref_image)

    attention_maps_dir = f'{WORKSPACE}attention_maps/attention_map'
    if not os.path.exists(attention_maps_dir):
        os.makedirs(attention_maps_dir)

    merged_image_dir = f'{WORKSPACE}attention_maps/merged_image'
    if not os.path.exists(merged_image_dir):
        os.makedirs(merged_image_dir)

    # merge 9 image patches to one
    merged_image = merge_images(vit_sample_image_dir)
    im = Image.fromarray(merged_image, 'RGB')
    im.save(f"{merged_image_dir}/{tile_id}_merged.png")
    print(f"merging {tile_id}...done!")

    # generate attention map and weights
    img_path = f"{merged_image_dir}/{tile_id}_merged.png"

    model_type = "dino"

    attention_weights = get_image_attention_weights(tile_id, img_path, vit_model, attention_maps_dir, model_type)

    # delete weight of center patch
    attention_weights.pop(4)
    return normalize_weights(np.array(attention_weights))

def tile_to_ref_weights():

    target_tile_image_dir = f"{IMAGE_DIR}target/"
    tile_dir = os.listdir(target_tile_image_dir)
    # print("tile paths: ", tile_dir)

    weights_dict_list = []

    # Load the model. 
    vit_model = load_model(MODELS_ZIP["vit_dino_base16"])
    print("Model loaded.")

    for i, tile_name in enumerate(tqdm(tile_dir)):

        # image id
        tile_id = os.path.splitext(os.path.basename(tile_name))[0]

        print(f"\nCalculating weights for {i}, {tile_id}\n")

        # ViT-DINO attention weights
        attention_weights = tile_to_ref_attention_weights(tile_id, vit_model)

        # weight dictionary for each tile
        new_weight = {
            "tile_id": tile_id,
            "attention_map_weights": attention_weights.tolist(),
        }
        weights_dict_list.append(new_weight)

        print("\n",tile_name, new_weight)


    # print("weight dict list: ", weights_dict_list)
    return weights_dict_list



if __name__ == '__main__':

    print("Start calculating weights ...")
    start_time = time.time()
    
    # Writing to json file

    for i in range(3, 11):

        weight_dict_list = tile_to_ref_weights()

        with open(f"{WORKSPACE}all_weights_{i}.json","w", encoding='utf-8') as file:
            json.dump(weight_dict_list, file)

        print(f"done. time used: {time.time() - start_time}")
    # # Writing to file
    # with open("weights.txt", "a") as file:
    #     # Writing data to a file
    #     file.write("image similarity: \n{}\n".format(similarity))
    #     file.write("image similarity weights: \n{}\n".format(similarity_weight.tolist()))
    #     file.write("distance: \n{}\n".format(distance.tolist()))
    #     file.write("distance weights: \n{}\n".format(distance_weight.tolist()))
    #     file.write("==================================================================================================================================\n")
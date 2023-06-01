# =======================================================================================================================================================
# merge_image_patch.py
# Author: Jiapan Wang
# Created Date: 30/05/2023
# Description: merge 3*3 image patches into one big image patch.
# =======================================================================================================================================================
"""


"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import numpy as np
from PIL import Image

IMAGE_DIR = './images_03_target_area_without_buildings/'
IMAGE_PATCH_SIZE = 256
IMAGE_PATCH_NUMBER_ROW = 3
IMAGE_PATCH_NUMBER_COL = 3
CHANNEL = 3

# Load images
def load_images(path):
    
    # Get the list of all files and directories
    dir_list = os.listdir(path)
    # print("Files and directories in '", path, "' :")
    # prints all files
    print(dir_list)
    
    filenames = dir_list
    image_paths = []
    for filename in filenames:
#         image_path = tf.keras.utils.get_file(fname=filename,
#                                             origin=path + filename,
#                                             untar=False)
        image_path = pathlib.Path(path+filename)
        image_paths.append(str(image_path))
    return image_paths

def parse_tile_name(name):
    index, zoom, TileX, TileY = [int(x) for x in name.split(".")]
    return index, TileX, TileY, zoom

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))



if __name__ == '__main__':

    image_paths = load_images(IMAGE_DIR)

    image_list = []
    
    for image_path in image_paths:

        patch_id = os.path.splitext(os.path.basename(image_path))[0]
        index, TileX, TileY, zoom = parse_tile_name(patch_id)

        image_np = load_image_into_numpy_array(image_path)
        image_list.append(image_np)
        print('{} Image patch {} loaded... \n'.format(index, image_path), end='')
        # print(np.shape(image_np), image_np)

    # print("image_list",np.shape(image_list), image_list)

    image_row1 = np.concatenate((image_list[0], image_list[1], image_list[2]), axis=1)
    image_row2 = np.concatenate((image_list[3], image_list[4], image_list[5]), axis=1)
    image_row3 = np.concatenate((image_list[6], image_list[7], image_list[8]), axis=1)
    merged_image = np.concatenate((image_row1, image_row2, image_row3), axis=0)

    im = Image.fromarray(merged_image, 'RGB')
    im.save("merged.png")
    print("merging...done!")

    
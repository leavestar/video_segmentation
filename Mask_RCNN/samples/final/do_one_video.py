import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# use davis format to save
from davis import *

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


FOLDER_NAMES = [_ for _ in cfg.SEQUENCES.keys()]
#TODO hardcoded Directory of images to run detection on
# let's do this iteratively
IMAGE_DIR_ = os.path.join(ROOT_DIR, "../davis-2017/data/DAVIS/JPEGImages/480p/")
SAVE_DIR_ = os.path.join(ROOT_DIR, "../davis-2017/data/DAVIS/MaskRCNN/480p/")
print (IMAGE_DIR_)
print (SAVE_DIR_)
#

FOLDER_NAMES = ["camel"]
for folder in FOLDER_NAMES:
    # try:
    IMAGE_DIR = IMAGE_DIR_ + folder + "/"
    FILES = sorted([file for file in os.listdir(IMAGE_DIR) if file.endswith(".jpg")])
    SAVE_DIR = SAVE_DIR_ + folder + "/"

    start_from = 0
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    else:
        saved_ = [file for file in os.listdir(SAVE_DIR) if file.endswith(".png")]
        start_from = len(saved_)

    images = []
    files = []

    for count, file in enumerate(FILES):
        if count < start_from:
            continue
        print(folder + ": " + file + " out of " + str(len(os.listdir(IMAGE_DIR))))
        image = skimage.io.imread(os.path.join(IMAGE_DIR, file))
        images.append(image)
        files.append(file)

        # need special treatment for the last one
        if count+1 == len(FILES) and len(images) != config.IMAGES_PER_GPU:
            # force operate
            while len(images) < config.IMAGES_PER_GPU:
                images.append(image)
                files.append(file)

        if len(images) == config.IMAGES_PER_GPU:
            results = model.detect(images, verbose=0)
            # save
            for file_, result_ in zip(files, results):
                data = np.squeeze(result_["masks"])
                data = data.astype('uint8')
                if np.atleast_3d(data).shape[2] != 1:
                    # when we detect more then one items
                    if data.shape[2] == 0:
                        data_ = np.zeros((data_.shape[:2])).astype('uint8')
                    else:
                        data_ = np.squeeze(data[:, :, 0])
                    for i in range(1, data.shape[2]):
                        data_[np.squeeze(data[:, :, i]) != 0] = (i + 1)
                    data = data_
                io.imwrite_indexed(os.path.join(SAVE_DIR, file_[:-3] + "png"), data)

            # before done, reinit images and files
            images = []
            files = []
    # except:
    #     print("something wrong with " + folder)

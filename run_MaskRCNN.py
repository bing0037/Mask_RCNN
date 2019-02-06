import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
# from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# # Import Mask RCNN
# sys.path.append(os.path.join(ROOT_DIR, "mrcnn/"))  # To find local version
# from visualize import display_images


# %matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
# if not os.path.exists(COCO_MODEL_PATH):
    # utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


# 1) Configurations
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# 2) Create Model and Load Trained Weights
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# 3) Class Names
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

# 4) Run Object Detection
# Load a random image from the images folder
# file_names = next(os.walk(IMAGE_DIR))[2]

# For .jpg images:
# image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
# image = skimage.io.imread("test.jpg")
image = skimage.io.imread("2383514521_1fc8d7b0de_z.jpg")

# # For .png images:
# from PIL import Image
# im = Image.open("imageToSave_320_240.png")
# bg = Image.new("RGB", im.size, (255,255,255))
# bg.paste(im,im)
# # bg.save("Converted_image.jpg")
# # image = skimage.io.imread(name + ".jpg")
# image = np.asarray(bg)


# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])

print('Hello Motor!')                            


# Add Color Splash effect: set uninterested area in gray and interested in original color! 
def color_splash(image, mask, class_id, selected_class_id):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # To test: build a black background!
    # gray = np.zeros((image.shape[0],image.shape[1],3))

    # We're treating all instances as one, so collapse the mask into one layer
    # np.sum(data,axis,keepdims=True): calculate the sum of the data along the assigned axis: 0: first axis; 1: second axis; 2: third axis; -1: last axis. -libn      
    if selected_class_id == -1:
        # Mark all pixel points belonging to a meaningful class instead of Background "True" in mask matrix. -libn
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
    else:
        # Mark all pixel points belonging to a assigned class instead of Background "True" in mask matrix. -libn
        # Retain Only One class:
        r_selection = np.array(np.where(class_id==selected_class_id))
        mask = mask[:,:,np.array(r_selection)].reshape(image.shape[0],image.shape[1],r_selection.size)
        mask = (np.sum(mask, -1, keepdims=True) >= 1)

    # Copy color pixels from the original color image where mask is set
    # Copy image to gray according to mask. -libn
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash

# Retain Only One Class: ClassNum_Giraffe = 24; ClassNum_Person = 1; ClassAll = -1;
selected_class_id = -1
splash = color_splash(image, r['masks'], r['class_ids'], selected_class_id)

plt.imshow(splash.astype(np.uint8), cmap=None,
                   norm=None, interpolation=None)
plt.show()

# display_images([splash], cols=1)                            

# Tests:
# Test 1: To test the coordinate of the image frame.
mask = np.ones((image.shape[0],image.shape[1],1))
mask[100:110,:,0] = np.zeros((10,image.shape[1]))                         
mask[200:210,:,0] = np.zeros((10,image.shape[1])) 
mask[:,30:40,0] = np.zeros((image.shape[0],10)) 
gray = np.zeros((image.shape[0],image.shape[1],3))
splash = np.where(mask, image, gray).astype(np.uint8)
plt.imshow(splash.astype(np.uint8), cmap=None,
                   norm=None, interpolation=None)
plt.show()
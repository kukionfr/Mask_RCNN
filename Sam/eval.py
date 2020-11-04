#%%

import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import pathlib
import skimage
from PIL import Image
import imgaug
from imgaug import augmenters as iaa
from skimage.filters import threshold_otsu
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

#%%

# minimum input size = 128
class ShapesConfig(Config):
    # Give the configuration a recognizable name
    NAME = "skin"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 16
    NUM_CLASSES = 1 + 2  # background + 2 types
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    RPN_ANCHOR_SCALES = (16,32,64,128,256)  # anchor side in pixels
    TRAIN_ROIS_PER_IMAGE = 8
    STEPS_PER_EPOCH = 626 // IMAGES_PER_GPU
    VALIDATION_STEPS = 626 // IMAGES_PER_GPU
    LEARNING_RATE = 0.001
    USE_MINI_MASK = False
    # gpu_options = True
config = ShapesConfig()
def get_ax(rows=1, cols=1, size=8):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

class ShapesDataset(utils.Dataset):
     def list_images(self,data_dir):
        # define classes
        self.add_class("skin", 1, "fibroblast")
        self.add_class("skin", 2, "falsePositive")

        train_images = list(data_dir.glob('*tile*/image/*.png'))
        print('# image in this dataset : ',len(train_images))
        for idx,train_image in enumerate(train_images):
            label = str(train_image).replace("image","mask")
            self.add_image("skin",image_id=idx,path=train_image,labelpath=label,
                           height=config.IMAGE_SHAPE[0],width=config.IMAGE_SHAPE[1])

        train_images = list(data_dir.glob('*false_positive*/image/*.png'))
        print('# image in this dataset : ',len(train_images))
        for idxx,train_image in enumerate(train_images):
            label = str(train_image).replace("image","mask")
            self.add_image("skin",image_id=idx+idxx,path=train_image,labelpath=label,
                           height=config.IMAGE_SHAPE[0],width=config.IMAGE_SHAPE[1])


     def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            print('grayscale to rgb')
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            print('rgba to rgb')
            image = image[..., :3]
        # image = cv2.resize(image,dsize=(256,256))
        return image.astype(np.uint8)

     def load_mask(self, image_id):
        label = self.image_info[image_id]['labelpath']
        mask = Image.open(label)
        mask = np.array(mask).astype('int')
        mask = mask[:,:,np.newaxis]
        if 'false_positive' in label:
            class_ids = np.array([2])
        else:
            class_ids = np.array([1])
        return mask,class_ids


     def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "skin":
            return info["truth"]
        else:
            super(self.__class__).image_reference(self, image_id)

#%%

class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MAX_DIM = 128
inference_config = InferenceConfig()

#%%

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


#%%

from PIL import Image
from skimage import io

## put folder path here to apply your model to classify
src = r'\\10.162.80.6\Kyu_Sync\Aging\data\svs\20x\segmentation\Wirtz.Denis_OTS-19_5021-003_false_positive_4\image'
##
dst = os.path.join(src,'classified')
if not os.path.exists(dst): os.mkdir(dst)

images = [os.path.join(src,_) for _ in os.listdir(src) if _.endswith('png')]
idd = []
for original_image in images:
    original_image2 = skimage.io.imread(original_image)
    results = model.detect([original_image2], verbose=1)
    r = results[0]
    masks = r['masks']
    masks = np.moveaxis(masks,2,0)
    if len(masks)<1:
        continue
    maskzero=np.zeros(masks[0].shape)
    for mask,id in zip(masks,r['class_ids']):
        idd.append(id)
        maskzero[mask]=id
    im = Image.fromarray(maskzero)
    im.save(os.path.join(dst, os.path.basename(original_image).replace('png','tif')))
print(idd)

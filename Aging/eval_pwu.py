from PIL import Image
import skimage
from skimage import io
import os
import mrcnn.model as modellib
from mrcnn.config import Config
import sys
import numpy as np
from matplotlib import pyplot as plt
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library
import cv2

class ShapesConfig(Config):
    # Give the configuration a recognizable name
    NAME = "cell_10t_10f_20w"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 6  # 2 for 1024x1024, 4 for 512x512
    # 10 for resnet 50 with 2080Ti
    NUM_CLASSES = 1 + 1  # background + 2 types
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # (4,8,16,32,64) #(16,32,64,128,256)  # anchor side in pixels
    STEPS_PER_EPOCH = 4250 // IMAGES_PER_GPU
    VALIDATION_STEPS = 750 // IMAGES_PER_GPU
    LEARNING_RATE = 0.001
    USE_MINI_MASK = True
    BACKBONE = "resnet101"
    TRAIN_ROIS_PER_IMAGE = 128  # increase to detect more instances, but may face OOM issue (320 for 2080ti)
    MEAN_PIXEL = np.array([5.82, 5.82, 5.82])  # ([123.7, 116.8, 103.9])

    RPN_TRAIN_ANCHORS_PER_IMAGE = 250
    RPN_NMS_THRESHOLD = 0.7  # increase for more proposal (def=0.7)

    MAX_GT_INSTANCES = 30
    DETECTION_MAX_INSTANCES = 35
    DETECTION_NMS_THRESHOLD = 0.2

    DETECTION_MIN_CONFIDENCE = 0.6  # decrease for more proposal (def=0.7)

class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MAX_DIM = 256
inference_config = InferenceConfig()

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
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

src = r'\\motherserverdw\Kyu_Sync\temp\210115 simulated bead images for deep learning\Dataset\set5\val\val_norm_imset5'


images = [os.path.join(src,_) for _ in os.listdir(src) if _.endswith('npy')]

for original_image in images:
    image = np.load(original_image)
    image = image.astype(np.uint8)
    results = model.detect([image], verbose=1)
    r = results[0]
    masks = r['masks']
    masks = np.moveaxis(masks,2,0)
    if len(masks)<1:
        print('no object detected')
        continue
    maskzero=np.zeros(masks[0].shape)
    for idx,mask in enumerate(masks):
        maskzero[mask]= idx
    maskzero = maskzero.astype(np.uint8)
    im = Image.fromarray(maskzero)

    dst1 = os.path.join(src,'predicted_mask_cell_10t_10f_20w_8sz_2overlap_60confidence')
    if not os.path.exists(dst1): os.mkdir(dst1)
    im.save(os.path.join(dst1, os.path.basename(original_image.replace('npy','tif'))))


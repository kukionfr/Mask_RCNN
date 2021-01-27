
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import skimage
import imgaug
from imgaug import augmenters as iaa
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils, visualize
import mrcnn.model as modellib
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
import pickle
from pycocotools import mask as maskUtils
import cv2

# minimum input size = 128
class ShapesConfig(Config):
    # Give the configuration a recognizable name
    NAME = "lvl_1"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 6  # 2 for 1024x1024, 4 for 512x512
    # 10 for resnet 50 with 2080Ti
    NUM_CLASSES = 1 + 1  # background + 2 types
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    RPN_ANCHOR_SCALES = (2,4,8,16,32) #(4, 8, 16, 32, 64)  # (8,16,32,64,128)  #(16,32,64,128,256)  # anchor side in pixels
    STEPS_PER_EPOCH = 4250 // IMAGES_PER_GPU
    VALIDATION_STEPS = 750 // IMAGES_PER_GPU
    LEARNING_RATE = 0.001
    USE_MINI_MASK = True
    BACKBONE = "resnet101"

    MEAN_PIXEL = np.array([0.246]*3)  # ([123.7, 116.8, 103.9])

    MAX_GT_INSTANCES = 11

    TRAIN_ROIS_PER_IMAGE = MAX_GT_INSTANCES * 4  # increase to detect more instances, but may face OOM issue (320 for 2080ti)
    RPN_TRAIN_ANCHORS_PER_IMAGE = TRAIN_ROIS_PER_IMAGE * 5
    DETECTION_MAX_INSTANCES = MAX_GT_INSTANCES+1
    DETECTION_NMS_THRESHOLD = 0.2
    RPN_NMS_THRESHOLD = 0.7  # increase for more proposal (def=0.7)
    DETECTION_MIN_CONFIDENCE = 0.7  # decrease for more proposal (def=0.7)


config = ShapesConfig()

class ShapesDataset(utils.Dataset):
    def list_images(self,data_dir):
        # define classes
        self.add_class("cell", 1, "fibroblast")
        train_images = list(data_dir.glob('*.npy'))
        print('# image in this dataset : ',len(train_images))
        for idx,train_image in enumerate(train_images):
            label = str(train_image).replace("norm_level_01","level_01_lbl").replace(".npy","_lbl.pkl")
            self.add_image("cell",image_id=idx,path=train_image,labelpath=label,
                           height=config.IMAGE_SHAPE[0],width=config.IMAGE_SHAPE[1])

    def load_image(self, image_id):
        image = np.load(self.image_info[image_id]['path'])
        return image.astype(np.uint8)

    def load_mask(self, image_id):
        label = self.image_info[image_id]['labelpath']
        with open(label, 'rb') as f:
            annotation = pickle.load(f)
        masklayers=maskUtils.decode(annotation['annotation'])
        class_ids=annotation['class_ids']
        class_ids=np.array(class_ids).astype('int')
        return masklayers,class_ids

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cell":
            return info["truth"]
        else:
            super(self.__class__).image_reference(self, image_id)

data_dir = pathlib.Path(r'\\10.162.80.6\Kyu_Sync\temp\210115 simulated bead images for deep learning\Dataset\level_1\train\norm_level_01')
dataset_train = ShapesDataset()
dataset_train.list_images(data_dir)
dataset_train.prepare()

data_dir_val = pathlib.Path(r'\\10.162.80.6\Kyu_Sync\temp\210115 simulated bead images for deep learning\Dataset\level_1\val\val_norm_level_01')
dataset_val = ShapesDataset()
dataset_val.list_images(data_dir_val)
dataset_val.prepare()

image_ids = np.random.choice(dataset_train.image_ids, 2)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

image_ids = np.random.choice(dataset_val.image_ids, 2)
for image_id in image_ids:
    image = dataset_val.load_image(image_id)
    mask, class_ids = dataset_val.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_val.class_names)

augmentation = iaa.Sometimes(0.9, [
    # iaa.color.AddToHueAndSaturation((-3,3)),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
    iaa.Multiply((0.9, 1.1)),
    iaa.GaussianBlur(sigma=(0.0, 5.0)),
])


model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)
#
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=20,
            augmentation=augmentation,
            layers='heads')

print("Fine tune Resnet stage 4 and up")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=40,
            layers='4+',
            augmentation=augmentation)

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=80,
            augmentation=augmentation,
            layers="all")
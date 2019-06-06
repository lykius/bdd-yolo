"""
bdd/consts.py
Definition of constant values.
"""

IMG_ZIP_FILE = 'archives/bdd100k_images.zip'
LBL_ZIP_FILE = 'archives/bdd100k_labels_release.zip'
ROOT = 'bdd100k/'
JSON_PATH = ROOT + 'json/'
IMG_PATH = ROOT + 'images/'
LBL_PATH = IMG_PATH
TRAIN_INFO_PATH = JSON_PATH + 'train.json'
TRAIN_IMG_PATH = IMG_PATH + 'train'
TRAIN_LBL_PATH = LBL_PATH + 'train'
TRAIN_LIST_FILE = IMG_PATH + 'train.list'
VAL_INFO_PATH = JSON_PATH + 'val.json'
VAL_IMG_PATH = IMG_PATH + 'val'
VAL_LBL_PATH = LBL_PATH + 'val'
VAL_LIST_FILE = IMG_PATH + 'val.list'
TEST_IMG_PATH = IMG_PATH + 'test'
TEST_LIST_FILE = IMG_PATH + 'test.list'
IMG_ORIGINAL_SIZE = (1280, 720)
IMG_REDUCED_SIZE = (416, 256)
IMG_EXT = '.jpg'
CLASSES = ['bike', 'bus', 'car', 'motor', 'person', 'rider',
           'traffic light', 'traffic sign', 'train', 'truck']

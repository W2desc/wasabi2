"""Various constants."""
import numpy as np


NUM_CLASS = 19
PIXEL_BORDER = 1 # border pixels on which the segmentation is ko
SKY_LAB_ID = 10

# cityscapes labels and colors
LABEL_NUM = 19
label_name = [
    'road',
    'sidewalk',
    'building',
    'wall',
    'fence',
    'pole',
    'traffic light',
    'traffic sign',
    'vegetation',
    'terrain',
    'sky',
    'person',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle', 
    'mask', # added by Assia
]

# rgb
palette = [[128, 64, 128], 
        [244, 35, 232], 
        [70, 70, 70], 
        [102, 102, 156], 
        [190, 153, 153], 
        [153, 153, 153], 
        [250, 170, 30],
        [220, 220, 0], 
        [107, 142, 35], 
        [152, 251, 152], 
        [70, 130, 180], 
        [220, 20, 60], 
        [255, 0, 0], 
        [0, 0, 142], 
        [0, 0, 70],
        [0, 60, 100], 
        [0, 80, 100], 
        [0, 0, 230], 
        [119, 11, 32],
        [0, 0, 0]] # class 20: ignored, I added it, not cityscapes

palette_bgr = [ [l[2], l[1], l[0]] for l in palette]

LABEL_IGNORE = 19 # mask that I artificially introduce

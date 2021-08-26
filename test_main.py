import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf

from utilities import *

from bricks.CenterNet import CenterNet
from bricks.XCeptionImageNet import XCeptionImageNet
import os
# @title Choose the model to use, then evaluate the cell.
MODELS = {
    'centernet_with_keypoints': 'center_net_hg104_512x512_kpts_coco17_tpu-32', 
    'centernet_without_keypoints': 'center_net_hg104_512x512_coco17_tpu-8',
    'xception_imagenet': 'xception_imagenet'
}

model_display_name = 'xception_imagenet' # @param ['centernet_with_keypoints', 'centernet_without_keypoints']
model_name = MODELS[model_display_name]

# Download the checkpoint and put it into models/research/object_detection/test_data/

if model_display_name == 'centernet_with_keypoints':
    model_instance = CenterNet(
        pipeline_config = 'models/research/object_detection/configs/tf2/center_net_hourglass104_512x512_coco17_tpu-8.config', 
        model_dir = 'models/research/object_detection/test_data/checkpoint/'
    )
if model_display_name == 'centernet_without_keypoints':
    model_instance = CenterNet(
        pipeline_config = '../models/research/object_detection/configs/tf2/center_net_hourglass104_512x512_coco17_tpu-8.config', 
        model_dir = '../models/research/object_detection/test_data/checkpoint/'
    )
if model_display_name == 'xception_imagenet':
    model_instance = XCeptionImageNet()

image_path = '/mnt/c/Users/alessandro.colombo/Pictures/fanart.tv/backgrounds/a-fantastic-fear-of-everything___a-fantastic-fear-of-everything-5c6668a4e419e.jpg'
#image_np = load_image_into_numpy_array(image_path)

detections = model_instance.get_detections(image_path)




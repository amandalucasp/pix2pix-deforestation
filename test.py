import numpy as np

import yaml
import time
import cv2
import joblib
import shutil
import numpy as np
import os
from utils import *


def apply_mask(img, mask):
    # inverting reference
    current_def = (np.logical_not(mask.copy()))*1
    # number of channels
    channels = img.shape[-1]
    print(channels)
    current_def_matrix = np.repeat(np.expand_dims(current_def, axis = -1), channels, axis=-1)
    # multiplying the image with the current deforestation mask
    masked_img = img * current_def_matrix
    return masked_img

stream = open('./config.yaml')
config = yaml.load(stream, Loader=yaml.CLoader)

image_array = load_tif_image(config['root_path'] + 'img/2019_10m_b2348.tif').astype('float32')
final_mask = load_tif_image(config['root_path'] + 'ref/r10m_def_2019.tif').astype('float32')

image_array = image_array[:2000,:2000,[0, 1, 3]]
final_mask = final_mask[:2000,:2000]

print(np.unique(final_mask))

final_mask = np.rot90(final_mask)
#image_array, final_mask = get_dataset(config)

print('image_array.shape:', image_array.shape, np.unique(image_array))
print('final_mask.shape:', final_mask.shape, np.unique(final_mask))

masked_img = apply_mask(image_array, final_mask)

masked_img = normalization(masked_img.copy(), norm_type = 2)

cv2.imwrite('masked.png', masked_img)
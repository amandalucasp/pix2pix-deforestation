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
    # number of channels
    channels = img.shape[-1]
    print(channels)
    current_def_matrix = np.repeat(np.expand_dims(mask, axis = -1), channels, axis=-1)
    # multiplying the image with the current deforestation mask
    masked_img = img * current_def_matrix
    return masked_img

stream = open('./config.yaml')
config = yaml.load(stream, Loader=yaml.CLoader)

image_array = load_tif_image(config['root_path'] + 'img/2019_10m_b2348.tif').astype('float32')
final_mask = load_tif_image(config['root_path'] + 'ref/r10m_def_2019.tif').astype('float32')
final_mask = np.transpose(final_mask.copy(), (1, 0))

image_array = image_array[:5000,:5000,[0, 1, 3]]
final_mask = final_mask[:5000,:5000]
final_mask = (np.logical_not(final_mask.copy()))*1
print(np.unique(final_mask))

normalized_image = normalization(image_array.copy(), norm_type = 2)
cv2.imwrite('image_array.png', normalized_image*255)

cv2.imwrite('mask.png', final_mask*127.5)


#image_array, final_mask = get_dataset(config)

print('image_array.shape:', image_array.shape, np.unique(image_array))
print('final_mask.shape:', final_mask.shape, np.unique(final_mask))

masked_img = apply_mask(image_array, final_mask)

masked_img = normalization(masked_img.copy(), norm_type = 2)
print(np.unique(masked_img))

cv2.imwrite('masked.png', masked_img*255)
import yaml
import gdal
import skimage
import time
import glob
import cv2
import shutil
import imageio
import numpy as np
from PIL import Image
import sys, os, platform
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.util import view_as_windows
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import *

stream = open('./config.yaml')
config = yaml.load(stream, Loader=yaml.CLoader)
start = time.time()

config['output_path'] = config['output_path'] + '/change_detection_' + str(config['change_detection']).lower() +  '_two_classes_' + str(config['two_classes_problem']).lower()
print(config)

final_out_path = config['output_path'] + '/trained_pix2pix_input_mode_' + str(config['synthetic_input_mode']) + '/'
rej_out_path = config['output_path'] + '/rejected_patches_npy/'
os.makedirs(final_out_path, exist_ok=True)
shutil.copy('./config.yaml', final_out_path)

list_imgs = glob.glob(rej_out_path + '*_img.npy')

print('[*] Reading files...')
rej_pairs, rej_pairs_ref = load_npy_files(list_imgs)

print(np.unique(rej_pairs_ref[0]))

print('[*] Processing masks...')
final_pairs, final_pairs_ref = process_masks(rej_pairs, rej_pairs_ref, config)
print('[*] Saving pairs...')
save_image_pairs(final_pairs, final_pairs_ref, final_out_path, config, synthetic_input_pairs=True)
elapsed_time = time.time() - start
print('[*] Done. Elapsed time:', str(elapsed_time), 'seconds.')
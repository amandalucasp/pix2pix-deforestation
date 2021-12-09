import yaml
import gdal
import skimage
import time
import glob
import cv2
import joblib
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

np.random.seed(0)

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
print(np.min(rej_pairs), np.max(rej_pairs))
print(np.min(rej_pairs_ref), np.max(rej_pairs_ref))

print('[*] Processing masks...')
final_pairs, final_pairs_ref = process_masks(rej_pairs, rej_pairs_ref, config)

if len(final_pairs) > config['max_input_samples']:
    selected_pos = np.random.choice(len(final_pairs), config['max_input_samples'], replace=False)
    final_pairs = final_pairs[selected_pos]
    final_pairs_ref = final_pairs_ref[selected_pos]
    print('=> Will save only randomly selected ', str(config['max_input_samples']), 'pairs.')

print('[*] Normalizing pairs...')
# Normalization done with training patches stats
print('> Loading provided scaler:', config['scaler_path'])
preprocessing_scaler = joblib.load(config['scaler_path'])
final_pairs, _ = normalize_img_array(final_pairs, config['type_norm'], scaler=preprocessing_scaler)
print('> Checking normalization:')
print(np.min(final_pairs), np.max(final_pairs))
if config['type_norm'] == 3:
    final_pairs_ref = final_pairs_ref - 1
print('> Ref values:', np.unique(final_pairs_ref))

print('[*] Saving pairs...')
save_image_pairs(final_pairs, final_pairs_ref, final_out_path, config, synthetic_input_pairs=True)
elapsed_time = time.time() - start
print('[*] Done. Elapsed time:', str(elapsed_time), 'seconds.')
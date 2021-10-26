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

config['output_path'] = config['output_path'] + '/change_detection_' + str(config['change_detection']).lower() +  '_two_classes_' + str(config['two_classes_problem']).lower()
print(config)

final_out_path = config['output_path'] + '/trained_pix2pix_input'
rej_out_path = config['output_path'] + '/rejected_patches_npy/'

list_imgs = glob.glob(rej_out_path + '*_img.npy')
list_refs = glob.glob(rej_out_path + '*_ref.npy')

print('[*] Reading img files...')
imgs = []
for npy_file in list_imgs:
    imgs.append(np.load(npy_file))
rej_pairs = np.concatenate((imgs), axis=0)

print('[*] Reading ref files...')
refs = []
for npy_file in list_refs:
    refs.append(np.load(npy_file))
rej_pairs_ref = np.concatenate((refs), axis=0)

if config['save_all_rejected']:
    no_deforestation, new_deforest, old_deforest, only_deforest, all_classes, only_old_deforest = classify_masks(rej_pairs_ref)
    out_path = config['output_path'] + '/rejected_pairs/no_deforestation'
    save_image_pairs(rej_pairs[no_deforestation], rej_pairs_ref[no_deforestation], out_path, config, synthetic_input_pairs=True)
    out_path = config['output_path'] + '/rejected_pairs/new_deforest'
    save_image_pairs(rej_pairs[new_deforest], rej_pairs_ref[new_deforest], out_path, config, synthetic_input_pairs=True)
    out_path = config['output_path'] + '/rejected_pairs/old_deforest'
    save_image_pairs(rej_pairs[old_deforest], rej_pairs_ref[old_deforest], out_path, config, synthetic_input_pairs=True)
    out_path = config['output_path'] + '/rejected_pairs/only_deforest'
    save_image_pairs(rej_pairs[only_deforest], rej_pairs_ref[only_deforest], out_path, config, synthetic_input_pairs=True)
    out_path = config['output_path'] + '/rejected_pairs/all_classes'
    save_image_pairs(rej_pairs[all_classes], rej_pairs_ref[all_classes], out_path, config, synthetic_input_pairs=True)
    out_path = config['output_path'] + '/rejected_pairs/only_old_deforest'
    save_image_pairs(rej_pairs[only_old_deforest], rej_pairs_ref[only_old_deforest], out_path, config, synthetic_input_pairs=True)

print('[*] Processing masks...')
final_pairs, final_pairs_ref = process_masks(rej_pairs, rej_pairs_ref, config)
print('[*] Saving pairs...')
save_image_pairs(final_pairs, final_pairs_ref, final_out_path, config, synthetic_input_pairs=True)
print('[*] Done.')
import yaml
import gdal
import skimage
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
print(config)
stride = int(config['patch_size'] / 2)
tiles_ts = (list(set(np.arange(20)+1)-set(config['tiles_tr'])-set(config['tiles_val'])))

os.makedirs(config['output_path'], exist_ok=True)
shutil.copy('./config.yaml', config['output_path'])

image_stack, final_mask = get_dataset(config)

# Normalization
image_array = normalization(image_stack.copy(), config['type_norm'])
del image_stack

sample = image_array[:500,:500,:3]
sample_mask = final_mask[:500, :500]
cv2.imwrite('sample_image_array_cv2.png', sample)
imageio.imwrite('sample_image_array.png', sample)
imageio.imwrite('sample_mask.png', sample_mask)

# Print percentage of each class (whole image)
print('Total no-deforestaion class is {}'.format(len(final_mask[final_mask==0])))
print('Total deforestaion class is {}'.format(len(final_mask[final_mask==1])))
print('Total past deforestaion class is {}'.format(len(final_mask[final_mask==2])))
print('Percentage of deforestaion class is {:.2f}'.format((len(final_mask[final_mask==1])*100)/len(final_mask[final_mask==0])))

################### EXTRACT PATCHES

print("[*] EXTRACTING PATCHES")

trn_out_path = config['output_path'] + '/training_data'
val_out_path = config['output_path'] + '/validation_data'
tst_out_path = config['output_path'] + '/testing_data'

os.makedirs(trn_out_path + '/imgs', exist_ok=True)
os.makedirs(trn_out_path + '/masks', exist_ok=True)
os.makedirs(val_out_path + '/imgs', exist_ok=True)
os.makedirs(val_out_path + '/masks', exist_ok=True)
os.makedirs(tst_out_path + '/imgs', exist_ok=True)
os.makedirs(tst_out_path + '/masks', exist_ok=True)

# Create tile mask
mask_tiles = create_mask(final_mask.shape[0], final_mask.shape[1], grid_size=(5, 4))
image_array = image_array[:mask_tiles.shape[0], :mask_tiles.shape[1],:]
final_mask = final_mask[:mask_tiles.shape[0], :mask_tiles.shape[1]]

patches_trn, patches_trn_ref = patch_tiles(config['tiles_tr'], mask_tiles, image_array, final_mask, config['patch_size'], stride)
patches_val, patches_val_ref = patch_tiles(config['tiles_val'], mask_tiles, image_array, final_mask, config['patch_size'], stride)
patches_tst, patches_tst_ref = patch_tiles(tiles_ts, mask_tiles, image_array, final_mask, config['patch_size'], stride)
patches_trn, patches_trn_ref = discard_patches_by_percentage(patches_trn, patches_trn_ref, config)
patches_val, patches_val_ref = discard_patches_by_percentage(patches_val, patches_val_ref, config)
patches_tst, patches_tst_ref = discard_patches_by_percentage(patches_tst, patches_tst_ref, config)
del image_array, final_mask

print('[*] Training patches:', patches_trn.shape)
print('[*] Validation patches:', patches_val.shape)
print('[*] Testing patches:', patches_tst.shape)

if config['save_patches']:
    print("[*] SAVING PATCHES")
    print('Saving training patches...')
    write_patches_to_disk(patches_trn, patches_trn_ref, trn_out_path)
    print('Saving validation patches...')
    write_patches_to_disk(patches_val, patches_val_ref, val_out_path)
    print('Saving testing patches...')
    write_patches_to_disk(patches_tst, patches_tst_ref, tst_out_path)

################### COMBINE PATCHES INTO INPUT FORMAT FOR PIX2PIX

print("[*] SAVING IMAGE PAIRS")
print('Saving training pairs...')
save_image_pairs(patches_trn, patches_trn_ref, trn_out_path, config)
print('Saving validation pairs...')
save_image_pairs(patches_val, patches_val_ref, val_out_path, config)
print('Saving testing pairs...')
save_image_pairs(patches_tst, patches_tst_ref, tst_out_path, config)
del patches_tst, patches_tst_ref

################### EXTRACT MINIPATCHES (FOREST AND DEFORESTATION)

if config['extract_minipatches']:
    print("EXTRACTING MINIPATCHES")
    print('[*] Saving training minipatches.')
    save_minipatches(patches_trn, patches_trn_ref, trn_out_path, config)
    print('[*] Saving validation minipatches.')
    save_minipatches(patches_val, patches_val_ref, val_out_path, config)

print('[*] Preprocessing done.')

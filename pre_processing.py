import yaml
import gdal
import skimage
import scipy.misc
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
mini_stride = int(config['minipatch_size']/4)

################### READ TIF IMAGES
image_stack, final_mask = get_dataset(config)

# Normalization
type_norm = 1
image_array = normalization(image_stack.copy(), type_norm)
del image_stack

scipy.misc.imsave('image_array.png', image_array)
scipy.misc.imsave('final_mask.png', final_mask)

# Print percentage of each class (whole image)
print('Total no-deforestaion class is {}'.format(len(final_mask[final_mask==0])))
print('Total deforestaion class is {}'.format(len(final_mask[final_mask==1])))
print('Total past deforestaion class is {}'.format(len(final_mask[final_mask==1])))
print('Percentage of deforestaion class is {:.2f}'.format((len(final_mask[final_mask==1])*100)/len(final_mask[final_mask==0])))

################### EXTRACT PATCHES

print("EXTRACTING PATCHES")

trn_out_path = config['output_path'] + '/training_data'
val_out_path = config['output_path'] + '/validation_data'
tst_out_path = config['output_path'] + '/testing_data'

os.makedirs(config['output_path'], exist_ok=True)
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

# Define tiles for training, validation, and test sets
print('[*]Tiles for Training:', config['tiles_tr'])
print('[*]Tiles for Validation:', config['tiles_val'])
print('[*]Tiles for Testing:', tiles_ts)

mask_tr_val = np.zeros((mask_tiles.shape)).astype('float32')
# Training and validation mask
for tr_ in config['tiles_tr']:
    mask_tr_val[mask_tiles == tr_] = 1

for val_ in config['tiles_val']:
    mask_tr_val[mask_tiles == val_] = 2

mask_amazon_ts = np.zeros((mask_tiles.shape)).astype('float32')
for ts_ in tiles_ts:
    mask_amazon_ts[mask_tiles == ts_] = 1


# print('[*]Patch size:', config['patch_size'])
# print('[*]Stride:', stride)
patches_trn, patches_trn_ref = patch_tiles(config['tiles_tr'], mask_tiles, image_array, final_mask, config['patch_size'], stride)
patches_val, patches_val_ref = patch_tiles(config['tiles_val'], mask_tiles, image_array, final_mask, config['patch_size'], stride)
patches_tst, patches_tst_ref = patch_tiles(tiles_ts, mask_tiles, image_array, final_mask, config['patch_size'], stride)
patches_trn, patches_trn_ref = discard_patches_by_percentage(patches_trn, patches_trn_ref, config['patch_size'], config['min_percentage'])
patches_val, patches_val_ref = discard_patches_by_percentage(patches_val, patches_val_ref, config['patch_size'], config['min_percentage'])
patches_tst, patches_tst_ref = discard_patches_by_percentage(patches_tst, patches_tst_ref, config['patch_size'], config['min_percentage'])
del image_array, final_mask

print('Filtered the patches by a minimum of', str(config['min_percentage']), '% of deforestation.')
print('[*] Training patches:', patches_trn.shape)
print('[*] Validation patches:', patches_val.shape)
print('[*] Testing patches:', patches_tst.shape)

print('Saving training patches...')
write_patches_to_disk(patches_trn, patches_trn_ref, trn_out_path)
print('Saving validation patches...')
write_patches_to_disk(patches_val, patches_val_ref, val_out_path)
print('Saving testing patches...')
write_patches_to_disk(patches_tst, patches_tst_ref, tst_out_path)
del patches_tst, patches_tst_ref

################### EXTRACT MINIPATCHES (FOREST AND DEFORESTATION)

print("EXTRACTING MINIPATCHES")
max_minipatches = None

os.makedirs(trn_out_path + '/texture_class_0', exist_ok=True)
os.makedirs(trn_out_path + '/texture_class_1', exist_ok=True)
os.makedirs(val_out_path + '/texture_class_0', exist_ok=True)
os.makedirs(val_out_path + '/texture_class_1', exist_ok=True)
# os.makedirs(tst_out_path + '/texture_class_0', exist_ok=True)
# os.makedirs(tst_out_path + '/texture_class_1', exist_ok=True)

os.makedirs(trn_out_path + '/texture_class_0_debug', exist_ok=True)
os.makedirs(trn_out_path + '/texture_class_1_debug', exist_ok=True)
os.makedirs(val_out_path + '/texture_class_0_debug', exist_ok=True)
os.makedirs(val_out_path + '/texture_class_1_debug', exist_ok=True)
# os.makedirs(tst_out_path + '/texture_class_0_debug', exist_ok=True)
# os.makedirs(tst_out_path + '/texture_class_1_debug', exist_ok=True)

print('[*] Saving training minipatches.')
save_minipatches(patches_trn, patches_trn_ref, trn_out_path, mini_stride, config)
print('[*] Saving validation minipatches.')
save_minipatches(patches_val, patches_val_ref, val_out_path, mini_stride, config)

print('[*] Preprocessing done.')
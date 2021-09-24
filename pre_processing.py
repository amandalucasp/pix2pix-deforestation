import yaml
import gdal
import skimage
import time
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

stride = int((1 - config['overlap']) * config['patch_size'])
tiles_ts = (list(set(np.arange(20)+1)-set(config['tiles_tr'])-set(config['tiles_val'])))

config['output_path'] = config['output_path'] + '/change_detection_' + str(config['change_detection']).lower() +  '_two_classes_' + str(config['two_classes_problem']).lower()
# config['output_path'] = config['output_path'] + 'min_percentage' + str(config['min_percentage'])

print(config)
os.makedirs(config['output_path'], exist_ok=True)
shutil.copy('./config.yaml', config['output_path'])

image_stack, final_mask = get_dataset(config)

# Normalization
image_array = normalization(image_stack.copy(), config['type_norm'])
del image_stack

# sample = image_array[:500,:500,:3]
# sample_mask = final_mask[:500, :500]
# cv2.imwrite('sample_image_array_cv2.png', sample)
# imageio.imwrite('sample_image_array.png', sample)
# imageio.imwrite('sample_mask.png', sample_mask)

# Print percentage of each class (whole image)
print('Total no-deforestaion class is {}'.format(len(final_mask[final_mask==0])))
print('Total deforestaion class is {}'.format(len(final_mask[final_mask==1])))
print('Total past deforestaion class is {}'.format(len(final_mask[final_mask==2])))
print('Percentage of deforestaion class is {:.2f}'.format((len(final_mask[final_mask==1])*100)/len(final_mask[final_mask==0])))


################### EXTRACT TILES

trn_out_path = config['output_path'] + '/training_data'
val_out_path = config['output_path'] + '/validation_data'
tst_out_path = config['output_path'] + '/testing_data'

# Create tile mask
mask_tiles = create_mask(final_mask.shape[0], final_mask.shape[1], grid_size=(5, 4))
image_array = image_array[:mask_tiles.shape[0], :mask_tiles.shape[1],:]
final_mask = final_mask[:mask_tiles.shape[0], :mask_tiles.shape[1]]

if config['save_tiles']:
    os.makedirs(tst_out_path + '/tiles_ts', exist_ok=True)
    for num_tile in tiles_ts:
        rows, cols = np.where(mask_tiles == num_tile)
        x1 = np.min(rows)
        y1 = np.min(cols)
        x2 = np.max(rows)
        y2 = np.max(cols)
        tile_img = image_array[x1:x2 + 1, y1:y2 + 1, :]
        tile_ref = final_mask[x1:x2 + 1, y1:y2 + 1]
        np.save(tst_out_path + '/tiles_ts/' + str(num_tile) + '_img.npy', tile_ref)
        np.save(tst_out_path + '/tiles_ts/' + str(num_tile) + '_ref.npy', tile_ref)
        if config['change_detection']:
            h, w, c = tile_img.shape
            cv2.imwrite(tst_out_path + '/tiles_ts/' + str(num_tile) + '_t1.jpeg', tile_img[:,:,:c//2])
            cv2.imwrite(tst_out_path + '/tiles_ts/' + str(num_tile) + '_t2.jpeg', tile_img[:,:,c//2:])
        else:
            cv2.imwrite(tst_out_path + '/tiles_ts/' + str(num_tile) + '_img.jpeg', tile_img)

exit()

################### EXTRACT PATCHES


print("[*] EXTRACTING PATCHES")

patches_trn, patches_trn_ref = patch_tiles(config['tiles_tr'], mask_tiles, image_array, final_mask, config['patch_size'], stride)
patches_val, patches_val_ref = patch_tiles(config['tiles_val'], mask_tiles, image_array, final_mask, config['patch_size'], stride)
patches_tst, patches_tst_ref = patch_tiles(tiles_ts, mask_tiles, image_array, final_mask, config['patch_size'], stride)
del image_array, final_mask
patches_trn, patches_trn_ref, rej_patches_trn, rej_patches_trn_ref, rej_count_trn = discard_patches_by_percentage(patches_trn, patches_trn_ref, config)
patches_val, patches_val_ref, rej_patches_val, rej_patches_val_ref, rej_count_val = discard_patches_by_percentage(patches_val, patches_val_ref, config)
patches_tst, patches_tst_ref, rej_patches_tst, rej_patches_tst_ref, rej_count_tst = discard_patches_by_percentage(patches_tst, patches_tst_ref, config)

print('[*] Training patches:', patches_trn.shape)
print('[*] Validation patches:', patches_val.shape)
print('[*] Testing patches:', patches_tst.shape)

if config['save_patches']:
    os.makedirs(trn_out_path + '/imgs', exist_ok=True)
    os.makedirs(trn_out_path + '/masks', exist_ok=True)
    os.makedirs(val_out_path + '/imgs', exist_ok=True)
    os.makedirs(val_out_path + '/masks', exist_ok=True)
    os.makedirs(tst_out_path + '/imgs', exist_ok=True)
    os.makedirs(tst_out_path + '/masks', exist_ok=True)
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
del patches_trn, patches_trn_ref, patches_val, patches_val_ref, patches_tst, patches_tst_ref

################### CREATE INPUT FOR THE TRAINED PI2XPI2 GENERATOR

if config['create_input_pix2pix']:
    print('[*] CREATING INPUT FOR THE TRAINED MODEL')
    rej_out_path = config['output_path'] + '/trained_input'
    print('Concatenating pairs...')
    rej_pairs = np.concatenate((rej_patches_trn, rej_patches_val, rej_patches_tst), axis=0)
    rej_pairs_ref = np.concatenate((rej_patches_trn_ref, rej_patches_val_ref, rej_patches_tst_ref),axis=0)
    print('Processing masks...')
    rej_pairs_ref = process_masks(rej_pairs_ref, config)
    print('Saving pairs...')
    save_image_pairs(rej_pairs, rej_pairs_ref, rej_out_path, config, synthetic_input_pairs=True)

################### EXTRACT MINIPATCHES (FOREST AND DEFORESTATION)

if config['extract_minipatches']:
    print("EXTRACTING MINIPATCHES")
    print('[*] Saving training minipatches.')
    save_minipatches(patches_trn, patches_trn_ref, trn_out_path, config)
    print('[*] Saving validation minipatches.')
    save_minipatches(patches_val, patches_val_ref, val_out_path, config)

print('[*] Preprocessing done.')

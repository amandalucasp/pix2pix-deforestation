import yaml
import time
import glob
import joblib
import shutil
import numpy as np
import os
from utils import *

np.random.seed(0)

stream = open('./config.yaml')
config = yaml.load(stream, Loader=yaml.CLoader)
start = time.time()
stride = int((1 - config['overlap']) * config['patch_size'])
patch_size = config['patch_size']
config['output_path'] = config['output_path'] + '/change_detection_' + str(config['change_detection']).lower() 
print(config)
final_out_path = config['output_path'] + '/trained_pix2pix_input' + '/'
rej_out_path = config['output_path'] + '/rejected_patches_npy/'
os.makedirs(final_out_path, exist_ok=True)
shutil.copy('./config.yaml', final_out_path)

# Load numpy array of rejected patches per tile
list_imgs = glob.glob(rej_out_path + '*_img.npy')

selected_image_patches = []

for img_path in list_imgs:
    print('[*] Reading file...')
    rej_pairs, rej_pairs_ref = load_npy_file(img_path)
    print(np.min(rej_pairs), np.max(rej_pairs))
    print(np.min(rej_pairs_ref), np.max(rej_pairs_ref))
    print('[*] Filtering only NO new deforestation image patches')
    # Get only NO new deforestation image patches
    no_deforestation, new_deforest, old_deforest, only_deforest, all_classes, only_old_deforest = classify_masks(rej_pairs_ref)
    selected_image_patches.append(rej_pairs[no_deforestation])
    del rej_pairs_ref, rej_pairs

selected_image_patches = np.concatenate(selected_image_patches)

print('selected_image_patches:', selected_image_patches.shape)

# Create mask
print('[*] Creating mask')
mask_tiles = create_mask(config['lim_x'], config['lim_y'], grid_size=(5, 4))
tiles = list(set(np.arange(20)+1))

# Load reference masks from previous years
print('[*] Loading reference masks from previous years...')
# List of deforestation masks to load
years = ['2015','2016','2017','2018']

# List of reference masks
patches_ref = []

for year in years:
    print('year:', year)
    filename = config['root_path'] + 'ref/r10m_def_' + year + '.tif'
    print('filename:', filename)
    curr_year_mask = load_tif_image(filename).astype('int')
    # Transpose reference
    curr_year_mask = np.transpose(curr_year_mask.copy(), (1, 0))
    curr_year_mask = curr_year_mask[:config['lim_x'], :config['lim_y']]
    # Iterate over tiles
    for num_tile in tiles:
        print(num_tile)
        rows, cols = np.where(mask_tiles == num_tile)
        x1 = np.min(rows)
        y1 = np.min(cols)
        x2 = np.max(rows)
        y2 = np.max(cols)
        tile_ref = curr_year_mask[x1:x2 + 1, y1:y2 + 1]
        # Creating just to be able to use code
        tile_img = np.zeros(shape=(tile_ref.shape[0],tile_ref.shape[1],3))
        # Extract patches
        patches_img, patch_ref = extract_patches(tile_img, tile_ref, patch_size, stride)
        # Creating just to be able to use code
        patch_ref_acc = np.zeros_like(patch_ref)
        # Discard by % 
        _, patch_ref, _, _, _ = discard_patches_by_percentage(patches_img, patch_ref, patch_ref_acc, config)
        # Append to list
        patches_ref.append(patch_ref)

selected_ref_patches = np.concatenate(patches_ref)
np.random.shuffle(selected_ref_patches)

if len(selected_ref_patches) > len(selected_image_patches):
    selected_ref_patches = selected_ref_patches[:len(selected_image_patches)]

if len(selected_image_patches) > config['max_input_samples']:
    selected_pos = np.random.choice(len(selected_image_patches), config['max_input_samples'], replace=False)
    final_pairs = selected_image_patches[selected_pos]
    final_pairs_ref = selected_ref_patches[selected_pos]
else:
    final_pairs = selected_image_patches
    final_pairs_ref = selected_ref_patches

# Normalization done with training patches stats
print('[*] Normalizing pairs...')
print('> Loading provided scaler:', config['scaler_path'])
preprocessing_scaler = joblib.load(config['scaler_path'])
final_pairs, _ = normalize_img_array(final_pairs, config['type_norm'], scaler=preprocessing_scaler)
print('> Checking normalization:')
print(np.unique(final_pairs))
print(np.unique(final_pairs_ref))

print('[*] Saving pairs...')
save_image_pairs(final_pairs, final_pairs_ref, final_out_path, config, synthetic_input_pairs=True)
elapsed_time = time.time() - start
print('[*] Done. Elapsed time:', str(elapsed_time), 'seconds.')
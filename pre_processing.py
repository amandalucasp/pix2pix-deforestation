import yaml
import time
import cv2
import joblib
import shutil
import numpy as np
import os
from utils import *

np.random.seed(0)

start_time = time.time()

stream = open('./config.yaml')
config = yaml.load(stream, Loader=yaml.CLoader)

stride = int((1 - config['overlap']) * config['patch_size'])
tiles_ts = (list(set(np.arange(20)+1)-set(config['tiles_tr'])-set(config['tiles_val'])))

print(config)
os.makedirs(config['output_path'], exist_ok=True)
shutil.copy('./config.yaml', config['output_path'])

image_array, final_mask = get_dataset(config)

# Print percentage of each class (whole image)
print('Total NO deforestation class is {}'.format(len(final_mask[final_mask==0])))
print('Total NEW deforestation class is {}'.format(len(final_mask[final_mask==1])))
print('Total OLD deforestation class is {}'.format(len(final_mask[final_mask==2])))
print('Percentage of new deforestaion class is {:.2f}'.format((len(final_mask[final_mask==1])*100)/len(final_mask[final_mask==0])))

################### EXTRACT TILES

print("[*] EXTRACTING TILES")

trn_out_path = config['output_path'] + '/training_data'
val_out_path = config['output_path'] + '/validation_data'
tst_out_path = config['output_path'] + '/testing_data'

# Create tile mask
mask_tiles = create_mask(final_mask.shape[0], final_mask.shape[1], grid_size=(5, 4))
image_array = image_array[:mask_tiles.shape[0], :mask_tiles.shape[1],:]
final_mask = final_mask[:mask_tiles.shape[0], :mask_tiles.shape[1]]


################### EXTRACT PATCHES

print("[*] EXTRACTING PATCHES")

print('Extracting training patches')
patches_trn, patches_trn_ref = patch_tiles(config['tiles_tr'], mask_tiles, image_array, final_mask, stride, config, save_rejected=True)
if not config['load_scaler']:
    patches_trn, train_scaler = normalize_img_array(patches_trn, config['type_norm'])
    joblib.dump(train_scaler, config['output_path'] + '/minmax_scaler.bin', compress=True)
else:
    print('Loading provided scaler:', config['scaler_path'])
    train_scaler = joblib.load(config['scaler_path'])
    patches_trn, _ = normalize_img_array(patches_trn, config['type_norm'], scaler=train_scaler)

print('Extracting validation patches')
patches_val, patches_val_ref = patch_tiles(config['tiles_val'], mask_tiles, image_array, final_mask, stride, config)
patches_val, _ = normalize_img_array(patches_val, config['type_norm'], scaler=train_scaler)
print('Extracting test patches')
patches_tst, patches_tst_ref = patch_tiles(tiles_ts, mask_tiles, image_array, final_mask, stride, config)
patches_tst, _ = normalize_img_array(patches_tst, config['type_norm'], scaler=train_scaler)
del image_array, final_mask

print('Checking normalized patches:')
print(np.min(patches_trn), np.max(patches_trn))
print(np.min(patches_val), np.max(patches_val))
print(np.min(patches_tst), np.max(patches_tst))

print('Ref values:', np.unique(patches_trn_ref), np.unique(patches_val_ref), np.unique(patches_tst_ref))

print('Scaler params:')
print(train_scaler.min_)
print(train_scaler.scale_)
print(train_scaler.data_min_)
print(train_scaler.data_max_)
print(train_scaler.data_range_)

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

if config['save_image_pairs']:
    print("[*] SAVING IMAGE PAIRS")
    print('Saving training pairs...')
    save_image_pairs(patches_trn, patches_trn_ref, trn_out_path, config)
    print('Saving validation pairs...')
    save_image_pairs(patches_val, patches_val_ref, val_out_path, config)
    print('Saving testing pairs...')
    save_image_pairs(patches_tst, patches_tst_ref, tst_out_path, config)

elapsed_time = time.time() - start_time
print('[*] Preprocessing done. Elapsed time:', elapsed_time, 'seconds.')
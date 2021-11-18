import yaml
import numpy as np
import cv2
from utils import *

#input_folder = '/share_alpha_2/amandalucas/pix2pix/samples_patch_size_128/change_detection_true_two_classes_false/unet-results/output_2021-10-27_13-36-23_patchsize_128_batchsize_32_epochs_100_patience_10_baseline/'
#input_folder = '/share_alpha_2/amandalucas/pix2pix/samples_patch_size_128/change_detection_true_two_classes_false/unet-results/output_2021-10-28_12-16-29_patchsize_128_batchsize_32_epochs_100_patience_10'
input_folder = '/share_alpha_2/amandalucas/pix2pix/samples_patch_size_128/change_detection_true_two_classes_false/unet-results/output_2021-11-17_18-56-36_patchsize_128_batchsize_32_epochs_150_patience_20'

stream = open('./config.yaml')
config = yaml.load(stream, Loader=yaml.CLoader)
tiles_ts = (list(set(np.arange(20)+1)-set(config['tiles_tr'])-set(config['tiles_val'])))

image_stack, final_mask = get_dataset(config)
final_mask[final_mask == 2] = 0
mask_tiles = create_mask(final_mask.shape[0], final_mask.shape[1], grid_size=(5, 4))

print('loading mean_prob')
mean_prob = np.load(input_folder + '/prob_mean.npy')
mean_prob[mean_prob >= 0.5] = 1.0
mean_prob[mean_prob < 0.5] = 0.0

print('generating plots')
for num_tile in tiles_ts:
    print(num_tile)
    rows, cols = np.where(mask_tiles == num_tile)
    x1 = np.min(rows)
    y1 = np.min(cols)
    x2 = np.max(rows)
    y2 = np.max(cols)
    tile_ref = final_mask[x1:x2 + 1, y1:y2 + 1]
    tile_inference = mean_prob[x1:x2 + 1, y1:y2 + 1]
    cv2.imwrite(input_folder + '/tile_' + str(num_tile) + '_ref.png', tile_ref*255)
    cv2.imwrite(input_folder + '/tile_' + str(num_tile) + '_inference.png', tile_inference*255)

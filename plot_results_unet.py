import yaml
import numpy as np
import cv2
from utils import *

base_folder = '/share_alpha_2/amandalucas/pix2pix/samples_patch_size_128_all_bands_baseline_unet/change_detection_true_two_classes_false/unet-results/'
exps_list = ['output_2021-12-14_17-00-06_patchsize_128_batchsize_32_epochs_100_patience_10_augmented']


stream = open('./config.yaml')
config = yaml.load(stream, Loader=yaml.CLoader)
tiles_ts = (list(set(np.arange(20)+1)-set(config['tiles_tr'])-set(config['tiles_val'])))

image_stack, final_mask = get_dataset(config)
image_array = normalization(image_stack.copy(), 0) 
final_mask[final_mask == 2] = 0
mask_tiles = create_mask(final_mask.shape[0], final_mask.shape[1], grid_size=(5, 4))

def extract_patches(input_image, reference, patch_size, stride=0.5):
    stride = int((1 - stride) * patch_size)
    window_shape = patch_size
    window_shape_array = (window_shape, window_shape, input_image.shape[2])
    window_shape_ref = (window_shape, window_shape)
    patches_array = np.array(view_as_windows(input_image, window_shape_array, step = stride))
    patches_ref = np.array(view_as_windows(reference, window_shape_ref, step = stride))
    num_row,num_col,p,row,col,depth = patches_array.shape
    patches_array = patches_array.reshape(num_row*num_col,row,col,depth)
    patches_ref = patches_ref.reshape(num_row*num_col,row,col)
    return patches_array, patches_ref

for exp in exps_list:
    print('exp:', exp)
    input_folder = base_folder + exp
    print('loading mean_prob')
    mean_prob = np.load(input_folder + '/prob_mean.npy')
    #mean_prob[mean_prob >= 0.5] = 1.0
    #mean_prob[mean_prob < 0.5] = 0.0

    saving_folder = input_folder + '/patches_inference/'
    os.makedirs(saving_folder, exist_ok=True)
    os.makedirs(saving_folder + 't1', exist_ok=True)
    os.makedirs(saving_folder + 't2', exist_ok=True)

    print('generating plots')

    channels = [0, 1, 3]
    channels2 = [10, 11, 13]
    tiles_ts = [2]
    for num_tile in tiles_ts:
        print(num_tile)
        rows, cols = np.where(mask_tiles == num_tile)
        x1 = np.min(rows)
        y1 = np.min(cols)
        x2 = np.max(rows)
        y2 = np.max(cols)
        tile_img_t1 = image_array[x1:x2 + 1, y1:y2 + 1, channels]
        tile_img_t2 = image_array[x1:x2 + 1, y1:y2 + 1, channels2]
        tile_ref = final_mask[x1:x2 + 1, y1:y2 + 1]
        tile_inference = mean_prob[x1:x2 + 1, y1:y2 + 1]
        cv2.imwrite(input_folder + '/tile_' + str(num_tile) + '_t1.png', tile_img_t1)
        cv2.imwrite(input_folder + '/tile_' + str(num_tile) + '_t2.png', tile_img_t2)
        cv2.imwrite(input_folder + '/tile_' + str(num_tile) + '_ref.png', tile_ref*255)
        cv2.imwrite(input_folder + '/tile_' + str(num_tile) + '_inference.png', tile_inference*255)

        patches_t1, patches_inference = extract_patches(tile_img_t1, tile_inference, patch_size=256, stride=0.5)
        patches_t2, patches_ref = extract_patches(tile_img_t2, tile_ref, patch_size=256, stride=0.5)
    
        i = 0
        for patch1, patch2, patch_inf, patch_ref in zip(patches_t1, patches_t2, patches_inference, patches_ref):

            cv2.imwrite(saving_folder + '/t1/tile_' + str(num_tile) + '_' + str(i) + '_t1.png', patch1)
            cv2.imwrite(saving_folder + '/t2/tile_' + str(num_tile) + '_' + str(i) + '_t2.png', patch2)       

            fig = plt.figure()
            plt.imshow(patch_ref, cmap='jet')
            plt.axis('off')
            fig.savefig(saving_folder + '/tile_' + str(num_tile) + '_' + str(i) + '_ref.png')
            plt.close(fig)

            fig = plt.figure()
            plt.imshow(patch_inf, cmap='jet')
            plt.axis('off')
            fig.savefig(saving_folder + '/tile_' + str(num_tile) + '_' + str(i) + '_inference.png')
            plt.close(fig)
            i+=1



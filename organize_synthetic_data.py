import numpy as np
import yaml
import glob
import cv2
import os

np.random.seed(0)

from utils import *

stream = open('./config.yaml')
config = yaml.load(stream, Loader=yaml.FullLoader)

rej_data_path = config['data_path'] + config['synthetic_masks_path'] # path to trained pix2pix input
pix2pix_output_path = config['synthetic_data_path'] # path to pix2pix output

final_out_path = config['synthetic_data_path'] + '/synthetic_dataset/'
os.makedirs(final_out_path + 'imgs', exist_ok=True)
os.makedirs(final_out_path + 'masks', exist_ok=True)
os.makedirs(final_out_path + 'combined', exist_ok=True)

# files of t1 / real t2 / mask / masked t2
array_files = sorted(glob.glob(rej_data_path + 'pairs/*.npy')) #os.listdir(rej_data_path + 'pairs')

sample = np.load(array_files[0])
h, w, c = sample.shape
w = w // 4

for i in np.arange(len(array_files)):
    print(str(i), end='\r')
    current_filename = os.path.basename(array_files[i])
    print('current_filename:', current_filename)
    # get t1, mask
    current_file = rej_data_path + 'pairs/' + current_filename
    image = np.load(current_file) # t1 / real t2 / mask / masked t2
    t1 = image[:,:w, :]
    real_t2 = image[:,w:2*w, :]
    mask = image[:,2*w:3*w,:]
    # load corresponding fake_t2
    current_fake_t2 = pix2pix_output_path + '/fake_t2/' + current_filename
    fake_t2 = np.load(current_fake_t2)
    print(t1.shape, real_t2.shape, mask.shape, fake_t2.shape)
    # concatenate t1, fake_t2
    t1_fake_t2 = np.concatenate((t1, fake_t2), axis=-1)
    # save img, mask
    np.save(final_out_path + '/imgs/' + current_filename, t1_fake_t2)
    np.save(final_out_path + '/masks/' + current_filename, mask)
    # save JPEG for visualization
    if config['debug_mode']:
        # t1 / real t2 / mask / fake t2
        combined = np.zeros(shape=(h, w*4, c))
        combined[:,:w,:] = (t1 + 1) * 127.5
        combined[:,w:w*2,:] = (real_t2 + 1) * 127.5
        combined[:,w*2:w*3,:] = mask * 127.5
        combined[:,w*3:,:] = (fake_t2 + 1) * 127.5
        if len(config['channels']) > 3:
            combined = combined[:,:,config['debug_channels']]
        cv2.imwrite(final_out_path + '/combined/' + str(i) + '_debug.jpg', combined)
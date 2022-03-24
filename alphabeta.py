import numpy as np
import glob as glob

train_folder = ''

trn_npy_list = glob.glob("C:\\Users\\amandalucs\\Documents\\Github\\Samples\\change_detection_true\\training_data\\masks_acc\\*.npy")

print(len(trn_npy_list))

npy_list = []
for npy_file in trn_npy_list:
    npy_list.append(np.load(npy_file))

npy_arr = np.concatenate(npy_list, axis=-1)

total_zeros = np.count_nonzero(npy_arr == 0)
total_ones = np.count_nonzero(npy_arr == 1) 
total_twos = np.count_nonzero(npy_arr == 2) 

total_pixels = np.size(npy_list)
total_check = total_zeros + total_ones + total_twos

print('total_zeros:', total_zeros)
print('total_ones:', total_ones)
print('total_twos:', total_twos)
print('total_pixels:', total_pixels)
print('total_check:', total_check)
print('total/total_ones:', str(total_pixels/total_ones))
print('total/total_zeros:', str(total_pixels/total_zeros))

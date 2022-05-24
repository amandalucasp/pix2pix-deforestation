import numpy as np
import glob as glob

train_folder = ''

trn_npy_list = glob.glob("D:\\amandalucs\\Samples\\change_detection_true\\training_data\\masks\\*.npy")

print(len(trn_npy_list))

npy_list = []
for npy_file in trn_npy_list:
    npy_mask = np.load(npy_file) + 1
    npy_list.append(npy_mask)

npy_arr = np.array(npy_list)
print(np.unique(npy_arr))

total_zeros = np.sum(npy_arr == 0)
total_ones = np.sum(npy_arr == 1) 
total_twos = np.sum(npy_arr == 2) 

total_zeros_twos = total_zeros + total_twos

total_pixels = np.size(npy_arr)
total_check = total_zeros + total_ones + total_twos

print('total_zeros:', total_zeros)
print('total_ones:', total_ones)
print('total_twos:', total_twos)
print('total_pixels:', total_pixels)
print('total_check:', total_check)
print('total/total_zeros:', str(total_pixels/total_zeros))
print('total/total_ones:', str(total_pixels/total_ones)) # alpha
print('total/total_twos:', str(total_pixels/total_twos))
print('total/total_zeros_twos:', str(total_pixels/total_zeros_twos)) # beta
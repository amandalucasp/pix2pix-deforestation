from pathlib import Path
import glob
import os

'''
forest_files_path = r'D:\amandalucs\Samples\change_detection_true\Floresta'

forest_files = os.listdir(forest_files_path)
print(len(forest_files))
print(forest_files[0])

forest_files_stem = []
for item in forest_files:
    forest_files_stem.append(item.replace('_debug.jpg',''))

with open('./forest_files.txt', mode='wt', encoding='utf-8') as myfile:
    myfile.write('\n'.join(forest_files_stem))
'''

excluded_imgs_file = r"C:\Users\amandalucs\Documents\Github\pix2pix-deforestation\excluded_imgs.txt"

with open(excluded_imgs_file) as file:
    excluded_files_list = [line.rstrip('\n') for line in file]
excluded_files_list = [int(x) for x in excluded_files_list]
print('excluded_files_list:', excluded_files_list[:10])

all_npy_files = glob.glob(r"D:\amandalucs\Samples\change_detection_true\trained_pix2pix_input\pairs\*.npy")

forest_files_stem = []
for item in all_npy_files:
    stem_npy = int(Path(item).stem)
    if stem_npy in excluded_files_list:
        continue
    else:
        forest_files_stem.append(str(stem_npy))

print(len(forest_files_stem))

with open('./forest_files_antonio.txt', mode='wt', encoding='utf-8') as myfile:
    myfile.write('\n'.join(forest_files_stem))
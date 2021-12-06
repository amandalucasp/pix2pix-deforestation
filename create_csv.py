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
import pandas as pd
import time


def get_data(image, ref, data):
    lista = []
    rows, cols, channels = image.shape
    for i in range(rows):
        print(i, end='\r')
        for j in range(cols):
            bands = image[i, j, :]
            assigned_class = ref[i, j]
            lista.append([data, i, j, bands, assigned_class])
    return lista


def get_data_v2(image_t1, image_t2, ref):
    lista = []
    rows, cols, channels = image_t1.shape
    rows2, cols2, channels2 = image_t2.shape
    for i in range(rows):
        #print(i, end='\r')
        for j in range(cols):
            bands_t1 = image_t1[i, j, :]
            bands_t2 = image_t2[i, j, :]
            assigned_class = ref[i, j]
            lista.append([i, j, bands_t1, bands_t2, assigned_class])
    return lista


def patch_tiles(tiles, mask_amazon, image_2018, image_2019, image_ref):
    for num_tile in tiles:
        start_time = time.time()
        print(num_tile)
        rows, cols = np.where(mask_amazon == num_tile)
        x1 = np.min(rows)
        y1 = np.min(cols)
        x2 = np.max(rows)
        y2 = np.max(cols)
        tile_t1 = image_2018[x1:x2 + 1, y1:y2 + 1, :]
        tile_t2 = image_2019[x1:x2 + 1, y1:y2 + 1, :]
        tile_ref = image_ref[x1:x2 + 1, y1:y2 + 1]
        data_tile = get_data_v2(tile_t1, tile_t2, tile_ref)
        df = pd.DataFrame(data=data_tile, columns=['i', 'j', 'T1', 'T2', 'Classe'])
        key_tile = 'df_' + str(num_tile)
        df.to_hdf(config['root_path'] + '2018_2019.h5', key=key_tile, index=False)
        elapsed_time = time.time() - start_time
        print('Elapsed time:', elapsed_time)
        del df


stream = open('./config.yaml')
config = yaml.load(stream, Loader=yaml.CLoader)
config['channels'] = np.arange(10)
print('channels:', config['channels'])
config['lim_x'] = 250
config['lim_y'] = 250
image_stack, final_mask = get_dataset(config, full_image=True, do_filter_outliers=False)

mask_tiles = create_mask(final_mask.shape[0], final_mask.shape[1], grid_size=(5, 4))
image_2018, image_2019 = np.split(image_stack, 2, axis = -1)
del image_stack
rows, cols, channels = image_2018.shape
print('[2018] rows, cols, channels:', rows, cols, channels)
rows, cols, channels = image_2019.shape
print('[2019] rows, cols, channels:', rows, cols, channels)

tiles = np.arange(1,21)
patch_tiles(tiles, mask_tiles, image_2018, image_2019, final_mask)
exit()

print('Processing data')
lista_geral = get_data_v2(image_2018, image_2019, final_mask)
print('Creating dataframe')
df = pd.DataFrame(data=lista_geral, columns=['i', 'j', 'T1', 'T2', 'Classe'])
print('Saving to disk')
#df.to_csv(config['root_path'] + '2018_2019_Sentinel2.csv', index=False)
start_time = time.time()
df.to_hdf(config['root_path'] + '2018_2019_Sentinel2.h5', key='df', index=False)
elapsed_time = time.time() - start_time
print('Elapsed time:', elapsed_time)
exit()


print('Processing T1')
lista_2018 = get_data(image_2018, final_mask, 'T1')
del image_2018
print('> len(lista_2018):', len(lista_2018))
print('> creating dataframe')
df = pd.DataFrame(data=lista_2018, columns=['Data', 'i', 'j', 'Bandas', 'Classe'])
print('> saving to disk')
df.to_csv(config['root_path'] + '2018_Sentinel2.csv', index=False)
del df

print('Processing T2')
lista_2019 = get_data(image_2019, final_mask, 'T2')
del image_2019
print('> len(lista_2019):', len(lista_2019))
print('> creating dataframe')
df = pd.DataFrame(data=lista_2019, columns=['Data', 'i', 'j', 'Bandas', 'Classe'])
print('> saving to disk')
df.to_csv(config['root_path'] + '2019_Sentinel2.csv', index=False)

print('Concatenating lists')
lista_geral = lista_2018 + lista_2019
del lista_2018, lista_2019
print('> len(lista_geral):', len(lista_geral))
print('> creating dataframe')
df = pd.DataFrame(data=lista_geral, columns=['Data', 'i', 'j', 'Bandas', 'Classe'])
print(df.head())
print('> saving to disk')
df.to_csv(config['root_path'] + '2018_2019_Sentinel2.csv', index=False)



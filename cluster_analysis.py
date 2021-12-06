import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from utils import *
import sklearn
import yaml
import time
import cv2


def get_data_v2(image_t1, image_t2, ref):
    lista = []
    rows, cols, channels = image_t1.shape
    for i in range(rows):
        for j in range(cols):
            bands_t1 = image_t1[i, j, :]
            bands_t2 = image_t2[i, j, :]
            assigned_class = ref[i, j]
            lista.append([i, j, bands_t1, bands_t2, assigned_class])
    return lista


def create_df_from_tiles(tiles, mask_amazon, image_2018, image_2019, image_ref):
    df_list = []
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
        df_list.append(df)
        key_tile = 'df_' + str(num_tile)
        df.to_hdf(config['root_path'] + '2018_2019.h5', key=key_tile, index=False)
        elapsed_time = time.time() - start_time
        print('Elapsed time:', elapsed_time)
        del df
    return pd.concat(df_list)


#### LOADING DATASET AND DATAFRAME MANIPULATION
# Loads images from T1, T2; filters them for outlier removal; creates dataframe with the following columns:
# ['i', 'j', 'T1', 'T2', 'Classe']

stream = open('./config.yaml')
config = yaml.load(stream, Loader=yaml.CLoader)
config['channels'] = np.arange(10)
create_df = False
tiles = [1] #np.arange(1,21)

if create_df == True:
    image_stack, final_mask = get_dataset(config, full_image=True, do_filter_outliers=True)
    mask_tiles = create_mask(final_mask.shape[0], final_mask.shape[1], grid_size=(10, 8))
    image_2018, image_2019 = np.split(image_stack, 2, axis = -1)
    del image_stack
    rows, cols, channels = image_2018.shape
    print('[2018] rows, cols, channels:', rows, cols, channels)
    rows, cols, channels = image_2019.shape
    print('[2019] rows, cols, channels:', rows, cols, channels)
    df = create_df_from_tiles(tiles, mask_tiles, image_2018, image_2019, final_mask)
else:
    print('Loading dataframe')
    df = pd.read_hdf(config['root_path'] + '2018_2019.h5', key='df_1')

print(df.head())

# Splitting bands
df_columns = [format(x, '02d') for x in config['channels']]
df_columns_t1 = ['T1_' + s for s in df_columns]
df_columns_t2 = ['T2_' + s for s in df_columns]
df[df_columns_t1] = pd.DataFrame(df.T1.tolist(), index=df.index)
df[df_columns_t2] = pd.DataFrame(df.T2.tolist(), index=df.index)

# Creating column Diff
df['Dif'] = df['T2'] - df['T1']
# Splitting Diff bands
df_columns_dif = ['Dif_' + s for s in df_columns]
df[df_columns_dif] = pd.DataFrame(df.Dif.tolist(), index=df.index)
df = df.drop(columns=['T1','T2','Dif'])
print(df.head())
print(df.columns)

#### SPLIT DATASET
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)

print('> DATAFRAME DE TREINO:', len(df_train.index))
print(df_train.Classe.value_counts())
print('> DATAFRAME DE TESTE:', len(df_test.index))
print(df_test.Classe.value_counts())

y_train = df_train['Classe']
x_train = df_train.drop(columns='Classe')
y_test = df_test['Classe']
x_test = df_test.drop(columns='Classe')

#### NORMALIZATION

scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(x_train)
scaled_x_train = scaler.transform(x_train)
scaled_x_test = scaler.transform(x_test)

#### FACTOR ANALYSIS
from sklearn.decomposition import FactorAnalysis

fa = FactorAnalysis(svd_method='lapack', random_state=42)
transformer = fa.fit(X=scaled_x_train, y=y_train)
cov = transformer.get_covariance()
print(transformer.get_params())

fig, ax = plt.subplots(figsize=(20,20))
ax = sns.heatmap(cov, 
    vmin=-1, vmax=1, center=0,
    #cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    annot=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
fig.savefig('covariance.png')
plt.close(fig)

w, v = np.linalg.eig(cov)
# column ``v[:,i]`` is the eigenvector corresponding to the eigenvalue ``w[i]``.
print('> Eigenvalues:', w)
print('> Normalized eigenvectors:', v) 

transf_train = transformer.transform(scaled_x_train)
transf_test = transformer.transform(scaled_x_test)

#### CLUSTERING



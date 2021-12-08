from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from utils import *
import sklearn
import yaml
import time
import cv2
import os


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


def get_df(config, cluster_config):

    create_df = cluster_config['create_df']
    include_dif = cluster_config['include_dif']
    use_only_dif = cluster_config['use_only_dif']

    if create_df == True:
        #### LOADING DATASET AND DATAFRAME MANIPULATION
        # Loads images from T1, T2; filters them for outlier removal; creates dataframe with the following columns:
        # ['i', 'j', 'T1', 'T2', 'Classe']
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

    if use_only_dif:
        df['Dif'] = df['T2'] - df['T1']
        df_columns_dif = ['Dif_' + s for s in df_columns]
        df[df_columns_dif] = pd.DataFrame(df.Dif.tolist(), index=df.index)
        df = df.drop(columns=['Dif','T1','T2'])
        return df

    # Splitting bands
    df_columns = [format(x, '02d') for x in config['channels']]
    df_columns_t1 = ['T1_' + s for s in df_columns]
    df_columns_t2 = ['T2_' + s for s in df_columns]
    df[df_columns_t1] = pd.DataFrame(df.T1.tolist(), index=df.index)
    df[df_columns_t2] = pd.DataFrame(df.T2.tolist(), index=df.index)
    if include_dif:
        # Creating column Diff
        df['Dif'] = df['T2'] - df['T1']
        # Splitting Diff bands
        df_columns_dif = ['Dif_' + s for s in df_columns]
        df[df_columns_dif] = pd.DataFrame(df.Dif.tolist(), index=df.index)
        df = df.drop(columns=['Dif'])
        print(df.head())
    df = df.drop(columns=['T1','T2'])

    return df


def get_cols_names(n_factors):
    cols = []
    for i in range(1, n_factors + 1):
        cols.append('F' + str(i))
    return cols


stream = open('./config.yaml')
config = yaml.load(stream, Loader=yaml.CLoader)
stream = open('./cluster_config.yaml')
cluster_config = yaml.load(stream, Loader=yaml.CLoader)

config['channels'] = np.arange(10)
tiles = [1] #np.arange(1,21)
output_folder = cluster_config['output_folder']
os.makedirs(output_folder, exist_ok = True)
n_factors = cluster_config['n_factors']

df = get_df(config, cluster_config)

#### SPLIT DATASET
df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)
y_train = df_train['Classe']
x_train = df_train.drop(columns=['Classe','i','j'])
y_test = df_test['Classe']
x_test = df_test.drop(columns=['Classe','i','j'])

#### NORMALIZATION
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(x_train)
scaled_x_train = pd.DataFrame(scaler.transform(x_train), columns=x_train.columns)
scaled_x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)
print(scaled_x_train.columns)

#### FACTOR ANALYSIS
if n_factors is None:
    # EXPLORATORY FACTOR ANALYSIS (EFA)
    print('[*] Performing EFA')
    fa = FactorAnalyzer(n_factors=len(scaled_x_train.columns), rotation=None, method='ml')
    fa.fit(scaled_x_train)
    eigval, eigvec = fa.get_eigenvalues()
    print('Eigenvalues:', eigval)
    fig, ax = plt.subplots(1, 1, figsize=plt.figaspect(0.5))
    ax.scatter(range(1, scaled_x_train.shape[1] + 1), eigval)
    ax.plot(range(1, scaled_x_train.shape[1] + 1), eigval)
    plt.hlines(1, 0, scaled_x_train.shape[1], colors='r')
    plt.title('Scree Plot') # line plot of the eigenvalues of factors or principal components in an analysis
    ax.set_xlabel('Factors')
    ax.set_ylabel('Eigenvalues')
    plt.grid()
    plt.show()
    fig.savefig(output_folder + 'scree_plot.png')
    plt.close(fig)
    exit()
else:
    print('[*] Performing FA, n_factors:', n_factors)
    fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')#, method='ml') convergence problems
    fa.fit(scaled_x_train)
    cols = get_cols_names(n_factors)
    print('cols:', cols)
    # LOADINGS
    loads = fa.loadings_
    loads_df = pd.DataFrame(data=loads, columns=cols, index=scaled_x_train.columns)
    print('>> Loadings:', loads_df)
    np.save(output_folder + 'loadings_'+ str(n_factors) + '.npy', loads)
    fig, ax = plt.subplots(figsize=plt.figaspect(4.))
    ax = sns.heatmap(loads_df, vmin=-1, vmax=1, center=0, square=True, annot=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    fig.savefig(output_folder + 'loadings.png')
    plt.close(fig)
    # VARIANCE
    var = fa.get_factor_variance()
    np.save(output_folder + 'factor_var_'+ str(n_factors) + '.npy', var)
    var_df = pd.DataFrame(data=var, columns=cols, index=['Variance','Proportinal Var', 'Cumulative Var'])
    print('>> Variance:\n', var_df)
    fig, ax = plt.subplots()
    ax = sns.heatmap(var_df, vmin=0, vmax=10, center=0, square=True, annot=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    fig.savefig(output_folder + 'var.png')
    plt.close(fig)
    # COMMUNALITIES: % da variancia explicada pelos fatores
    comm = fa.get_communalities()
    comm_df = pd.DataFrame(comm, index=scaled_x_train.columns, columns=['Communalities'])
    comm_df['Variables'] = scaled_x_train.columns
    print('>> Communalities:\n', comm_df)
    np.save(output_folder + 'comm_'+ str(n_factors) + '.npy', comm)
    fig, ax = plt.subplots(1, 1, figsize=plt.figaspect(0.5))
    ax = sns.barplot(x='Variables', y = 'Communalities', data=comm_df)
    #ax.plot(range(1, scaled_x_train.shape[1] + 1), comm)
    plt.title('Communalities') # line plot of the eigenvalues of factors or principal components in an analysis
    ax.set_xlabel('Variables')
    ax.set_ylabel('Communality')
    plt.grid()
    plt.show()
    fig.savefig(output_folder + 'comm.png')
    plt.close(fig)
    # GET FACTOR SCORES FOR THE DATA
    features_train = fa.transform(scaled_x_train)
    features_test = fa.transform(scaled_x_test)

#### CLUSTERING
print('features_train:', features_train.shape)


#LinearDiscriminantAnalysis
#KMeans

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from factor_analyzer import FactorAnalyzer

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from utils import *
import sklearn
import imageio
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


def bench_k_means(kmeans, name, data, labels):
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
    """Benchmark to evaluate the KMeans initialization methods.

    Parameters
    ----------
    kmeans : KMeans instance
        A :class:`~sklearn.cluster.KMeans` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    t0 = time.time()
    estimator = kmeans.fit(data)
    fit_time = time.time() - t0
    results = [name, fit_time, estimator.inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        sklearn.metrics.homogeneity_score,
        sklearn.metrics.completeness_score,
        sklearn.metrics.v_measure_score,
        sklearn.metrics.adjusted_rand_score,
        sklearn.metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator.labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        sklearn.metrics.silhouette_score(
            data,
            estimator.labels_,
            metric="euclidean",
            sample_size=300,
        )
    ]

    # Show the results
    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print('\n')
    print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")
    print(formatter_result.format(*results))

    return estimator


stream = open('./config.yaml')
config = yaml.load(stream, Loader=yaml.CLoader)
stream = open('./cluster_config.yaml')
cluster_config = yaml.load(stream, Loader=yaml.CLoader)

config['channels'] = np.arange(10)
tiles = [1] #np.arange(1,21)
output_folder = cluster_config['output_folder']
os.makedirs(output_folder, exist_ok = True)
n_factors = cluster_config['n_factors']
n_classes = cluster_config['n_classes']

df = get_df(config, cluster_config)
print('>> Length of input dataframe:', len(df.index))

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

scaled_df = df.drop(columns=['Classe','i','j'])
scaled_df = pd.DataFrame(scaler.transform(scaled_df), columns=scaled_df.columns)
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
    features_complete = fa.transform(scaled_df)

#### CLUSTERING
print('features_train:', features_train.shape)

# KMeans
print(82 * "_")
kmeans = KMeans(init="random", n_clusters=n_classes, n_init=4, random_state=42)
estimator = bench_k_means(kmeans=kmeans, name="random", data=features_train, labels=y_train)
print(82 * "_")

# Visualization of clusters on Test Data
fig = plt.figure()
colors = ["#4EACC5", "#FF9C34", "#4E9A06"]
k_means_cluster_centers = estimator.cluster_centers_
k_means_labels = pairwise_distances_argmin(features_test, k_means_cluster_centers)

ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(n_classes), colors):
    my_members = k_means_labels == k
    ax.plot(features_test[my_members, 0], features_test[my_members, 1], "w", markerfacecolor=col, marker=".")
for k, col in zip(range(n_classes), colors):
    cluster_center = k_means_cluster_centers[k]
    ax.plot(
        cluster_center[0],
        cluster_center[1],
        "o",
        markerfacecolor=col,
        markeredgecolor="k",
        markersize=6,
    )
ax.set_title("KMeans")
ax.set_xticks(())
ax.set_yticks(())
plt.show()
fig.savefig(output_folder + 'kmeans.png')
plt.close(fig)

# Ground Truth
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(n_classes), colors):
    my_members = y_test == k
    ax.plot(features_test[my_members, 0], features_test[my_members, 1], "w", markerfacecolor=col, marker=".")
ax.set_title("Ground Truth")
ax.set_xticks(())
ax.set_yticks(())
plt.show()
fig.savefig(output_folder + 'ground_truth.png')
plt.close(fig)

# Confusion Matrix
cm = confusion_matrix(y_test, k_means_labels)
fig, ax = plt.subplots(figsize=plt.figaspect(.5))
ax = sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=[0, 1, 2],
            yticklabels=[0, 1, 2])
ax.set_xlabel('true label')
ax.set_ylabel('predicted label')
plt.show()
fig.savefig(output_folder + 'confusion_matrix.png')
plt.close(fig)

# Create Output Image
scaled_df['Classe'] = df['Classe']
scaled_df['i'] = df['i']
scaled_df['j'] = df['j']
print(scaled_df.head())

num_rows = scaled_df.i.max() - scaled_df.i.min() + 1
num_cols = scaled_df.j.max() - scaled_df.j.min() + 1
prediction_image = np.zeros(shape=(num_rows, num_cols, 1))
gt_image = np.np.zeros(shape=(num_rows, num_cols, 1))
k_means_labels = pairwise_distances_argmin(features_complete, k_means_cluster_centers)
scaled_df['k_means_labels'] = k_means_labels

print('Creating output image')
for (index, row) in scaled_df.iterrows():
    prediction_image[int(row['i']), int(row['j'])] = int(row['k_means_labels'])
    gt_image[int(row['i']), int(row['j'])] = int(row['Classe'])

prediction_image = cv2.cvtColor(prediction_image.astype('uint8'), cv2.COLOR_GRAY2RGB)
gt_image = cv2.cvtColor(gt_image.astype('uint8'), cv2.COLOR_GRAY2RGB)
imageio.imwrite(output_folder + 'prediction.png', prediction_image*127.5)
imageio.imwrite(output_folder + 'gt_image.png', gt_image*127.5)


#LinearDiscriminantAnalysis
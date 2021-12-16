from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.cluster import KMeans
from factor_analyzer import FactorAnalyzer

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sklearn
import imageio
import shutil
import yaml
import time
import cv2
import os

from utils import *


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
    df_list = pd.DataFrame(columns=['i', 'j', 'T1', 'T2', 'Classe', 'tile'])
    for num_tile in tiles:
        start_time = time.time()
        print(num_tile)
        rows, cols = np.where(mask_amazon == num_tile)
        print(rows, cols)
        x1 = np.min(rows)
        y1 = np.min(cols)
        x2 = np.max(rows)
        y2 = np.max(cols)
        tile_t1 = image_2018[x1:x2 + 1, y1:y2 + 1, :]
        tile_t2 = image_2019[x1:x2 + 1, y1:y2 + 1, :]
        tile_ref = image_ref[x1:x2 + 1, y1:y2 + 1]
        data_tile = get_data_v2(tile_t1, tile_t2, tile_ref)
        df = pd.DataFrame(data=data_tile, columns=['i', 'j', 'T1', 'T2', 'Classe'])
        df['tile'] = num_tile
        print(df.head())
        df_list = df_list.append(df)
        key_tile = 'df_' + str(num_tile)
        df.to_hdf(config['root_path'] + '2018_2019_all_20t.h5', key=key_tile, index=False)
        elapsed_time = time.time() - start_time
        print('Elapsed time:', elapsed_time)
        del df
    print(df_list.tile.value_counts())
    return df_list


def save_confmat(y_test, y_predicted, name='', labels=['0', '1']):

    report = classification_report(y_test, y_predicted, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(output_folder + name + '.csv')

    cm = confusion_matrix(y_test, y_predicted)
    fig, ax = plt.subplots(figsize=plt.figaspect(.5))
    ax = sns.heatmap(cm, square=True, annot=True, fmt='d', cbar=False, xticklabels=labels, yticklabels=labels)
    ax.set_ylabel('true label')
    ax.set_xlabel('predicted label')
    fig.savefig(output_folder + name + '_confusion_matrix.png')
    plt.close(fig)
    plt.show()
    cm = confusion_matrix(y_test, y_predicted, normalize='true')
    fig, ax = plt.subplots(figsize=plt.figaspect(.5))
    ax = sns.heatmap(cm, square=True, annot=True, fmt='f', cbar=False, xticklabels=labels, yticklabels=labels)
    ax.set_ylabel('true label')
    ax.set_xlabel('predicted label')
    fig.savefig(output_folder + name + '_confusion_matrix_norm.png')
    plt.close(fig)


def get_df(config, cluster_config, tiles, filename='2018_2019_all_20t.h5'):

    create_df = cluster_config['create_df']
    include_dif = cluster_config['include_dif']
    use_only_dif = cluster_config['use_only_dif']

    if create_df == True:
        #### LOADING DATASET AND DATAFRAME MANIPULATION
        # Loads images from T1, T2; filters them for outlier removal; creates dataframe with the following columns:
        # ['i', 'j', 'T1', 'T2', 'Classe']
        image_stack, final_mask = get_dataset(config, full_image=True, do_filter_outliers=True)
        if cluster_config['n_classes'] == 2:
            final_mask[final_mask == 2.] = 1
            print('Using two classes! np.unique(final_mask):', np.unique(final_mask))
        mask_tiles = create_mask(final_mask.shape[0], final_mask.shape[1], grid_size=(5, 4))
        image_2018, image_2019 = np.split(image_stack, 2, axis = -1)
        del image_stack
        rows, cols, channels = image_2018.shape
        print('[2018] rows, cols, channels:', rows, cols, channels)
        rows, cols, channels = image_2019.shape
        print('[2019] rows, cols, channels:', rows, cols, channels)
        df = create_df_from_tiles(tiles, mask_tiles, image_2018, image_2019, final_mask)
    else:
        print('Loading dataframe')
        df = pd.DataFrame()
        for tile in tiles:
            key_tile = 'df_' + str(tile)
            df_ = pd.read_hdf(config['root_path'] + filename, key=key_tile)
            df = df.append(df_)

    df_columns = [format(x, '02d') for x in config['channels']]
    if use_only_dif:
        df['Dif'] = df['T2'] - df['T1']
        df_columns_dif = ['Dif_' + s for s in df_columns]
        df[df_columns_dif] = pd.DataFrame(df.Dif.tolist(), index=df.index)
        df = df.drop(columns=['Dif','T1','T2'])
        return df

    # Splitting bands
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


start_time = time.time()
stream = open('./config.yaml')
config = yaml.load(stream, Loader=yaml.CLoader)
stream = open('./cluster_config.yaml')
cluster_config = yaml.load(stream, Loader=yaml.CLoader)
shutil.copy('./cluster_config.yaml', cluster_config['output_folder'])
config['channels'] = np.arange(10)
output_folder = cluster_config['output_folder'] + '_include_dif_' + str(cluster_config['include_dif']) + \
    '_use_only_dif_' + str(cluster_config['use_only_dif']) + '/'
os.makedirs(output_folder, exist_ok = True)
n_factors = cluster_config['n_factors']
n_classes = cluster_config['n_classes']

tiles = [1, 2]
df = get_df(config, cluster_config, tiles)

classes_df = df['Classe']
classes_df[classes_df == 2] = 0
df['Classe'] = classes_df

print('>> Length of input dataframe:', len(df.index))

print(df.Classe.value_counts())

#### SPLIT DATASET
df_train = df[df.tile == 1]
df_test = df[df.tile == 2]
print('df_train:')
print(df_train.head())
print('df_test:')
print(df_test.head())
#df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)
y_train = df_train['Classe']
x_train = df_train.drop(columns=['Classe','i','j','tile'])
y_test = df_test['Classe']
x_test = df_test.drop(columns=['Classe','i','j','tile'])

#### NORMALIZATION
print('Normalizing data')
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(x_train)
scaled_x_train = pd.DataFrame(scaler.transform(x_train), columns=x_train.columns)
scaled_x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

#scaled_df = df.drop(columns=['Classe','i','j','tile'])
#scaled_df = pd.DataFrame(scaler.transform(scaled_df), columns=scaled_df.columns)

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
    loads = np.square(fa.loadings_)
    loads_df = pd.DataFrame(data=loads, columns=cols, index=scaled_x_train.columns)
    print('>> Loadings:', loads_df)
    np.save(output_folder + 'loadings_'+ str(n_factors) + '.npy', loads)
    fig, ax = plt.subplots(figsize=plt.figaspect(4.))
    ax = sns.heatmap(loads_df, vmin=0, vmax=1, center=0, square=True, annot=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    fig.savefig(output_folder + 'loadings.png')
    plt.close(fig)
    # VARIANCE
    var = fa.get_factor_variance()
    np.save(output_folder + 'factor_var_'+ str(n_factors) + '.npy', var)
    var_df = pd.DataFrame(data=var, columns=cols, index=['Variance','Proportinal Var', 'Cumulative Var'])
    print('>> Variance:\n', var_df)
    fig, ax = plt.subplots()
    ax = sns.heatmap(var_df, vmin=0, vmax=10, center=0.5, square=True, annot=True)
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

# KMeans
print('>> Fitting K-Means - using factors')
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

save_confmat(y_test, k_means_labels,'kmeans')

# LinearDiscriminantAnalysis 
print('>> Fitting LDA - using all components')
clf = LinearDiscriminantAnalysis()
# original variables
clf.fit(scaled_x_train, y_train)
lda_labels_original_data = clf.predict(scaled_x_test)
save_confmat(df_test['Classe'], lda_labels_original_data,'lda_orig')
print('>> Fitting LDA - using factors')
# using factors
clf.fit(features_train, y_train)
lda_labels_factors = clf.predict(features_test)
save_confmat(df_test['Classe'], lda_labels_factors, 'lda_fa')

#### OUTPUT MAPS

df_test['k_means_labels'] = k_means_labels
df_test['lda_orig_labels'] = lda_labels_original_data
df_test['lda_fa_labels'] = lda_labels_factors

np.save(output_folder + 'k_means_labels.npy', k_means_labels)

# Create Output Image

print(df_test.head())

num_rows = df_test.i.max() - df_test.i.min() + 1
num_cols = df_test.j.max() - df_test.j.min() + 1
kmeans_image = np.zeros(shape=(num_rows, num_cols, 1))
gt_image = np.zeros(shape=(num_rows, num_cols, 1))
lda_image_orig = np.zeros(shape=(num_rows, num_cols, 1))
lda_image_fa = np.zeros(shape=(num_rows, num_cols, 1))

print('>> Creating output image')
for (index, row) in df_test.iterrows():
    kmeans_image[int(row['i']), int(row['j'])] = int(row['k_means_labels'])
    gt_image[int(row['i']), int(row['j'])] = int(row['Classe'])
    lda_image_orig[int(row['i']), int(row['j'])] = int(row['lda_orig_labels'])
    lda_image_fa[int(row['i']), int(row['j'])] = int(row['lda_fa_labels'])

print('>> Saving output image')
imgs = ['kmeans_image', 'gt_image', 'lda_image_orig', 'lda_image_fa']
i = 0
for image in [kmeans_image, gt_image, lda_image_orig, lda_image_fa]:
    fig = plt.figure(figsize=(15,12))
    plt.imshow(image*0.5, cmap='viridis')
    plt.show()
    plt.axis('off')
    fig.savefig(output_folder + imgs[i] + '.png')
    i+=1

np.save(output_folder + 'kmeans_image.npy', kmeans_image)

kmeans_image = cv2.cvtColor(kmeans_image.astype('uint8'), cv2.COLOR_GRAY2RGB)
gt_image = cv2.cvtColor(gt_image.astype('uint8'), cv2.COLOR_GRAY2RGB)
lda_image_orig = cv2.cvtColor(lda_image_orig.astype('uint8'), cv2.COLOR_GRAY2RGB)
lda_image_fa = cv2.cvtColor(lda_image_fa.astype('uint8'), cv2.COLOR_GRAY2RGB)

imageio.imwrite(output_folder + 'prediction_kmeans_full.png', kmeans_image*127.5)
imageio.imwrite(output_folder + 'gt_image_full.png', gt_image*127.5)
imageio.imwrite(output_folder + 'lda_image_orig_full.png', lda_image_orig*127.5)
imageio.imwrite(output_folder + 'lda_image_fa_full.png', lda_image_fa*127.5)

elapsed_time = time.time() - start_time
print('Elapsed time:', str(elapsed_time//60), 'min')
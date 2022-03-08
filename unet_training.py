from contextlib import redirect_stdout
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from osgeo import ogr, gdal
import matplotlib.colors
from PIL import Image
import numpy as np
import datetime
import pathlib
import shutil
import time
import yaml
import os


import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, average_precision_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model, load_model, Sequential
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *

import skimage
from skimage.util.shape import view_as_windows
from skimage.transform import resize
from skimage.morphology import disk
from skimage.filters import rank
import skimage.morphology 

from utils import *
from utils_unet import * 
import joblib

from tensorflow.keras.preprocessing.image import ImageDataGenerator

seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)


stream = open('./config.yaml')
config = yaml.load(stream, Loader=yaml.CLoader)

patch_size = config['patch_size']
channels = config['input_channels']
number_class = 2 
weights = [0.2, 0.8] 
epochs = config['epochs_unet']
stride = int((1 - config['overlap']) * config['patch_size'])
tiles_ts = (list(set(np.arange(20)+1)-set(config['tiles_tr'])-set(config['tiles_val'])))

root_path = config['unet_data_path']

training_dir = '/training_data' 
validation_dir = '/validation_data'
testing_dir = '/testing_data'
testing_tiles_dir = '/testing_data/tiles_ts/'

batch_size = config['batch_size_unet']
epochs = config['epochs_unet']
nb_filters = config['nb_filters']
number_runs = config['times']
patience_value = config['patience_value']
type_norm_unet = config['type_norm_unet']

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
output_folder = config['unet_data_path'] + config['training_output_path'] + 'output_' + st + '_patchsize_' + str(patch_size) + '_batchsize_' + str(batch_size) + '_epochs_' + str(epochs) + '_patience_' + str(patience_value)
if config['synthetic_data_path'] != '':
  output_folder = output_folder + '_augmented'
if config['augment_data']:
  output_folder = output_folder + '_classic_data_augmentation'
os.makedirs(output_folder, exist_ok = True)
os.makedirs(output_folder + '/checkpoints', exist_ok=True)
shutil.copy('./config.yaml', output_folder)


print('[*] Loading patches...')

# Training patches
print('[*] Loading training patches.')
patches_train, patches_tr_ref = load_patches(root_path, training_dir, augment_data=config['augment_data']) # retorna np.array(patches), np.array(patches_ref)
print('> train:', np.unique(patches_train),  np.unique(patches_tr_ref))

if config['synthetic_data_path'] != '':
  synt_data_path = config['synthetic_data_path']
  patches_train_synt, patches_tr_synt_ref = load_patches(config['synthetic_data_path'], '', from_pix2pix=True, pix2pix_max_samples=config['pix2pix_max_samples'])
  print('> pix2pix:', np.min(patches_train_synt), np.max(patches_train_synt))
  patches_train = np.concatenate((patches_train, patches_train_synt))
  patches_tr_ref = np.concatenate((patches_tr_ref, patches_tr_synt_ref))

print('>>patches_train:', np.unique(patches_train),  np.unique(patches_tr_ref))
# Validation patches
print('[*] Loading validation patches.')
patches_val, patches_val_ref = load_patches(root_path, validation_dir, augment_data=config['augment_data'])
print('> val:', np.unique(patches_val),  np.unique(patches_val_ref))

print('[*] Loading image array...')
image_array, final_mask, _ = get_dataset(config)
# normalize image to [-1, +1] with the same scaler used in preprocessing
print('> Loading provided scaler:', config['scaler_path'])
preprocessing_scaler = joblib.load(config['scaler_path'])
image_array, _ = normalize_img_array(image_array, config['type_norm'], scaler=preprocessing_scaler) # [-1, +1]
# u-net expects input to be  [0, 1]. [-1, +1] => [0, 1]:
image_array = image_array*0.5 + 0.5
print('> image_array:', np.min(image_array), np.max(image_array))

mask_tiles = create_mask(final_mask.shape[0], final_mask.shape[1], grid_size=(5, 4))
print('[*] Creating padded image...')
n_pool = 3
n_rows = 5
n_cols = 4
rows, cols = image_array.shape[:2]
pad_rows = rows - np.ceil(rows/(n_rows*2**n_pool))*n_rows*2**n_pool
pad_cols = cols - np.ceil(cols/(n_cols*2**n_pool))*n_cols*2**n_pool
npad = ((0, int(abs(pad_rows))), (0, int(abs(pad_cols))), (0, 0))
image1_pad = np.pad(image_array, pad_width=npad, mode='reflect')
h, w, c = image1_pad.shape
patch_size_rows = h//n_rows
patch_size_cols = w//n_cols
num_patches_x = int(h/patch_size_rows)
num_patches_y = int(w/patch_size_cols)
input_shape=(patch_size_rows,patch_size_cols, c)

print("[*] Patches for Training:", str(patches_train.shape), str(patches_tr_ref.shape))
print("[*] Patches for Validation:", str(patches_val.shape), str(patches_val_ref.shape))

patches_val_lb_h = tf.keras.utils.to_categorical(patches_val_ref, number_class)
patches_tr_lb_h = tf.keras.utils.to_categorical(patches_tr_ref, number_class)

if config['augment_data']:
  data_gen_args = dict(horizontal_flip = True, vertical_flip = True, featurewise_center=False)
  patches_train_datagen = ImageDataGenerator(data_gen_args)
  patches_train_ref_datagen = ImageDataGenerator(data_gen_args)
  patches_valid_datagen = ImageDataGenerator(data_gen_args)
  patches_valid_ref_datagen = ImageDataGenerator(data_gen_args)
  steps_per_epoch = len(patches_train)*3//config['batch_size_unet']
  validation_steps = len(patches_val)*3//config['batch_size_unet']
else:
  patches_train_datagen = ImageDataGenerator()
  patches_train_ref_datagen = ImageDataGenerator()
  patches_valid_datagen = ImageDataGenerator()
  patches_valid_ref_datagen = ImageDataGenerator()
  steps_per_epoch = len(patches_train)//config['batch_size_unet']
  validation_steps = len(patches_val)//config['batch_size_unet']

patches_train_generator = patches_train_datagen.flow(patches_train, batch_size=batch_size, shuffle=True, seed=seed)
patches_train_ref_generator = patches_train_datagen.flow(patches_tr_lb_h, batch_size=batch_size, shuffle=True, seed=seed)
train_generator = (pair for pair in zip(patches_train_generator, patches_train_ref_generator))

patches_valid_generator = patches_valid_datagen.flow(patches_val, batch_size=batch_size, shuffle=False, seed=seed)
patches_valid_ref_generator = patches_valid_ref_datagen.flow(patches_val_lb_h, batch_size=batch_size, shuffle=False, seed=seed)
valid_generator = (pair for pair in zip(patches_valid_generator, patches_valid_ref_generator))

adam = Adam(lr = config['lr_unet'] , beta_1=config['beta1_unet'])
loss = weighted_categorical_crossentropy(weights)
cmap = matplotlib.colors.ListedColormap(['black', 'white'])

time_tr = []
time_ts = []

for run in range(0, number_runs):
  print('[*] Start training', str(run))
  net = build_unet((patch_size, patch_size, channels), nb_filters, number_class)
  net.summary()
  net.compile(loss = loss, optimizer=adam , metrics=['accuracy'])

  start_training = time.time()
  earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=config['patience_value'], verbose=1, mode='min')
  # checkpoint = ModelCheckpoint(path_models+ '/' + method +'_'+str(run)+'.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
  lr_reduce = ReduceLROnPlateau(factor=0.9, min_delta=0.0001, patience=5, verbose=1)
  callbacks_list = [earlystop] #, checkpoint]

  print('Running fit.')
  history = net.fit_generator(train_generator,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    callbacks=callbacks_list,
    validation_data=valid_generator,
    validation_steps=validation_steps)
  end_training = time.time() - start_training
  time_tr.append(end_training)
  name = 'checkpoints/model_{}_epochs_{}_bsize_{}.tf'.format(run, epochs, batch_size)
  net.save(output_folder + '/' + name)

  fig = plt.figure(figsize=(8, 8))
  plt.subplot(2, 1, 1)
  plt.plot(history.history['accuracy'], label='Training Accuracy')
  plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.ylabel('Accuracy')
  plt.ylim([min(plt.ylim()),1])
  fig.savefig(output_folder + '/accuracy_model_' + str(run) + '.png')
  plt.close(fig)

  fig = plt.figure(figsize=(8, 8))
  plt.subplot(2, 1, 1)
  plt.plot(history.history['loss'], label='Training Loss')
  plt.plot(history.history['val_loss'], label='Validation Loss')
  plt.legend(loc='lower right')
  plt.ylabel('Loss')
  plt.ylim([min(plt.ylim()),1])
  fig.savefig(output_folder + '/loss_model_' + str(run) + '.png')
  plt.close(fig)

  # testing the model
  new_model = build_unet(input_shape, nb_filters, number_class)
  # model = load_model(name, compile=False)
  for l in range(1, len(net.layers)):
    new_model.layers[l].set_weights(net.layers[l].get_weights())
  print('Loaded weights for testing model')
  patch_t = []
  start_test = time.time()
  for i in range(0,num_patches_y):
      for j in range(0,num_patches_x):
          patch = image1_pad[patch_size_rows*j:patch_size_rows*(j+1), patch_size_cols*i:patch_size_cols*(i+1), :]
          predictions_ = new_model.predict(np.expand_dims(patch, axis=0)) # talvez era isso que faltava no outro
          # predicted_class = np.argmax(predictions_, axis=-1)
          # print(predictions_, predicted_class)
          del patch 
          # patch_t.append(predicted_class)
          patch_t.append(predictions_[:,:,:,1])
          del predictions_
  end_test =  time.time() - start_test
  patches_pred = np.asarray(patch_t).astype(np.float32)
  print(patches_pred.shape)
  prob_recontructed = pred_reconctruct(h, w, num_patches_x, num_patches_y, patch_size_rows, patch_size_cols, patches_pred)
  np.save(output_folder +'/prob_'+str(run)+'.npy',prob_recontructed) 
  time_ts.append(end_test)
  del prob_recontructed, net, patches_pred

time_tr_array = np.asarray(time_tr)
np.save(output_folder+'/metrics_tr.npy', time_tr_array)
time_ts_array = np.asarray(time_ts)
np.save(output_folder + '/metrics_ts.npy', time_ts_array)

prob_rec = np.zeros((image1_pad.shape[0],image1_pad.shape[1], number_runs))
for run in range (0, number_runs):
    prob_rec[:,:,run] = np.load(output_folder+'/'+'prob_'+str(run)+'.npy').astype(np.float32)

mean_prob = np.mean(prob_rec, axis = -1)
np.save(output_folder + '/prob_mean.npy', mean_prob)

# Plot mean map and reference
fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(121)
plt.title('Prediction')
ax1.imshow(mean_prob, cmap = cmap)
ax1.axis('off')

ref2 = final_mask.copy()
ref2 [final_mask == 2] = 0
ax2 = fig.add_subplot(122)
plt.title('Reference')
ax2.imshow(ref2, cmap = cmap)
ax2.axis('off')
fig.savefig(output_folder + '/mean_map_and_ref.png')

mean_prob = mean_prob[:final_mask.shape[0], :final_mask.shape[1]]

mask_amazon_ts = np.zeros((mask_tiles.shape)).astype('float32')
for ts_ in tiles_ts:
    mask_amazon_ts[mask_tiles == ts_] = 1

ref1 = np.ones_like(final_mask).astype(np.float32)
TileMask = mask_amazon_ts * ref1
GTTruePositives = final_mask==1

Npoints = 10
Pmax = np.max(mean_prob[GTTruePositives * TileMask ==1])
ProbList = np.linspace(Pmax,0,Npoints)

metrics_ = matrics_AA_recall(ProbList, mean_prob, final_mask, mask_amazon_ts, 625)
np.save(output_folder+'/acc_metrics.npy',metrics_)

# Complete NaN values
metrics_copy = metrics_.copy()
metrics_copy = complete_nan_values(metrics_copy)

# Comput Mean Average Precision (mAP) score 
Recall = metrics_copy[:,0]
Precision = metrics_copy[:,1]
AA = metrics_copy[:,2]
    
DeltaR = Recall[1:]-Recall[:-1]
AP = np.sum(Precision[:-1]*DeltaR)
print(output_folder)
print('mAP:', AP)

#X -> Recall
#Y -> Precision
mAP_func = Area_under_the_curve(Recall, Precision)
print('mAP_func:', mAP_func)

# Plot Recall vs. Precision curve
plt.close('all')
fig = plt.figure(figsize=(15,10))
plt.plot(metrics_copy[:,0],metrics_copy[:,1])
plt.plot(metrics_copy[:,0],metrics_copy[:,2])
plt.grid()
fig.savefig(output_folder + '/roc.png')
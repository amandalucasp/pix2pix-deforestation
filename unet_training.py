import matplotlib.pyplot as plt
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *

from utils_unet import * 

seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

stream = open('./config.yaml')
config = yaml.load(stream, Loader=yaml.CLoader)

patch_size = config['patch_size']
channels = config['input_channels']
number_class = 3 
weights = [0.2, 0.8, 0.0] 
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

st = datetime.datetime.fromtimestamp(time.time()).strftime("%Y_%m_%d_%H_%M_%S")
output_folder = config['training_output_path'] + st
if config['synthetic_data_path'] != '':
  output_folder = output_folder + '_augmented'
if config['augment_data']:
  output_folder = output_folder + '_classic_data_augmentation'
os.makedirs(output_folder, exist_ok = True)
path_models = output_folder + '/checkpoints'
os.makedirs(path_models, exist_ok=True)
shutil.copy('./config.yaml', output_folder)


print('[*] Loading patches...')

if 'real_max_samples' in config.keys():
  max_samples=config['real_max_samples']
else:
  max_samples=-1

# Training patches
print('[*] Loading training patches.')
patches_train, patches_tr_ref = load_patches(root_path, training_dir, max_samples=max_samples, augment_data=config['augment_data']) # retorna np.array(patches), np.array(patches_ref)
print('> Real data samples:', len(patches_train), np.min(patches_train), np.max(patches_train),  np.unique(patches_tr_ref))

# Validation patches
print('[*] Loading validation patches.')
patches_val, patches_val_ref = load_patches(root_path, validation_dir, augment_data=config['augment_data'])
print('> Real val Samples:', len(patches_val), np.min(patches_val), np.max(patches_val),  np.unique(patches_val_ref))

if config['synthetic_data_path'] != '':
  config['synthetic_masks_path'] = os.path.join(root_path, config['synthetic_masks_path'])
  patches_synt, patches_synt_ref = load_patches_synt(pix2pix_output_path=config['synthetic_data_path'], 
                                                              pix2pix_input_path=config['synthetic_masks_path'], 
                                                              pix2pix_max_samples=config['pix2pix_max_samples'], 
                                                              augment_data=config['augment_data'], 
                                                              selected_synt_file=config['selected_synt_file'], 
                                                              combine_t2=config['combine_t2'])
  print('> Synthetic data samples:', len(patches_synt), np.min(patches_synt), np.max(patches_synt),  np.unique(patches_synt_ref))
  #patches_train_synt, patches_tr_synt_ref = discard_patches_by_percentage(patches_train_synt, patches_tr_synt_ref, config)
  #print('> Synthetic data samples (after checking %):', len(patches_train_synt), np.min(patches_train_synt), np.max(patches_train_synt),  np.unique(patches_tr_synt_ref))

  patches_train_synt, patches_tr_synt_ref, patches_val_synt, patches_val_synt_ref = train_test_split(patches_synt, patches_synt_ref, test_size=0.2, random_state=seed)
  # alternative code for spliting:
  #train_size = int(0.8 * len(patches_synt))
  #patches_train_synt = patches_synt[:train_size]
  #patches_tr_synt_ref = patches_synt_ref[:train_size]
  #patches_val_synt = patches_synt[train_size:]
  #patches_val_synt_ref = patches_synt_ref[train_size:]

  patches_train = np.concatenate((patches_train, patches_train_synt))
  patches_tr_ref = np.concatenate((patches_tr_ref, patches_tr_synt_ref))
  patches_val = np.concatenate((patches_val, patches_val_synt))
  patches_val_ref = np.concatenate((patches_val_ref, patches_val_synt_ref))

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

time_tr = []

for run in range(0, number_runs):
  print('[*] Start training', str(run))
  net = build_unet((patch_size, patch_size, channels), nb_filters, number_class)
  net.summary()
  net.compile(loss = loss, optimizer=adam , metrics=['accuracy'])

  start_training = time.time()
  earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=config['patience_value'], verbose=1, mode='min')
  checkpoint = ModelCheckpoint(path_models + '/model_' + str(run) + '.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
  lr_reduce = ReduceLROnPlateau(factor=0.9, min_delta=0.0001, patience=5, verbose=1)
  callbacks_list = [earlystop, checkpoint]

  print('Running fit.')
  history = net.fit_generator(train_generator,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    callbacks=callbacks_list,
    validation_data=valid_generator,
    validation_steps=validation_steps)
  end_training = time.time() - start_training
  time_tr.append(end_training)

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

time_tr_array = np.asarray(time_tr)
np.save(output_folder+'/metrics_tr.npy', time_tr_array)

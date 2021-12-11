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
import joblib

from tensorflow.keras.preprocessing.image import ImageDataGenerator

np.random.seed(0)

stream = open('./config.yaml')
config = yaml.load(stream, Loader=yaml.CLoader)

patch_size = config['patch_size']
channels = config['input_channels']
number_class = 2 if config['two_classes_problem'] else 3
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
shutil.copy('./config.yaml', output_folder)

#exp_path = 'output_2021-11-30_01-43-34_patchsize_128_batchsize_32_epochs_100_patience_10'
#output_folder = config['unet_data_path'] + config['training_output_path'] + exp_path

# Function to compute mAP
def Area_under_the_curve(X, Y):
    #X -> Recall
    #Y -> Precision
    dx = np.diff(X)
    X_ = np.array([])
    Y_ = np.array([])
    
    eps = 5e-3
    for i in range(len(dx)):
        if dx[i] > eps:
            x0 = X[i]; x1 = X[i+1]
            y0 = Y[i]; y1 = Y[i+1]
            a = (y1 - y0) / (x1 - x0)
            b = y0 - a * x0
            x = np.arange(x0, x1, eps)
            y = a * x + b                
            X_ = np.concatenate((X_, x))
            Y_ = np.concatenate((Y_, y))
        else:
            X_ = np.concatenate((X_, X[i:i+1]))
            Y_ = np.concatenate((Y_, Y[i:i+1]))
    X_ = np.concatenate((X_, X[-1:]))
    Y_ = np.concatenate((Y_, Y[-1:]))
    new_dx = np.diff(X_)
    area = 100 * np.inner(Y_[:-1], new_dx)
    return area


def normalization_image(image, norm_type = 1):
    image_reshaped = image.reshape((image.shape[0]*image.shape[1]),image.shape[2])
    if (norm_type == 1):
        scaler = StandardScaler()
    if (norm_type == 2):
        scaler = MinMaxScaler(feature_range=(0,1))
    if (norm_type == 3):
        scaler = MinMaxScaler(feature_range=(-1,1))
    if (norm_type == 4):
        scaler = MinMaxScaler(feature_range=(0,2))
    #scaler = scaler.fit(image_reshaped)
    image_normalized = scaler.fit_transform(image_reshaped)
    image_normalized1 = image_normalized.reshape(image.shape[0],image.shape[1],image.shape[2])
    return image_normalized1


def test_model(model, patch_test):
  result = model.predict(patch_test)
  predicted_class = np.argmax(result, axis=-1)
  return predicted_class


def normalization_unet(image, norm_type = 1, scaler = None):
    if image.ndim == 4:
      image_reshaped = image.reshape((image.shape[0]*image.shape[1]*image.shape[2]),image.shape[3])
    if image.ndim == 3:
      image_reshaped = image.reshape((image.shape[0]*image.shape[1]),image.shape[2])

    if scaler != None:
      print('Fitting data to provided scaler...')
    else:
      print('No scaler was provided. Fitting scaler', str(norm_type), 'to data...')
      if (norm_type == 1):
          scaler = StandardScaler()
      if (norm_type == 2):
          scaler = MinMaxScaler(feature_range=(0,1))
      if (norm_type == 3):
          scaler = MinMaxScaler(feature_range=(-1,1))
      if (norm_type == 4):
          scaler = MinMaxScaler(feature_range=(0,2))
      if (norm_type == 5):
          scaler = MinMaxScaler(feature_range=(0,255))
      scaler = scaler.fit(image_reshaped)
    image_normalized = scaler.transform(image_reshaped)
    if image.ndim == 4:
      image_normalized1 = image_normalized.reshape(image.shape[0],image.shape[1],image.shape[2], image.shape[3])
    if image.ndim == 3:
      image_normalized1 = image_normalized.reshape(image.shape[0],image.shape[1],image.shape[2])
    return image_normalized1, scaler


def build_unet(input_shape, nb_filters, n_classes):
    input_layer = Input(input_shape)
 
    conv1 = Conv2D(nb_filters[0], (3 , 3) , activation='relu' , padding='same', name = 'conv1')(input_layer) 
    pool1 = MaxPooling2D((2 , 2))(conv1)
  
    conv2 = Conv2D(nb_filters[1], (3 , 3) , activation='relu' , padding='same', name = 'conv2')(pool1)
    pool2 = MaxPooling2D((2 , 2))(conv2)
  
    conv3 = Conv2D(nb_filters[2], (3 , 3) , activation='relu' , padding='same', name = 'conv3')(pool2)
    pool3 = MaxPooling2D((2 , 2))(conv3)
  
    conv4 = Conv2D(nb_filters[2], (3 , 3) , activation='relu' , padding='same', name = 'conv4')(pool3)
    
    conv5 = Conv2D(nb_filters[2], (3 , 3) , activation='relu' , padding='same', name = 'conv5')(conv4)
    
    conv6 = Conv2D(nb_filters[2], (3 , 3) , activation='relu' , padding='same', name = 'conv6')(conv5)
     
    tconv3 = Conv2D(nb_filters[2], (3 , 3), activation = 'relu', padding = 'same', name = 'upsampling3')(UpSampling2D(size = (2,2))(conv6))
    merged3 = concatenate([conv3, tconv3], name='concatenate3')
      
    tconv2 = Conv2D(nb_filters[1], (3 , 3), activation = 'relu', padding = 'same', name = 'upsampling2')(UpSampling2D(size = (2,2))(merged3))
    merged2 = concatenate([conv2, tconv2], name='concatenate2')
  
    tconv1 = Conv2D(nb_filters[0], (3 , 3), activation = 'relu', padding = 'same', name = 'upsampling1')(UpSampling2D(size = (2,2))(merged2))
    merged1 = concatenate([conv1, tconv1], name='concatenate1')
        
    output = Conv2D(n_classes,(1,1), activation = 'softmax')(merged1)
    
    return Model(input_layer , output)


def weighted_categorical_crossentropy(weights):
        """
        A weighted version of keras.objectives.categorical_crossentropy
        
        Variables:
            weights: numpy array of shape (C,) where C is the number of classes
        
        Usage:
            weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
            loss = weighted_categorical_crossentropy(weights)
            model.compile(loss=loss,optimizer='adam')
        """
        weights = K.variable(weights)
        def loss(y_true, y_pred):
            # scale predictions so that the class probas of each sample sum to 1
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            loss = y_true * K.log(y_pred) + (1-y_true) * K.log(1-y_pred)
            loss = loss * weights 
            loss = - K.mean(loss, -1)
            return loss
        return loss


def compute_metrics(true_labels, predicted_labels):
  accuracy = 100*accuracy_score(true_labels, predicted_labels)
  f1score = 100*f1_score(true_labels, predicted_labels, average=None)
  recall = 100*recall_score(true_labels, predicted_labels, average=None)
  precision = 100*precision_score(true_labels, predicted_labels, average=None)
  return accuracy, f1score, recall, precision


def data_augmentation(img, mask):
  image = np.rot90(img, 1)
  ref = np.rot90(mask, 1)
  return image, ref


def load_patches(root_path, folder, from_pix2pix=False, pix2pix_max_samples=1000, augment_data=False, number_class=3):
  imgs_dir = root_path + folder + '/imgs/'
  masks_dir = root_path + folder + '/masks/'
  img_files = os.listdir(imgs_dir)
  patches = []
  patches_ref = []
  selected_pos = np.arange(len(img_files))

  if from_pix2pix and len(img_files) > pix2pix_max_samples:
    print('[*]Loading input from pix2pix.', from_pix2pix)
    selected_pos = np.random.choice(len(img_files), pix2pix_max_samples, replace=False)
    print('Some of the randomly chosen samples:', selected_pos[0:20])

  for i in selected_pos:
    img_path = imgs_dir + img_files[i]
    mask_path = masks_dir + img_files[i]    
    img = np.load(img_path) 
    mask = np.load(mask_path)
    if augment_data and np.random.rand() > 0.5:
      img, mask = data_augmentation(img, mask)
    mask = tf.keras.utils.to_categorical(mask, number_class)
    patches.append(img)
    patches_ref.append(mask)
  
  patches, patches_ref = to_unet_format(np.array(patches), np.array(patches_ref))
  return patches, patches_ref


def to_unet_format(img, mask):
  # pix2pix generates a [-1, +1] output, but u-net expects [0, 1]
  # [-1, 1] => [0, 1]
  img = img + 1
  # u-net expects a 1-channel mask
  # [-1, 0, +1] => [0, 1, 2]
  mask = mask + 1 
  return img, mask


def load_tiles(root_path, testing_tiles_dir, tiles_ts):
  dir_ts = root_path + testing_tiles_dir
  img_list = []
  ref_list = []
  for num_tile in tiles_ts:
    img = np.load(dir_ts + str(num_tile) + '_img.npy')
    #img = normalization_unet(img, norm_type = type_norm_unet) # normaliza entre -1 e +1
    ref = np.load(dir_ts + str(num_tile) + '_ref.npy')
    print(img.shape, ref.shape)
    img_list.append(img)
    ref_list.append(ref)
  return np.array(img_list), np.array(ref_list)


def pred_reconctruct(h, w, num_patches_x, num_patches_y, patch_size_x, patch_size_y, patches_pred):
    count = 0
    img_reconstructed = np.zeros((h, w)).astype(np.float32)
    for i in range(0,num_patches_y):
        for j in range(0,num_patches_x):
            img_reconstructed[patch_size_x*j:patch_size_x*(j+1),patch_size_y*i:patch_size_y*(i+1)]=patches_pred[count]
            count+=1
    return img_reconstructed


def matrics_AA_recall(thresholds_, prob_map, ref_reconstructed, mask_amazon_ts_, px_area):
    thresholds = thresholds_    
    metrics_all = []
    for thr in thresholds:
        start_time = time.time()
        print('threshold:', thr)  

        img_reconstructed = np.zeros_like(prob_map).astype(np.int8)
        img_reconstructed[prob_map >= thr] = 1
    
        mask_areas_pred = np.ones_like(ref_reconstructed)
        area = skimage.morphology.area_opening(img_reconstructed, area_threshold = px_area, connectivity=1)
        area_no_consider = img_reconstructed-area
        mask_areas_pred[area_no_consider==1] = 0
        
        # Mask areas no considered reference
        mask_borders = np.ones_like(img_reconstructed)
        #ref_no_consid = np.zeros((ref_reconstructed.shape))
        mask_borders[ref_reconstructed==2] = 0
        #mask_borders[ref_reconstructed==-1] = 0
        
        mask_no_consider = mask_areas_pred * mask_borders 
        ref_consider = mask_no_consider * ref_reconstructed
        pred_consider = mask_no_consider*img_reconstructed
        
        ref_final = ref_consider[mask_amazon_ts_==1]
        pre_final = pred_consider[mask_amazon_ts_==1]
        
        # Metrics
        cm = confusion_matrix(ref_final, pre_final)
        #TN = cm[0,0]
        FN = cm[1,0]
        TP = cm[1,1]
        FP = cm[0,1]
        precision_ = TP/(TP+FP)
        recall_ = TP/(TP+FN)
        aa = (TP+FP)/len(ref_final)
        mm = np.hstack((recall_, precision_, aa))
        metrics_all.append(mm)
        elapsed_time = time.time() - start_time
        print('elapsed_time:', elapsed_time)
    metrics_ = np.asarray(metrics_all)
    return metrics_
 
    
def complete_nan_values(metrics):
    vec_prec = metrics[:,1]
    for j in reversed(range(len(vec_prec))):
        if np.isnan(vec_prec[j]):
            vec_prec[j] = 2*vec_prec[j+1]-vec_prec[j+2]
            if vec_prec[j] >= 1:
                vec_prec[j] == 1
    metrics[:,1] = vec_prec
    return metrics 


print('[*] Loading patches...')

# Training patches
print('[*] Loading training patches.')
patches_train, patches_tr_ref = load_patches(root_path, training_dir, augment_data=config['augment_data']) # retorna np.array(patches), np.array(patches_ref)
print('>train:', np.min(patches_train), np.max(patches_train))

if config['synthetic_data_path'] != '':
  synt_data_path = config['synthetic_data_path']
  patches_train_synt, patches_tr_synt_ref = load_patches(config['synthetic_data_path'], '', from_pix2pix=True, pix2pix_max_samples=config['pix2pix_max_samples'])
  print('>pix2pix:', np.min(patches_train_synt), np.max(patches_train_synt))
  patches_train = np.concatenate((patches_train, patches_train_synt))
  patches_tr_ref = np.concatenate((patches_tr_ref, patches_tr_synt_ref))

print('>>patches_train:', np.min(patches_train), np.max(patches_train))

# Validation patches
print('[*] Loading validation patches.')
patches_val, patches_val_ref = load_patches(root_path, validation_dir, augment_data=config['augment_data'])
print('>val:', np.min(patches_val), np.max(patches_val))

print('[*] Normalizing image array...')
image_array, final_mask = get_dataset(config)
# normalize image to [-1, +1] with the same scaler used in preprocessing
print('> Loading provided scaler:', config['scaler_path'])
preprocessing_scaler = joblib.load(config['scaler_path'])
image_array, _ = normalize_img_array(image_array, config['type_norm'], scaler=preprocessing_scaler) # [-1, +1]
# u-net expects input to be  [0, 1]. [-1, +1] => [0, 1]:
image_array = image_array + 1
print('>image_array:', np.min(image_array), np.max(image_array))

#image_stack_1, _ = normalization_unet(image_stack.copy(), norm_type = 5) 
#image_array, _ = normalization_unet(image_stack_1.copy(), norm_type = type_norm_unet, scaler = train_scaler)

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
print('h, w, c:', h, w, c)
patch_size_rows = h//n_rows
patch_size_cols = w//n_cols
num_patches_x = int(h/patch_size_rows)
num_patches_y = int(w/patch_size_cols)
input_shape=(patch_size_rows,patch_size_cols, c)

print("[*] Patches for Training:", str(patches_train.shape), str(patches_tr_ref.shape))
print("[*] Patches for Validation:", str(patches_val.shape), str(patches_val_ref.shape))

patches_val_lb_h = patches_val_ref 
patches_tr_lb_h = patches_tr_ref 

if config['augment_data']:
  data_gen_args = dict(horizontal_flip = True, vertical_flip = True)
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

seed = 1
patches_train_generator = patches_train_datagen.flow(patches_train, batch_size=batch_size, shuffle=True, seed=seed)
patches_train_ref_generator = patches_train_datagen.flow(patches_tr_lb_h, batch_size=batch_size, shuffle=True, seed=seed)
train_generator = (pair for pair in zip(patches_train_generator, patches_train_ref_generator))

patches_valid_generator = patches_valid_datagen.flow(patches_val, batch_size=batch_size, shuffle=False, seed=seed)
patches_valid_ref_generator = patches_valid_ref_datagen.flow(patches_val_lb_h, batch_size=batch_size, shuffle=False, seed=seed)
valid_generator = (pair for pair in zip(patches_valid_generator, patches_valid_ref_generator))

os.makedirs(output_folder + '/checkpoints', exist_ok=True)

adam = Adam(lr = config['lr_unet'] , beta_1=config['beta1_unet'])
# 0: forest, 1: new deforestation, 2: old deforestation
weights = [0.2, 0.8, 0.0] # desconsidero a classe 2 nesse problema
print("[*] Weights for CE:", weights)
loss = weighted_categorical_crossentropy(weights)

cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'white'])

time_tr = []
time_ts = []
exp = 1
method = 'unet'

path_exp = output_folder+'/experiments/exp'+str(exp)
path_models = path_exp+'/models'
path_maps = path_exp+'/pred_maps'

if not os.path.exists(path_exp):
    os.makedirs(path_exp)   
if not os.path.exists(path_models):
    os.makedirs(path_models)   
if not os.path.exists(path_maps):
    os.makedirs(path_maps)

epochs = config['epochs_unet']


for run in range(0, number_runs):
  print('[*] Start training', str(run))
  net = build_unet((patch_size, patch_size, channels), nb_filters, number_class)
  net.summary()
  net.compile(loss = loss, optimizer=adam , metrics=['accuracy'])

  start_training = time.time()
  earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=config['patience_value'], verbose=1, mode='min')
  checkpoint = ModelCheckpoint(path_models+ '/' + method +'_'+str(run)+'.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
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
  name = 'checkpoints/model_{}_epochs_{}_bsize_{}.h5'.format(run, epochs, batch_size)
  net.save(output_folder + '/' + name)


  fig = plt.figure(figsize=(8, 8))
  plt.subplot(2, 1, 1)
  plt.plot(history.history['acc'], label='Training Accuracy')
  plt.plot(history.history['val_acc'], label='Validation Accuracy')
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
ref1 [final_mask == 2] = 0
TileMask = mask_amazon_ts * ref1
GTTruePositives = final_mask==1

Npoints = 50
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
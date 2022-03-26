from contextlib import redirect_stdout
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from osgeo import ogr, gdal
import matplotlib 
matplotlib.use('Agg')
import matplotlib.colors
from PIL import Image
import numpy as np
import datetime
import pathlib
import shutil
import time
import yaml
import json
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

from tensorflow.keras.preprocessing.image import ImageDataGenerator


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


def parse_file_to_list(file_path):
  with open(file_path) as f:
    file_list = f.read().splitlines()
  file_list = [int(x) for x in file_list]
  return file_list


def data_augmentation(img, mask):
  image = np.rot90(img, 1)
  ref = np.rot90(mask, 1)
  return image, ref


def load_patches(root_path, folder, from_pix2pix=False, pix2pix_max_samples=10000, augment_data=False, selected_synt_file=None):
  imgs_dir = root_path + folder + '/imgs/'
  masks_dir = root_path + folder + '/masks/'
  img_files = os.listdir(imgs_dir)
  patches = []
  patches_ref = []
  selected_pos = np.arange(len(img_files))

  if from_pix2pix:
    print('[*] Loading input from pix2pix.')
    if selected_synt_file:
      selected_pos = parse_file_to_list(selected_synt_file)
    if len(selected_pos) > pix2pix_max_samples:
      selected_pos = np.random.choice(selected_pos, pix2pix_max_samples, replace=False)
    print('Some of the randomly chosen samples:', selected_pos[0:20])

  for i in selected_pos:
    img_path = imgs_dir + img_files[i]
    mask_path = masks_dir + img_files[i]    
    img = np.load(img_path) 
    mask = np.load(mask_path)
    img, mask = to_unet_format(img, mask)
    if augment_data and np.random.rand() > 0.5:
      img, mask = data_augmentation(img, mask)
    patches.append(img)
    patches_ref.append(mask)
  
  return np.array(patches), np.squeeze(np.array(patches_ref))


def to_unet_format(img, mask):
  # pix2pix generates a [-1, +1] output, but u-net expects [0, 1]
  # [-1, 1] => [0, 2]
  img = img*0.5 + 0.5
  # u-net expects a 1-channel mask
  # [-1, 0, +1] => [0, 1, 2]
  if mask.ndim > 2:
    mask = mask[:, :, -1]
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
    

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
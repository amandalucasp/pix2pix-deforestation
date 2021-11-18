from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
# from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Activation, Flatten, Input, concatenate, UpSampling2D, BatchNormalization
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

stream = open('./config.yaml')
config = yaml.load(stream, Loader=yaml.CLoader)

patch_size = config['patch_size']
channels = config['input_channels']
number_class = 2 if config['two_classes_problem'] else 3

stride = int((1 - config['overlap']) * config['patch_size'])
tiles_ts = (list(set(np.arange(20)+1)-set(config['tiles_tr'])-set(config['tiles_val'])))

root_path = config['unet_data_path']

training_dir = '/training_data' #/augmented_dataset' 
validation_dir = '/validation_data'
testing_dir = '/testing_data'
testing_tiles_dir = '/testing_data/tiles_ts/'

input_shape = [config['patch_size'], config['patch_size'], config['input_channels']]
batch_size = config['batch_size_unet']
epochs = config['epochs_unet']
nb_filters = config['nb_filters']
number_runs = config['times']
patience_value = config['patience_value']

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
output_folder = config['unet_data_path'] + config['training_output_path'] + 'output_' + st + '_patchsize_' + str(patch_size) + '_batchsize_' + str(batch_size) + '_epochs_' + str(epochs) + '_patience_' + str(patience_value)
if config['synthetic_data_path'] != '':
  output_folder = output_folder + '_augmented'
os.makedirs(output_folder, exist_ok = True)
shutil.copy('./config.yaml', output_folder)


def normalization(image, norm_type = 1):
    image_reshaped = image.reshape((image.shape[0]*image.shape[1]),image.shape[2])
    if (norm_type == 1):
        scaler = StandardScaler()
    if (norm_type == 2):
        scaler = MinMaxScaler(feature_range=(0,1))
    if (norm_type == 3):
        scaler = MinMaxScaler(feature_range=(-1,1))
    if (norm_type == 4):
        scaler = MinMaxScaler(feature_range=(0,2))
    scaler = scaler.fit(image_reshaped)
    image_normalized = scaler.fit_transform(image_reshaped)
    image_normalized1 = image_normalized.reshape(image.shape[0],image.shape[1],image.shape[2])
    return image_normalized1


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


def train_model(net, patches_train, patches_tr_lb_h, patches_val, patches_val_lb_h, batch_size, epochs, patience_value):
  print('Start training.. ')

  waiting_time = 0

  validation_accuracy = []
  training_accuracy = []

  for epoch in range(epochs):
    loss_tr = np.zeros((1 , 2))
    loss_val = np.zeros((1 , 2))
    # Computing the number of batchs
    n_batchs_tr = patches_train.shape[0]//batch_size
    # Random shuffle the data
    patches_train , patches_tr_lb_h = shuffle(patches_train , patches_tr_lb_h , random_state = 0)
        
    # Training the network per batch
    for  batch in range(n_batchs_tr):
      x_train_b = patches_train[batch * batch_size : (batch + 1) * batch_size , : , : , :]
      y_train_h_b = patches_tr_lb_h[batch * batch_size : (batch + 1) * batch_size , :, :, :]
      loss_tr = loss_tr + net.train_on_batch(x_train_b , y_train_h_b)
    
    # Training loss
    loss_tr = loss_tr/n_batchs_tr
    print("%d [Training loss: %f , Train acc.: %.2f%%]" %(epoch , loss_tr[0 , 0], 100*loss_tr[0 , 1]))
    training_accuracy.append(loss_tr[0 , 1])

    # Computing the number of batchs
    n_batchs_val = patches_val.shape[0]//batch_size

    # Evaluating the model in the validation set
    for  batch in range(n_batchs_val):
      x_val_b = patches_val[batch * batch_size : (batch + 1) * batch_size , : , : , :]
      y_val_h_b = patches_val_lb_h[batch * batch_size : (batch + 1) * batch_size , :, :, :]
      loss_val = loss_val + net.test_on_batch(x_val_b , y_val_h_b)
    
    # validation loss
    loss_val = loss_val/n_batchs_val
    print("%d [Validation loss: %f , Validation acc.: %.2f%%]" %(epoch , loss_val[0 , 0], 100*loss_val[0 , 1]))
    validation_accuracy.append(loss_val[0 , 1])

    if epoch == 0:
      best_loss_val = loss_val[0 , 0]

    print("=> Best_loss_val [ES Criteria]:", str(best_loss_val), "Patience:", str(waiting_time))

    # early stopping
    if loss_val[0 , 0] < best_loss_val:
      # reset waiting time
      waiting_time = 0
      # update best_loss_val
      best_loss_val = loss_val[0 , 0]
    else:
      # increment waiting time
      waiting_time +=1
    
    # if patience value is reached
    if waiting_time == patience_value:
      # stop training
      return net, validation_accuracy, training_accuracy

  return net, validation_accuracy, training_accuracy


def test_model(model, patch_test):
  result = model.predict(patch_test)
  predicted_class = np.argmax(result, axis=-1)
  return predicted_class

   
def load_patches(root_path, folder, from_pix2pix=False):
  imgs_dir = root_path + folder + '/imgs/'
  masks_dir = root_path + folder + '/masks/'
  img_files = os.listdir(imgs_dir)
  np.save(root_path + '/img_files.npy', img_files)
  patches = []
  patches_ref = []

  #todo remover ou parametrizar isso aqui
  if len(img_files) > 5000:
    img_files = img_files[:5000]

  for i in range(len(img_files)):
    if i%100==0:
      print('Loaded {} images'.format(i))
    img_path = imgs_dir + img_files[i]
    mask_path = masks_dir + img_files[i]
    img = np.load(img_path)
    img = normalization(img, norm_type = 3) # normaliza entre -1 e +1
    patches.append(img)
    mask = np.load(mask_path)
    
    if from_pix2pix: # todo consertar a parada na saida do pix2pixutils pra nao precisar disso aqui
      mask = to_unet_format(mask)
    patches_ref.append(mask)
  return np.array(patches), np.array(patches_ref)


def to_unet_format(mask):
  # pix2pix is fed a 3-channels mask, but u-net expects a 1-channel mask
  mask = mask[:,:,0]
  # print(mask.shape)
  unique, counts = np.unique(mask, return_counts=True)
  # print(unique, counts)
  mask[mask == 1] = 2
  mask[mask == 0] = 1
  mask[mask == -1] = 0
  unique, counts = np.unique(mask, return_counts=True)
  print(unique, counts)
  return mask


def load_tiles(root_path, testing_tiles_dir, tiles_ts):
  dir_ts = root_path + testing_tiles_dir
  img_list = []
  ref_list = []
  for num_tile in tiles_ts:
    img = np.load(dir_ts + str(num_tile) + '_img.npy')
    img = normalization(img, norm_type = 3) # normaliza entre -1 e +1
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
        print(thr)  

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

image_stack, final_mask = get_dataset(config)
print('[*] Normalizing image array...')
image_array = normalization(image_stack.copy(), 3) # -1 to +1
mask_tiles = create_mask(final_mask.shape[0], final_mask.shape[1], grid_size=(5, 4))

print('[*] Creating padded image...')
time_ts = []
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

# check normalization, maybe apply some map
print('[*] Loading patches...')
# Training patches
patches_train, patches_tr_ref = load_patches(root_path, training_dir) # retorna np.array(patches), np.array(patches_ref)
if config['synthetic_data_path'] != '':
  synt_data_path = config['data_path'] + config['synthetic_data_path']
  patches_train_synt, patches_tr_synt_ref = load_patches(root_path, config['synthetic_data_path'], from_pix2pix=True)
  print(patches_tr_ref.shape, patches_tr_synt_ref.shape)
  print(np.unique(patches_tr_ref), np.unique(patches_tr_synt_ref))
  patches_train = np.concatenate((patches_train, patches_train_synt))
  patches_tr_ref = np.concatenate((patches_tr_ref, patches_tr_synt_ref))
# Validation patches
patches_val, patches_val_ref = load_patches(root_path, validation_dir)
# Test patches
# patches_test, patches_test_ref = load_patches(root_path, testing_dir)
# Test tiles
tiles_test, tiles_test_ref = load_tiles(root_path, testing_tiles_dir, tiles_ts)

print("[*] Patches for Training:", str(patches_train.shape), str(patches_tr_ref.shape))
print("[*] Patches for Validation:", str(patches_val.shape), str(patches_val_ref.shape))
# print("[*] Patches for Testing:", str(patches_test.shape), str(patches_test_ref.shape))
print("[*] Tiles for Testing:", str(tiles_test.shape), str(tiles_test_ref.shape))

patches_val_lb_h = tf.keras.utils.to_categorical(patches_val_ref, number_class)
# patches_te_lb_h = tf.keras.utils.to_categorical(patches_test_ref, number_class)
patches_tr_lb_h = tf.keras.utils.to_categorical(patches_tr_ref, number_class)


def batch_generator(batches, image, reference, target_size, number_class):
    """Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator.
    """
    image = image.reshape(-1, image.shape[-1])
    reference = reference.reshape(final_mask.shape[0]*final_mask.shape[1])
    while True:
        batch_x, batch_y = next(batches)
        batch_x = np.squeeze(batch_x.astype('int64'))
        # print(batch_x.shape)
        batch_img = np.zeros((batch_x.shape[0], target_size, target_size, image.shape[-1]))
        batch_ref = np.zeros((batch_x.shape[0], target_size, target_size, number_class))
        
        for i in range(batch_x.shape[0]):
            if np.random.rand()>0.5:
                batch_x[i] = np.rot90(batch_x[i], 1)
            batch_img[i] = image[batch_x[i]] 
            batch_ref[i] = tf.keras.utils.to_categorical(reference[batch_x[i]] , number_class)
                       
        yield (batch_img, batch_ref)

train_datagen = ImageDataGenerator(horizontal_flip = True,
                                   vertical_flip = True)
valid_datagen = ImageDataGenerator(horizontal_flip = True, 
                                   vertical_flip = True)

X_train = patches_train
X_valid = patches_val

print(X_train.shape, X_valid.shape)

y_train = np.zeros((len(X_train)))
y_valid = np.zeros((len(X_valid)))

train_gen = train_datagen.flow(np.expand_dims(X_train, axis = -1), y_train, batch_size=batch_size,  shuffle=True)
valid_gen = valid_datagen.flow(X_valid, y_valid, batch_size=batch_size, shuffle=False)

number_class = 3
train_gen_crops = batch_generator(train_gen, image_array, final_mask, patch_size, number_class)
valid_gen_crops = batch_generator(valid_gen, image_array, final_mask, patch_size, number_class)

exp = 1
path_exp = root_path+'experiments/exp'+str(exp)
path_models = path_exp+'/models'
path_maps = path_exp+'/pred_maps'
os.makedirs(path_exp, exist_ok=True)   
os.makedirs(path_models, exist_ok=True)   
os.makedirs(path_maps, exist_ok=True)

model = build_unet(input_shape, nb_filters, number_class)

method = 'unet'
weights = [0.2, 0.8, 0]
adam = Adam(lr = 1e-3 , beta_1=0.9)
loss = weighted_categorical_crossentropy(weights)

time_tr = []
times = 5  
for tm in range(0,times):
    print('time: ', tm)
    model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
    model.summary()
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='min')
    checkpoint = ModelCheckpoint(path_models+ '/' + method +'_'+str(tm)+'.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    lr_reduce = ReduceLROnPlateau(factor=0.9, min_delta=0.0001, patience=5, verbose=1)
    callbacks_list = [earlystop, checkpoint]
    # train the model
    start_training = time.time()
    history = model.fit_generator(train_gen_crops,
                              steps_per_epoch=len(X_train)*3//train_gen.batch_size,
                              validation_data=valid_gen_crops,
                              validation_steps=len(X_valid)*3//valid_gen.batch_size,
                              epochs=100,
                              callbacks=callbacks_list)
    end_training = time.time() - start_training
    time_tr.append(end_training)
time_tr_array = np.asarray(time_tr)
# Save training time
np.save(path_exp+'/metrics_tr.npy', time_tr_array)
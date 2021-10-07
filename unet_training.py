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

root_path = config['data_path']

training_dir = '/training_data' #/augmented_dataset' 
validation_dir = '/validation_data'
testing_dir = '/testing_data'
testing_tiles_dir = '/testing_data/tiles_ts/'

batch_size = config['batch_size_unet']
epochs = config['epochs_unet']
nb_filters = config['nb_filters']
number_runs = config['times']
patience_value = config['patience_value']

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
output_folder = config['training_output_path'] + 'output_' + st + '_patchsize_' + str(patch_size) + '_batchsize_' + str(batch_size) + '_epochs_' + str(epochs) + '_patience_' + str(patience_value)
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

   
def load_patches(root_path, folder):
  imgs_dir = root_path + folder + '/imgs/'
  masks_dir = root_path + folder + '/masks/'
  img_files = os.listdir(imgs_dir)
  patches = []
  patches_ref = []
  for i in range(len(img_files)):
    if i%100==0:
      print('Loaded {} images'.format(i))
    img_path = imgs_dir + img_files[i]
    mask_path = masks_dir + img_files[i]
    img = np.load(img_path)
    img = normalization(img, norm_type = 3) # normaliza entre -1 e +1
    patches.append(img)
    patches_ref.append(np.load(mask_path))
  return np.array(patches), np.array(patches_ref)


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
patches_train, patches_tr_ref = load_patches(root_path, training_dir)
# Validation patches
patches_val, patches_val_ref = load_patches(root_path, validation_dir)
# Test patches
patches_test, patches_test_ref = load_patches(root_path, testing_dir)
# Test tiles
tiles_test, tiles_test_ref = load_tiles(root_path, testing_tiles_dir, tiles_ts)

print("[*] Patches for Training:", str(patches_train.shape), str(patches_tr_ref.shape))
print("[*] Patches for Validation:", str(patches_val.shape), str(patches_val_ref.shape))
print("[*] Patches for Testing:", str(patches_test.shape), str(patches_test_ref.shape))
print("[*] Tiles for Testing:", str(tiles_test.shape), str(tiles_test_ref.shape))

patches_val_lb_h = tf.keras.utils.to_categorical(patches_val_ref, number_class)
patches_te_lb_h = tf.keras.utils.to_categorical(patches_test_ref, number_class)
patches_tr_lb_h = tf.keras.utils.to_categorical(patches_tr_ref, number_class)

os.makedirs(output_folder + '/checkpoints', exist_ok=True)

adam = Adam(lr = 0.0001 , beta_1=0.9)
# 0: forest, 1: new deforestation, 2: old deforestation
weights = [0.1, 0.6, 0.0] # desconsidero a classe 2 nesse problema
print("[*] Weights for CE:", weights)
loss = weighted_categorical_crossentropy(weights)

cmap = matplotlib.colors.ListedColormap(['blue', 'olive', 'brown'])

for run in range(number_runs):

  net = build_unet((patch_size, patch_size, channels), nb_filters, number_class)
  net.summary()
  net.compile(loss = loss, optimizer=adam , metrics=['accuracy'])

  model, validation_accuracy, training_accuracy = train_model(net, patches_train, patches_tr_lb_h, patches_val, patches_val_lb_h, batch_size, epochs, patience_value)
  name = 'checkpoints/model_{}_epochs_{}_bsize_{}.h5'.format(run, epochs, batch_size)
  model.save(output_folder + '/' + name)

  fig = plt.figure(figsize=(8, 8))
  plt.subplot(2, 1, 1)
  plt.plot(training_accuracy, label='Training Accuracy')
  plt.plot(validation_accuracy, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.ylabel('Accuracy')
  plt.ylim([min(plt.ylim()),1])
  fig.savefig(output_folder + '/accuracy_model_' + str(run) + '.png')
  plt.close(fig)

  # test the model on testing patches 
  print(patches_test.shape)
  predicted_labels = test_model(model, patches_test)
  metrics = compute_metrics(patches_test_ref.flatten(), predicted_labels.flatten())
  print("Accuracy:", metrics[0])
  print("F1 Score:", metrics[1])
  print("Recall:", metrics[2])
  print("Precision:", metrics[3])

  # testing the model
  new_model = build_unet(input_shape, nb_filters, number_class)
  # model = load_model(name, compile=False)
  for l in range(1, len(model.layers)):
    new_model.layers[l].set_weights(model.layers[l].get_weights())
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
  del prob_recontructed, model, patches_pred

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

ax2 = fig.add_subplot(122)
plt.title('Reference')
ax2.imshow(final_mask, cmap = cmap)
ax2.axis('off')
fig.savefig('mean_map_and_ref.png')

mean_prob = mean_prob[:final_mask.shape[0], :final_mask.shape[1]]

# print(tiles_ts.shape)
print(mask_tiles.shape)
print(mean_prob.shape)

mask_amazon_ts = np.zeros((mask_tiles.shape)).astype('float32')
for ts_ in tiles_ts:
    mask_amazon_ts[mask_tiles == ts_] = 1

ref1 = np.ones_like(final_mask).astype(np.float32)
ref1 [final_mask == 2] = 0
TileMask = mask_amazon_ts * ref1
GTTruePositives = final_mask==1
Pmax = np.max(mean_prob[GTTruePositives * TileMask ==1])
print(Pmax)

# adicionar funcao de metricas recall e precision

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
print('mAP', AP)

# Plot Recall vs. Precision curve
plt.close('all')
fig = plt.figure(figsize=(15,10))
plt.plot(metrics_copy[:,0],metrics_copy[:,1])
plt.plot(metrics_copy[:,0],metrics_copy[:,2])
plt.grid()
fig.savefig(output_folder + '/roc.png')

# f = open(output_folder + '/results.txt', 'w+')
# metrics_ = []
# for tile in tiles_ts:
#   current_tile_pred = mean_prob[mask_tiles == tile]
#   print(current_tile_pred.shape)
#   print(np.unique(current_tile_pred))
#   tile_mask = final_mask[mask_tiles == tile]
#   print(tile_mask.shape)
#   print(np.unique(tile_mask))
#   metrics = compute_metrics(tile_mask.flatten(), current_tile_pred.flatten())
#   metrics_.append(metrics)
#   f.write('metrics - ' + str(tile) + ':\n')
#   f.write('accuracy, f1score, recall, precision')
#   f.write(str(metrics[0]) + ',' +  str(metrics[1]) + ',' + str(metrics[2]) + ',' + str(metrics[3]) + '\n')

# # accuracy, f1score, recall, precision
# final_metrics = np.mean(metrics_)
# f.write('final_metrics:')
# f.write('accuracy, f1score, recall, precision')
# f.write(str(final_metrics[0]) + ',' +  str(final_metrics[1]) + ',' + str(final_metrics[2]) + ',' + str(final_metrics[3]) + '\n')

# f.close()


# 000000

# rows, cols = np.where(mask_tiles == 17.)
# x1 = np.min(rows)
# y1 = np.min(cols)
# x2 = np.max(rows)
# y2 = np.max(cols)
# print(mean_prob.shape, final_mask.shape)
# tile_pred = mean_prob[x1:x2 + 1, y1:y2 + 1]
# tile_ref = final_mask[x1:x2 + 1, y1:y2 + 1]
# # Plot mean map and reference
# fig = plt.figure(figsize=(15,10))
# ax1 = fig.add_subplot(121)
# plt.title('Prediction')
# im = ax1.imshow(tile_pred, cmap =cmap)
# ax1.axis('off')
# fig.savefig(output_folder + '/prediction_17.png')
# plt.close()
# fig = plt.figure(figsize=(15,10))
# ax2 = fig.add_subplot(122)
# plt.title('Reference')
# ax2.imshow(tile_ref, cmap =cmap)
# # plt.colorbar(im)
# ax2.axis('off')
# fig.savefig(output_folder + '/ref_17.png')
# plt.close()

  # prob_recontructed = pred_reconctruct(h, w, num_patches_x, num_patches_y, patch_size_rows, patch_size_cols, patches_pred)
  # np.save(output_folder +'/prob_'+str(run)+'.npy',prob_recontructed) 
  # del prob_recontructed
  # prob_rec = np.zeros((image1_pad.shape[0],image1_pad.shape[1], times))
  # prob_rec[:,:,run] = np.load(output_folder + '/prob_'+str(run)+'.npy').astype(np.float32)
  # prob_rec[prob_rec >= 0.5] = 1
  # prob_rec[prob_rec < 0.5] = 0
  # mean_prob = np.mean(prob_rec, axis = -1)
  # np.save(output_folder + '/prob_mean.npy', mean_prob)

  # rows, cols = np.where(mask_tiles == 17.)
  # x1 = np.min(rows)
  # y1 = np.min(cols)
  # x2 = np.max(rows)
  # y2 = np.max(cols)
  # print(mean_prob.shape, final_mask.shape)
  # tile_pred = mean_prob[x1:x2 + 1, y1:y2 + 1]
  # tile_ref = final_mask[x1:x2 + 1, y1:y2 + 1]
  # # Plot mean map and reference
  # fig = plt.figure(figsize=(15,10))
  # ax1 = fig.add_subplot(121)
  # plt.title('Prediction')
  # im = ax1.imshow(tile_pred, cmap =cmap)
  # ax1.axis('off')
  # fig.savefig(output_folder + '/prediction_17.png')
  # plt.close()
  # fig = plt.figure(figsize=(15,10))
  # ax2 = fig.add_subplot(122)
  # plt.title('Reference')
  # ax2.imshow(tile_ref, cmap =cmap)
  # # plt.colorbar(im)
  # ax2.axis('off')
  # fig.savefig(output_folder + '/ref_17.png')
  # plt.close()

  # rows, cols = np.where(mask_tiles == 15.)
  # x1 = np.min(rows)
  # y1 = np.min(cols)
  # x2 = np.max(rows)
  # y2 = np.max(cols)
  # print(mean_prob.shape, final_mask.shape)
  # tile_pred = mean_prob[x1:x2 + 1, y1:y2 + 1]
  # tile_ref = final_mask[x1:x2 + 1, y1:y2 + 1]
  # # Plot mean map and reference
  # fig = plt.figure(figsize=(15,10))
  # ax1 = fig.add_subplot(121)
  # plt.title('Prediction')
  # im = ax1.imshow(tile_pred, cmap =cmap)
  # ax1.axis('off')
  # fig.savefig(output_folder + '/prediction_5.png')
  # plt.close()
  # fig = plt.figure(figsize=(15,10))
  # ax2 = fig.add_subplot(122)
  # plt.title('Reference')
  # ax2.imshow(tile_ref, cmap =cmap)
  # # plt.colorbar(im)
  # ax2.axis('off')
  # fig.savefig(output_folder + '/ref_5.png')
  # plt.close()


  # predicted_tiles = test_model(model, tiles_test)
  # test the model on tiles
  # for i in range(len(tiles_test)):
  #   print('TILE:', str(i), tiles_test[i].shape)
  #   predicted_tile = test_model(model, tiles_test[i])
  #   metrics = compute_metrics(tiles_test_ref[i].flatten(), predicted_tile.flatten())
  #   # print("Accuracy:", metrics[0])
  #   # print("F1 Score:", metrics[1])
  #   # print("Recall:", metrics[2])
  #   # print("Precision:", metrics[3])

  #   fig = plt.figure(figsize=(15,10))
  #   ax1 = fig.add_subplot(121)
  #   plt.title('Prediction')
  #   im = ax1.imshow(predicted_tile, cmap =cmap)
  #   ax1.axis('off')
  #   fig.savefig(output_folder + '/prediction_' + str(i) + '.png')
  #   plt.close()
  #   fig = plt.figure(figsize=(15,10))
  #   ax2 = fig.add_subplot(122)
  #   plt.title('Reference')
  #   ax2.imshow(tiles_test_ref[i], cmap =cmap)
  #   ax2.axis('off')
  #   fig.savefig(output_folder + '/ref_' + str(i) + '.png')
  #   plt.close()

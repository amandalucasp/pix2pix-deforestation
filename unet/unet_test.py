from sklearn.feature_extraction.image import extract_patches_2d
from contextlib import redirect_stdout
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from osgeo import ogr, gdal
import matplotlib.colors
from PIL import Image
import numpy as np
import datetime
import pathlib
import time
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

cmap = matplotlib.colors.ListedColormap(['blue', 'olive'])

patch_size = 256
channels = 3
stride = int(patch_size/2)
number_class = 2
root_path = '/share_epsilon/amandalucas/pix2pix/Sentinel2/samples_new_split/' 
training_dir = 'training_data'
validation_dir = 'validation_data'
testing_dir = 'testing_data'
lim_x = 17000 # 1000
lim_y = 9200 # 7000
CHANNELS = [1, 2, 6]

# output_folder = 'augmented_output_256_32_100_10_2021-07-05_02-47-02/'
output_folder = 'baseline_output_256_32_100_10_2021-07-05_01-33-13/'
name = output_folder + 'model_100_epochs_32_bsize.h5'

tiles_tr = [1,3,5,7,8,10,11,13,14,16,18,20,4,6,19]
tiles_val = [2, 9, 12] # [4,6,19]
tiles_ts = (list(set(np.arange(20)+1)-set(tiles_tr)-set(tiles_val)))

batch_size = 32
epochs = 100
nb_filters = [32, 64, 128]
times = 1
patience_value = 2


ts = time.time()


def pred_reconctruct(h, w, num_patches_x, num_patches_y, patch_size_x, patch_size_y, patches_pred):
    count = 0
    img_reconstructed = np.zeros((h, w)).astype(np.float32)
    for i in range(0,num_patches_y):
        for j in range(0,num_patches_x):
            img_reconstructed[patch_size_x*j:patch_size_x*(j+1),patch_size_y*i:patch_size_y*(i+1)]=patches_pred[count]
            count+=1
    return img_reconstructed


def load_tif_image(patch):
    # Read tiff Image
    print (patch)
    #img_tif = TIFF.open(patch)
    #img = img_tif.read_image()
    gdal_header = gdal.Open(patch)
    img = gdal_header.ReadAsArray()
    print(type(img), img.shape)
    img = np.moveaxis(img, 0, -1)
    print(type(img), img.shape)
    return img


def resize_image(image, height, width):
    im_resized = np.zeros((height, width, image.shape[2]), dtype='float32')
    for b in range(image.shape[2]):
        band = Image.fromarray(image[:,:,b])
        #(width, height) = (ref_2019.shape[1], ref_2019.shape[0])
        im_resized[:,:,b] = np.array(band.resize((width, height), resample=Image.NEAREST))
    return im_resized


def filter_outliers(img, bins=1000000, bth=0.001, uth=0.999, mask=[0]):
    img[np.isnan(img)]=0 # Filter NaN values.
    if len(mask)==1:
        mask = np.zeros((img.shape[:2]), dtype='int64')
    for band in range(img.shape[-1]):
        hist = np.histogram(img[:mask.shape[0], :mask.shape[1]][mask!=2, band].ravel(),bins=bins) # select not testing pixels
        cum_hist = np.cumsum(hist[0])/hist[0].sum()
        max_value = np.ceil(100*hist[1][len(cum_hist[cum_hist<uth])])/100
        min_value = np.ceil(100*hist[1][len(cum_hist[cum_hist<bth])])/100
        img[:,:, band][img[:,:, band]>max_value] = max_value
        img[:,:, band][img[:,:, band]<min_value] = min_value
    return img


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


def Train_model(net, patches_train, patches_tr_lb_h, patches_val, patches_val_lb_h, batch_size, epochs, patience_value):
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


def Test(model, patch_test):
  result = model.predict(patch_test)
  predicted_class = np.argmax(result, axis=-1)
  return predicted_class


def compute_metrics(true_labels, predicted_labels):
  accuracy = 100*accuracy_score(true_labels, predicted_labels)
  f1score = 100*f1_score(true_labels, predicted_labels, average=None)
  recall = 100*recall_score(true_labels, predicted_labels, average=None)
  precision = 100*precision_score(true_labels, predicted_labels, average=None)
  return accuracy, f1score, recall, precision


def extract_patches(image, reference, patch_size=256, stride=128):

  transformed_ref = np.zeros(shape=(reference.shape[0],reference.shape[1]))
  h, w, c = reference.shape
  print("Transforming reference image to Int (RGB -> Int)...")
  for row in range(h):
    for column in range(w):
      transformed_ref[row,column] = CLASSES.index(list(reference[row,column,:]))
  # print("Done. Extracting patches...")

  border_size = patch_size#int((patch_size-stride)/2)
  image = np.pad(image,((border_size,border_size),(border_size,border_size),(0,0)), mode='reflect')
  transformed_ref = np.pad(transformed_ref,((border_size,border_size),(border_size,border_size)), mode='reflect')

  print("Image shape:",image.shape,"\n Transf Reference shape:",transformed_ref.shape)

  plt.figure(figsize=(10, 10))
  ax = plt.subplot(1,2,1)
  im = plt.imshow(image)
  ax = plt.subplot(1,2,2)
  im = plt.imshow(transformed_ref)
  plt.show()

  image_list = []
  ref_list = []
  num_y = image.shape[0]
  num_x = image.shape[1]

  counter = 0
  for posicao_y in range(stride, (num_y-(stride)), stride):
    for posicao_x in range(stride, (num_x-(stride)), stride):
      y1 = posicao_y-stride
      y2 = posicao_y+stride
      x1 = posicao_x-stride
      x2 = posicao_x+stride
      aux = image[y1:y2, x1:x2,:]
      aux2 = transformed_ref[y1:y2, x1:x2]
      # print(counter,y1,y2,x1,x2, aux.shape)
      image_list.append(aux)
      ref_list.append(aux2)
      counter+=1

  img_array = np.array(image_list)
  ref_array = np.array(ref_list)

  return img_array, ref_array, border_size


def unpatch_image(patches, stride, border_size, original_image):

  predicted_img_shape = ((original_image.shape[0]+border_size),(original_image.shape[1]+border_size))

  new_image = np.zeros(shape=(predicted_img_shape))
  num_rows = predicted_img_shape[0] # vertical
  num_cols = predicted_img_shape[1] # horizontal
  pos_x = 0
  pos_y = 0
  for patch in patches:
    x0 = pos_x
    y0 = pos_y
    x1 = pos_x + stride
    # condição de borda
    if x1 > num_rows: # vertical
      x1 = num_rows
    y1 = pos_y + stride
    # condição de borda
    if y1 > num_cols: # horizontal
      y1 = num_cols
    # print(x0,y0,x1,y1)
    aux = patch[:x1-x0,:y1-y0]
    new_image[x0:x1,y0:y1] = aux
    # ando na horizontal 
    pos_y += stride

    if pos_y > num_cols: 
      # atingiu a última posição na horizontal
      pos_y = 0
      # avanço na vertical
      pos_x += stride

  # remove borders from padding
  output_image = new_image[border_size:new_image.shape[0],border_size:new_image.shape[1]]

  return output_image

   
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


def load_patches(root_path, folder):
  imgs_dir = root_path + folder + '/imgs/'
  masks_dir = root_path + folder + '/masks/'
  img_files = os.listdir(imgs_dir)
  # mask_files = os.list_dir(masks_dir)
  patches = []
  patches_ref = []
  for i in range(len(img_files)):
    if i%100==0:
      print('Loaded {} images'.format(i))
    img_path = imgs_dir + img_files[i]
    mask_path = masks_dir + img_files[i]
    patches.append(np.load(img_path))
    patches_ref.append(np.load(mask_path))
  return np.array(patches), np.array(patches_ref)


def create_mask(size_rows, size_cols, grid_size=(6,3)):
    num_tiles_rows = size_rows//grid_size[0]
    num_tiles_cols = size_cols//grid_size[1]
    print('Tiles size: ', num_tiles_rows, num_tiles_cols)
    patch = np.ones((num_tiles_rows, num_tiles_cols))
    mask = np.zeros((num_tiles_rows*grid_size[0], num_tiles_cols*grid_size[1]))
    count = 0
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            count = count+1
            mask[num_tiles_rows*i:(num_tiles_rows*i+num_tiles_rows), num_tiles_cols*j:(num_tiles_cols*j+num_tiles_cols)] = patch*count
    #plt.imshow(mask)
    print('Mask size: ', mask.shape)
    return mask


print(root_path)
sent2_2019_1 = load_tif_image(root_path + '2019_10m_b2348.tif').astype('float32')
sent2_2019_2 = load_tif_image(root_path + '2019_20m_b5678a1112.tif').astype('float32')
sent2_2019_2 = resize_image(sent2_2019_2.copy(), sent2_2019_1.shape[0], sent2_2019_1.shape[1])
sent2_2019 = np.concatenate((sent2_2019_1, sent2_2019_2), axis=-1)
sent2_2019 = sent2_2019[:, :, CHANNELS]
del sent2_2019_1, sent2_2019_2
image_stack = filter_outliers(sent2_2019.copy())
del sent2_2019

final_mask = np.load(root_path+'final_mask_label.npy').astype('float32')
final_mask[final_mask == 2] = 1
image_stack = image_stack[:lim_x, :lim_y, :]
final_mask = final_mask[:lim_x, :lim_y]
print('[1]final mask unique:', np.unique(final_mask))
# Normalization
type_norm = 1 # mesma que foi usada nos imgs e masks
image_array = normalization(image_stack.copy(), type_norm)
print(np.min(image_array), np.max(image_array))
del image_stack
# Create tile mask
mask_tiles = create_mask(final_mask.shape[0], final_mask.shape[1], grid_size=(5, 4))
image_array = image_array[:mask_tiles.shape[0], :mask_tiles.shape[1],:]
final_mask = final_mask[:mask_tiles.shape[0], :mask_tiles.shape[1]]

# Training and validation mask
mask_tr_val = np.zeros((mask_tiles.shape)).astype('float32')
for tr_ in tiles_tr:
    mask_tr_val[mask_tiles == tr_] = 1

for val_ in tiles_val:
    mask_tr_val[mask_tiles == val_] = 2

mask_amazon_ts = np.zeros((mask_tiles.shape)).astype('float32')
for ts_ in tiles_ts:
    mask_amazon_ts[mask_tiles == ts_] = 1

plt.imshow(mask_tiles)
plt.savefig(output_folder + '/mask_tiles.png')
plt.figure(figsize=(10,5))
plt.imshow(final_mask, cmap = cmap)
print('[2]final mask unique:', np.unique(final_mask))
plt.savefig(output_folder + '/final_mask.png')

tm = 0
time_ts = []
# tiles mask
n_pool = 3
n_rows = 5
n_cols = 4
rows, cols = image_array.shape[:2]
pad_rows = rows - np.ceil(rows/(n_rows*2**n_pool))*n_rows*2**n_pool
pad_cols = cols - np.ceil(cols/(n_cols*2**n_pool))*n_cols*2**n_pool
print(pad_rows, pad_cols)

npad = ((0, int(abs(pad_rows))), (0, int(abs(pad_cols))), (0, 0))
image1_pad = np.pad(image_array, pad_width=npad, mode='reflect')
print('image1_pad.shape:', image1_pad.shape)
h, w, c = image1_pad.shape
print('h, w, c:', h, w, c)
patch_size_rows = h//n_rows
patch_size_cols = w//n_cols
num_patches_x = int(h/patch_size_rows)
num_patches_y = int(w/patch_size_cols)

input_shape=(patch_size_rows,patch_size_cols, c)
print('input_shape:', input_shape)
new_model = build_unet(input_shape, nb_filters, number_class)
model = load_model(name, compile=False)

for l in range(1, len(model.layers)):
    new_model.layers[l].set_weights(model.layers[l].get_weights())

start_test = time.time()
patch_t = []
for i in range(0,num_patches_y):
    for j in range(0,num_patches_x):
        patch = image1_pad[patch_size_rows*j:patch_size_rows*(j+1), patch_size_cols*i:patch_size_cols*(i+1), :]
        predictions_ = new_model.predict(np.expand_dims(patch, axis=0)) # talvez era isso que faltava no outro
        del patch 
        patch_t.append(predictions_[:,:,:,1])
        del predictions_
end_test =  time.time() - start_test
patches_pred = np.asarray(patch_t).astype(np.float32)
print(patches_pred.shape)
# (20, 1, 3400, 2304)

prob_recontructed = pred_reconctruct(h, w, num_patches_x, num_patches_y, patch_size_rows, patch_size_cols, patches_pred)
np.save(output_folder +'/prob_'+str(tm)+'.npy',prob_recontructed) 
time_ts.append(end_test)
del prob_recontructed, model, patches_pred

time_ts_array = np.asarray(time_ts)
# Save test time
np.save(output_folder + '/metrics_ts.npy', time_ts_array)

# Compute mean of the tm predictions maps
prob_rec = np.zeros((image1_pad.shape[0],image1_pad.shape[1], times))
print('prob_rec:', prob_rec.shape)
# for tm in range (0, times):
print(tm)
# prob_rec[:,:,tm] = np.load('prob_'+str(tm)+'.npy').astype(np.float32)
prob_rec[:,:,tm] = np.load(output_folder + '/prob_'+str(tm)+'.npy').astype(np.float32)

prob_rec[prob_rec >= 0.5] = 1
prob_rec[prob_rec < 0.5] = 0

print('prob_rec unique:', np.unique(prob_rec))
mean_prob = np.mean(prob_rec, axis = -1)
np.save(output_folder + '/prob_mean.npy', mean_prob)

rows, cols = np.where(mask_tiles == 17.)
x1 = np.min(rows)
y1 = np.min(cols)
x2 = np.max(rows)
y2 = np.max(cols)
print(mean_prob.shape, final_mask.shape)
tile_pred = mean_prob[x1:x2 + 1, y1:y2 + 1]
tile_ref = final_mask[x1:x2 + 1, y1:y2 + 1]
# Plot mean map and reference
fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(121)
plt.title('Prediction')
im = ax1.imshow(tile_pred, cmap =cmap)
ax1.axis('off')
fig.savefig(output_folder + '/prediction_17.png')
plt.close()
fig = plt.figure(figsize=(15,10))
ax2 = fig.add_subplot(122)
plt.title('Reference')
ax2.imshow(tile_ref, cmap =cmap)
# plt.colorbar(im)
ax2.axis('off')
fig.savefig(output_folder + '/ref_17.png')
plt.close()

rows, cols = np.where(mask_tiles == 5.)
x1 = np.min(rows)
y1 = np.min(cols)
x2 = np.max(rows)
y2 = np.max(cols)
print(mean_prob.shape, final_mask.shape)
tile_pred = mean_prob[x1:x2 + 1, y1:y2 + 1]
tile_ref = final_mask[x1:x2 + 1, y1:y2 + 1]
# Plot mean map and reference
fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(121)
plt.title('Prediction')
im = ax1.imshow(tile_pred, cmap =cmap)
ax1.axis('off')
fig.savefig(output_folder + '/prediction_5.png')
plt.close()
fig = plt.figure(figsize=(15,10))
ax2 = fig.add_subplot(122)
plt.title('Reference')
ax2.imshow(tile_ref, cmap =cmap)
# plt.colorbar(im)
ax2.axis('off')
fig.savefig(output_folder + '/ref_5.png')
plt.close()


# Computing metrics
mean_prob = mean_prob[:final_mask.shape[0], :final_mask.shape[1]]
# ref1 = np.ones_like(final_mask).astype(np.float32)

# print('mean_prob.shape:', mean_prob)
# print('final_mask.shape:', final_mask)

# tiles_ts = [17, 15]
# mask_tiles
f = open(output_folder + 'results.txt', 'w+')
metrics_ = []
for tile in tiles_ts:
  current_tile_pred = mean_prob[mask_tiles == tile]
  print(current_tile_pred.shape)
  tile_mask = final_mask[mask_tiles == tile]
  print(tile_mask.shape)
  metrics = compute_metrics(tile_mask.flatten(), current_tile_pred.flatten())
  metrics_.append(metrics)
  f.write('metrics - ' + str(tile) + ':\n')
  f.write('accuracy, f1score, recall, precision')
  f.write(str(metrics[0]) + ',' +  str(metrics[1]) + ',' + str(metrics[2]) + ',' + str(metrics[3]) + '\n')

# accuracy, f1score, recall, precision
final_metrics = np.mean(metrics_)
f.write('final_metrics:')
f.write('accuracy, f1score, recall, precision')
f.write(str(final_metrics[0]) + ',' +  str(final_metrics[1]) + ',' + str(final_metrics[2]) + ',' + str(final_metrics[3]) + '\n')

f.close()

# ref1 [final_mask == 2] = 0 # class 1 and class 2 were merged
# TileMask = mask_amazon_ts * ref1
# GTTruePositives = final_mask==1
    
# Npoints = 50
# Pmax = np.max(mean_prob[GTTruePositives * TileMask ==1])
# ProbList = np.linspace(Pmax,0,Npoints)

# print('matrics_AA_recall')
# metrics_ = matrics_AA_recall(ProbList, mean_prob, final_mask, mask_amazon_ts, 625)
# np.save(output_folder + '/acc_metrics.npy',metrics_)

# print('Complete NaN values')
# # Complete NaN values
# metrics_copy = metrics_.copy()
# metrics_copy = complete_nan_values(metrics_copy)

# print('Complete mAP')
# # Comput Mean Average Precision (mAP) score 
# Recall = metrics_copy[:,0]
# print('Recall:', Recall)
# Precision = metrics_copy[:,1]
# print('Precision:', Precision)
# AA = metrics_copy[:,2]
    
# DeltaR = Recall[1:]-Recall[:-1]
# AP = np.sum(Precision[:-1]*DeltaR)
# print('mAP:', AP)

# # Plot Recall vs. Precision curve
# plt.close('all')
# fig = plt.figure(figsize=(15,10))
# plt.plot(metrics_copy[:,0],metrics_copy[:,1])
# plt.plot(metrics_copy[:,0],metrics_copy[:,2])
# plt.grid()
# fig.savefig(output_folder + '/recall-precision-curve.png')

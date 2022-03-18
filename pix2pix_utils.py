
#from SpectralNormalizationKeras import ConvSN2D
from sklearn.preprocessing import MinMaxScaler
import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from skimage import img_as_ubyte
import tensorflow as tf
#import tensorflow_addons as tfa
import numpy as np
import imageio
import yaml
import cv2
import os

import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )

stream = open('./config.yaml')
config = yaml.load(stream, Loader=yaml.FullLoader)

IMG_WIDTH = config['image_width']
IMG_HEIGHT = config['image_height'] 
NUM_CHANNELS = config['output_channels'] 


def draw_mask_contour(mask, real_t2):
  # get NEW DEFORESTATION mask 
  inp_mask = mask == 1
  mask = tf.cast(inp_mask, tf.float32).numpy().squeeze()[:, :, -1]
  mask = mask.astype(np.uint8)
  # draw contours
  contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  contour = contours[-1] 
  real_t2 = cv2.drawContours(real_t2, contour, -1, (0,255,0), 1)
  return real_t2


def plot_image(plot_list, columns, rows, title, filename=None, pad=1):
  fig = plt.figure()
  for i in range(0, columns*rows):
    fig.add_subplot(rows, columns, i + 1)
    plt.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
    if i == 0 or i == 1:
      plt.title(title[i])
    plt.imshow(plot_list[i]  * 0.5 + 0.5)
  fig.tight_layout(pad=pad)
  if filename:
    fig.savefig(filename)
  plt.close(fig)


def generate_images(model, test_input, tar, filename=None):
  # test_input : (t1, mask)
  # tar: real t2
  prediction = model(test_input, training=True)
  # prediction: fake t2

  chans = [0, 1, 3]
  t1 = cv2.cvtColor(test_input[0].numpy()[:,:,chans], cv2.COLOR_BGR2RGB)
  mask = cv2.cvtColor(test_input[0].numpy()[:,:,-1], cv2.COLOR_BGR2RGB)
  real_t2 = cv2.cvtColor(tar[0].numpy()[:,:,chans], cv2.COLOR_BGR2RGB)
  fake_t2 = cv2.cvtColor(prediction[0].numpy()[:,:,chans], cv2.COLOR_BGR2RGB)

  fake_t2 = draw_mask_contour(mask, fake_t2)
  real_t2 = draw_mask_contour(mask, real_t2)

  plot_list = [t1, real_t2, fake_t2]

  if filename:
    columns = 3
    rows = 1
    title = ['T1', 'Real T2', 'Fake T2']
    plot_image(plot_list, columns, rows, title, filename, pad=3)

  return prediction[0]


def plot_imgs(generator, test_ds, out_dir, counter):
  i = 0
  plot_list = []
  for inp, tar in test_ds.take(3):
    prediction = generate_images(generator, inp, tar)
    # inp: (t1, mask)
    # tar: real t2
    # prediciton: fake t2

    chans = [0, 1, 3, 10, 11, 13]
    t1 = cv2.cvtColor(inp[0].numpy()[:,:,chans[:3]], cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(inp[0].numpy()[:,:,-1], cv2.COLOR_BGR2RGB)
    real_t2 = cv2.cvtColor(tar[0].numpy()[:,:,chans[:3]], cv2.COLOR_BGR2RGB)
    fake_t2 = cv2.cvtColor(prediction.numpy()[:,:,chans[:3]], cv2.COLOR_BGR2RGB)

    fake_t2 = draw_mask_contour(mask, fake_t2)
    real_t2 = draw_mask_contour(mask, real_t2)

    plot_list.append(t1)
    plot_list.append(real_t2) # real t2
    plot_list.append(fake_t2) # fake t2
    i+=1

  columns = 3
  rows = 3
  title = ['T1', 'Real T2', 'Fake T2']
  filename = out_dir + str(counter) + '.png'
  plot_image(plot_list, columns, rows, title, filename)


def save_synthetic_img(t1_mask, t2_img, saving_path, filename):
  t1_mask = np.squeeze(t1_mask)
  os.makedirs(saving_path + '/imgs/', exist_ok=True)
  os.makedirs(saving_path + '/masks/', exist_ok=True)
  os.makedirs(saving_path + '/combined/', exist_ok=True)
  
  t1 = t1_mask[:,:,:NUM_CHANNELS]
  mask = t1_mask[:,:,NUM_CHANNELS:]
  t1_t2 = np.concatenate((t1, t2_img), axis=-1)

  mask_copy = mask.copy()
  h, w, _ = t1.shape
  combined = np.zeros(shape=(h,w*3,3), dtype=np.float32)
  combined[:,:w,:] = t1[:,:,config['debug_channels']]
  combined[:,w:w*2,:] = cv2.cvtColor(mask_copy, cv2.COLOR_GRAY2RGB)
  combined[:,2*w:,:] = t2_img.numpy()[:,:,config['debug_channels']]

  combined = (combined - np.min(combined))/np.ptp(combined)
  combined = img_as_ubyte(combined)
  imageio.imwrite(saving_path + '/combined/' + filename + '.png', cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
  np.save(saving_path + '/imgs/' + filename + '.npy', t1_t2)
  np.save(saving_path + '/masks/' + filename + '.npy', mask)


def load_npy_sample(npy_file):
  image = np.load(npy_file)
  w = image.shape[1]
  w = w // 3
  # image is T1 // T2 // mask
  t1_image = image[:,:w, :]
  t2_image = image[:,w:2*w,:]
  mask_image = image[:,2*w:,:]

  input_image = np.concatenate((t1_image, mask_image), axis=-1)
  input_image = make_mask_2d(input_image)
  real_image = t2_image
  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)
  return input_image, real_image


def load_npy(npy_file):
  image = np.load(npy_file)
  w = image.shape[1]
  w = w // 3
  # image is T1 // T2 // mask
  t1_image = image[:,:w, :]
  t2_image = image[:,w:2*w,:]
  mask_image = image[:,2*w:,:]

  input_image = np.concatenate((t1_image, mask_image), axis=-1)
  real_image = t2_image
  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)
  return input_image, real_image


def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.AREA)
  real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.AREA)
  return input_image, real_image


def random_crop(input_image, real_image, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS=3):
  t1 = input_image[:, :, :NUM_CHANNELS]
  mask = input_image[:, :, NUM_CHANNELS:]
  stacked_image = tf.stack([t1, mask, real_image])
  cropped_image = tf.image.random_crop(stacked_image, size=[3, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS])
  concat_image = tf.concat([cropped_image[0], cropped_image[1]], axis=-1)
  return concat_image, cropped_image[2]


def normalize(input_image, real_image):   
  # Normalizing the images to [-1, 1]
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1
  return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image, NUM_CHANNELS=3):
  # Resizing to 286x286
  input_image, real_image = resize(input_image, real_image, (IMG_HEIGHT + 30), (IMG_WIDTH + 30))
  # Random cropping back to 256x256
  input_image, real_image = random_crop(input_image, real_image, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)
  if tf.random.uniform(()) > 0.5:
    # Random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)
  return input_image, real_image


def make_mask_2d(input_image):
  # transform n-dimensional mask to 2-dimensional
  # input: (patch_size, patch_size, c) -> t1 (c//2) + mask (c//2)
  # output: (patch_size, patch_size, c//2 + 1) -> t1 (c//2) + mask (1)
  t1, mask = tf.split(input_image, 2, axis=-1)
  return tf.concat([t1, tf.expand_dims(mask[:,:,0], axis=-1)], axis=-1)


def load_npy_train(image_file):
  input_image, real_image = load_npy(image_file)
  input_image, real_image = random_jitter(input_image, real_image, NUM_CHANNELS)
  input_image = make_mask_2d(input_image)
  return input_image, real_image


def load_npy_test(image_file):
  input_image, real_image = load_npy(image_file)
  input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
  input_image = make_mask_2d(input_image)
  return input_image, real_image


def set_shapes(img, label, img_shape, label_shape):
  img.set_shape(img_shape)
  label.set_shape(label_shape)
  return img, label


def downsample(filters, size, apply_batchnorm=True, strides=2, padding_mode='same'):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  if apply_batchnorm:
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=strides, padding=padding_mode,
                             kernel_initializer=initializer, use_bias=False)) # use_bias = True is redundant when using BN
    result.add(tf.keras.layers.BatchNormalization(epsilon=1e-5,
                                                  momentum=0.1,
                                                  gamma_initializer=tf.random_normal_initializer(1.0, 0.02)))
  else:
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=strides, padding=padding_mode,
                             kernel_initializer=initializer))
  result.add(tf.keras.layers.LeakyReLU(alpha=0.2))
  return result


def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))
  result.add(tf.keras.layers.BatchNormalization(epsilon=1e-5,
                                                  momentum=0.1,
                                                  gamma_initializer=tf.random_normal_initializer(1.0, 0.02)))
  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))
  result.add(tf.keras.layers.ReLU())
  return result


def res_encoder_block(input_data, n_filters, k_size=3, strides=2, activation='relu', padding='same', SN=False, batchnorm=True, name='None'):
    # weight initialization
    init = tf.random_normal_initializer(0., 0.02)
    if SN:
        x = ConvSN2D(n_filters, k_size, strides=strides, padding=padding, kernel_initializer=init, name=name+'_convSN2D')(input_data)
    else:
        x = tf.keras.layers.Conv2D(n_filters, k_size, strides=strides, padding=padding, kernel_initializer=init, name=name+'_conv2D')(input_data)

    if batchnorm:
        x = tf.keras.layers.BatchNormalization(momentum=0.8, name=name+'_bn')(x, training=True)
    if activation is 'LReLU':
        x = tf.keras.layers.LeakyReLU(alpha=0.2, name=name+'_act_LReLU')(x)        
    else:
        x = tf.keras.layers.Activation('relu', name=name+'_act_relu')(x)
    return x


def res_decoder_block(input_data, n_filters, k_size=3, strides=2, padding='same', name='None'):
    # weight initialization
    init = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv2DTranspose(n_filters, k_size, strides=strides, padding=padding, kernel_initializer=init, name=name+'_deconv2D')(input_data)
    x = tf.keras.layers.BatchNormalization(momentum=0.8, name=name+'_bn')(x, training=True)
    x = tf.keras.layers.Activation('relu', name=name+'_act_relu')(x)
    return x


def residual_block(input_x, n_kernels, name='name'):
    x = res_encoder_block(input_x, n_kernels, strides=1, name=name+'rba')
    x = tf.keras.layers.Dropout(0.5, name=name+'drop')(x, training=True)
    x = res_encoder_block(x, n_kernels,  strides=1, activation='linear', name=name+'rbb')
    x = tf.keras.layers.Add(name=name+'concatenate')([x, input_x])
    return x
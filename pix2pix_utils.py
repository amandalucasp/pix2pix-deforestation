
#from SpectralNormalizationKeras import ConvSN2D
from sklearn.preprocessing import MinMaxScaler
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
NUM_CHANNELS = config['output_channels'] # NUMERO DE CANAIS DE CADA IMAGEM (T1, T2, MASCARA)
BINARY_MASK = config['binary_mask']


def plot_imgs(generator, test_ds, out_dir, counter):
  i = 0
  plot_list = []
  for inp, tar in test_ds.take(3):
    prediction = generate_images(generator, inp, tar)
    # inp: masked t2
    # tar: real t2
    # prediciton: fake t2

    chans = [0, 1, 3]
    masked_t2 = cv2.cvtColor(inp[0].numpy()[:,:,chans], cv2.COLOR_BGR2RGB)
    real_t2 = cv2.cvtColor(tar[0].numpy()[:,:,chans], cv2.COLOR_BGR2RGB)
    fake_t2 = cv2.cvtColor(prediction.numpy()[:,:,chans], cv2.COLOR_BGR2RGB)
    # get mask from masked t2
    inp_ones = masked_t2 == 0
    mask = tf.cast(inp_ones, tf.float32).numpy().squeeze()[:, :, -1]
    mask = mask.astype(np.uint8)
    # draw contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key = cv2.contourArea)
    real_t2 = cv2.drawContours(real_t2, contour, -1, (0,255,0), 1)
    plot_list.append(real_t2) # real t2
    plot_list.append(fake_t2) # fake t2
    i+=1

  fig = plt.figure()
  columns = 2
  rows = 3
  title = ['Masked Real T2', 'Predicted T2']
  for i in range(0, columns*rows):
    fig.add_subplot(rows, columns, i + 1)
    plt.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
    if i == 0 or i == 1:
      plt.title(title[i], fontsize=20)
    plt.imshow(plot_list[i]  * 0.5 + 0.5)
  fig.tight_layout(pad=1)
  fig.savefig(out_dir + str(counter) + '.png')
  plt.close(fig)


def save_synthetic_img(input, prediction, saving_path, filename):
  # input: masked_t2 
  # prediction: fake_t2 
  
  os.makedirs(saving_path + '/masked_t2/', exist_ok=True)
  os.makedirs(saving_path + '/fake_t2/', exist_ok=True)
  os.makedirs(saving_path + '/combined/', exist_ok=True)

  masked_t2 = np.squeeze(input)
  fake_t2 = prediction.numpy()
  np.save(saving_path + '/masked_t2/' + filename + '.npy', masked_t2)
  np.save(saving_path + '/fake_t2/' + filename + '.npy', fake_t2)

  
  # salva imagem JPEG para visualizar
  h, w, _ = masked_t2.shape
  combined = np.zeros(shape=(h,w*2,3), dtype=np.float32)
  combined[:,:w,:] = masked_t2[:,:,config['debug_channels']]
  combined[:,w:,:] = fake_t2[:,:,config['debug_channels']]
  combined = (combined - np.min(combined))/np.ptp(combined)
  combined = img_as_ubyte(combined)
  imageio.imwrite(saving_path + '/combined/' + filename + '.png', cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))


def load_npy(npy_file):
  image = np.load(npy_file)
  w = image.shape[1]
  w = w // 4

  # image is T1 // T2 // mask // masked T2
  t1_image = image[:,:w, :]
  t2_image = image[:,w:2*w,:]
  mask_image = image[:,2*w:3*w,:]
  masked_t2 = image[:,3*w:,:]

  # input image: masked T2
  input_image = tf.cast(masked_t2, tf.float32)
  # real image: real T2
  real_image = tf.cast(t2_image, tf.float32)

  return input_image, real_image


def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.AREA)
  real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.AREA)
  return input_image, real_image


def random_crop(input_image, real_image, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS=3):
  stacked_image = tf.stack([input_image, real_image])
  cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS])
  return cropped_image[0], cropped_image[1]


@tf.function()
def random_jitter(input_image, real_image, NUM_CHANNELS=3):
  # Resizing 
  input_image, real_image = resize(input_image, real_image, (IMG_HEIGHT + 30), (IMG_WIDTH + 30))
  # Random cropping back to original size
  input_image, real_image = random_crop(input_image, real_image, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)
  if tf.random.uniform(()) > 0.5:
    # Random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)
  return input_image, real_image


def load_npy_train(image_file):
  input_image, real_image = load_npy(image_file)
  input_image, real_image = random_jitter(input_image, real_image, NUM_CHANNELS)
  return input_image, real_image


def load_npy_test(image_file):
  input_image, real_image = load_npy(image_file)
  input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
  return input_image, real_image


def set_shapes(img, label, img_shape, label_shape):
  img.set_shape(img_shape)
  label.set_shape(label_shape)
  return img, label


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


def cross_entropy_loss(labels, logits):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    return loss


def lsgan_loss(labels, logits):
        loss = tf.reduce_mean(tf.math.squared_difference(logits, labels))
        return loss 
    
    
def l1_loss(a, b):
    loss = tf.reduce_mean(tf.math.abs(a - b))
    return loss


def downsample(filters, size, apply_batchnorm=True, strides=2, padding_mode='same', sn=False):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  if sn:
    # nao funciona em tf2
    result.add(
      tf.keras.layers.ConvSN2D(filters, size, strides=strides, padding=padding_mode,
                             kernel_initializer=initializer, use_bias=False))
  else:
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=strides, padding=padding_mode,
                             kernel_initializer=initializer, use_bias=False))
  if apply_batchnorm:
    # teste affine-layer
    result.add(tf.keras.layers.BatchNormalization(epsilon=1e-5,
                                                  momentum=0.1,
                                                  gamma_initializer=tf.random_normal_initializer(1.0, 0.02)))
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
  #result.add(tf.keras.layers.BatchNormalization(momentum=0.8,
  #                                                gamma_initializer=tf.random_normal_initializer(1.0, 0.02)))
  # affine-layer:
  result.add(tf.keras.layers.BatchNormalization(epsilon=1e-5,
                                                  momentum=0.1,
                                                  gamma_initializer=tf.random_normal_initializer(1.0, 0.02)))
  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))
  result.add(tf.keras.layers.ReLU())
  return result


def residual_block_v1(filters, size, apply_batchnorm=True, padding_mode='same'):
    # x = encoder_block(input_x, filters, strides=1, name=name+'rba')
    # x = Dropout(0.5, name=name+'drop')(x, training=True)
    # x = encoder_block(x, filters,  strides=1, activation='linear', name=name+'rbb')
    # x = Add(name=name+'concatenate')([x, input_x])
    result = tf.keras.Sequential()
    # encoder 1
    initializer = tf.random_normal_initializer(0., 0.02)
    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=1, padding=padding_mode,
                             kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
      result.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    result.add(tf.keras.layers.ReLU())
    # dropout
    result.add(tf.keras.layers.Dropout(0.5))
    # encoder 2
    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=1, padding=padding_mode,
                             kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
      result.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    result.add(tf.keras.layers.ReLU())
    return result


def generate_images(model, test_input, tar, filename=None):
  # test_input : masked t2
  # tar: real t2

  prediction = model(test_input, training=True)
  # prediction: fake t2

  if filename:
    fig = plt.figure()
    chans = [0, 1, 3] # rgb channels
    display_list = [cv2.cvtColor(test_input[0].numpy()[:,:,chans], cv2.COLOR_BGR2RGB), # masked t2
                    cv2.cvtColor(tar[0].numpy()[:,:,chans], cv2.COLOR_BGR2RGB), # real t2
                    cv2.cvtColor(prediction[0].numpy()[:,:,chans], cv2.COLOR_BGR2RGB)] # fake t2
    title = ['Input Masked T2', 'Real T2', 'Predicted T2']
    for i in range(3):
      plt.subplot(1, 3, i+1)
      plt.title(title[i])
      # Getting the pixel values in the [0, 1] range to plot.
      plt.imshow(display_list[i] * 0.5 + 0.5)
      plt.axis('off')
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.tight_layout(pad=1)
    fig.savefig(filename)
    plt.close(fig)

  return prediction[0]
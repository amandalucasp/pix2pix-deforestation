from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import imageio
import yaml
import cv2
import os

stream = open('./config.yaml')
config = yaml.load(stream)

IMG_WIDTH = config['image_width']
IMG_HEIGHT = config['image_height'] 
NUM_CHANNELS = config['output_channels'] # NUMERO DE CANAIS DE CADA IMAGEM (T1, T2, MASCARA)


def save_synthetic_img(t1_mask, t2_img, saving_path, filename):
    t1_mask = np.squeeze(t1_mask)
    os.makedirs(saving_path + '/imgs/', exist_ok=True)
    os.makedirs(saving_path + '/masks/', exist_ok=True)
    os.makedirs(saving_path + '/combined/', exist_ok=True)
    
    t1 = t1_mask[:,:,:NUM_CHANNELS]
    mask = t1_mask[:,:,NUM_CHANNELS:]
    t1_t2 = np.concatenate((t1, t2_img), axis=-1)

    h, w, c = t1.shape
    combined = np.zeros(shape=(h,w*3,c), dtype=np.float32)
    combined[:,:w,:] = t1
    combined[:,w:w*2,:] = mask
    combined[:,2*w:,:] = t2_img

    imageio.imwrite(saving_path + '/combined/' + filename + '.png', cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    np.save(saving_path + '/imgs/' + filename + '.npy', t1_t2)
    np.save(saving_path + '/masks/' + filename + '.npy', mask)


def load_npy(npy_file):
  image = np.load(npy_file)
  w = image.shape[1]
  w = w // 3
  # image is T1 // T2 // mask
  t1_image = image[:,:w, :]
  t2_image = image[:,w:2*w,:]
  mask_image = image[:,2*w:,:]
  input_image = np.concatenate((t1_image, mask_image), axis=-1) #  t1 + mascara
  real_image = t2_image
  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)
  return input_image, real_image


def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
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


def load_npy_train(image_file):
  input_image, real_image = load_npy(image_file)
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)
  return input_image, real_image


def load_npy_test(image_file):
  input_image, real_image = load_npy(image_file)
  input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)
  return input_image, real_image


def set_shapes(img, label, img_shape, label_shape):
  img.set_shape(img_shape)
  label.set_shape(label_shape)
  return img, label


def downsample(filters, size, apply_batchnorm=True, padding_mode='same'):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding=padding_mode,
                             kernel_initializer=initializer, use_bias=False))
  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())
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
  result.add(tf.keras.layers.BatchNormalization())
  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))
  result.add(tf.keras.layers.ReLU())
  return result


def generate_images(model, test_input, tar, filename=None):
  prediction = model(test_input, training=True)
  if filename:
    fig = plt.figure(figsize=(15, 15))
    display_list = [cv2.cvtColor(test_input[0][:,:,:NUM_CHANNELS].numpy(), cv2.COLOR_BGR2RGB),
                    cv2.cvtColor(test_input[0][:,:,NUM_CHANNELS:].numpy(), cv2.COLOR_BGR2RGB),
                    cv2.cvtColor(tar[0].numpy(), cv2.COLOR_BGR2RGB),
                    cv2.cvtColor(prediction[0].numpy(), cv2.COLOR_BGR2RGB)]
    title = ['Input Image T1', 'Mask', 'Actual T2', 'Predicted T2']
    for i in range(4):
      plt.subplot(1, 4, i+1)
      plt.title(title[i])
      # Getting the pixel values in the [0, 1] range to plot.
      plt.imshow(display_list[i] * 0.5 + 0.5)
      plt.axis('off')
    fig.savefig(filename)
    plt.close(fig)
  return prediction[0]
#from SpectralNormalizationKeras import ConvSN2D
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import argparse
import pathlib
import imageio
import shutil
import glob
import yaml
import time
import cv2
import os

import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )

from pix2pix_utils import *

ap = argparse.ArgumentParser()
ap.add_argument('-t', '--train', help='run train + inference', action='store_true')
ap.add_argument('-i', '--inference', help='run inference on input', action='store_true')
args = ap.parse_args()

print(tf.executing_eagerly())

stream = open('./config.yaml')
config = yaml.load(stream, Loader=yaml.FullLoader)

time_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
if args.inference:
  time_string = time_string + '_inference'
output_folder = config['data_path'] + '/pix2pix/' + time_string
os.makedirs(output_folder)
shutil.copy('./config.yaml', output_folder)

np.random.seed(0)

BATCH_SIZE = config['batch_size']
IMG_WIDTH = config['image_width']
IMG_HEIGHT = config['image_height']
OUTPUT_CHANNELS = config['output_channels']
# Generator Loss Term
LAMBDA = config['lambda']
GAN_WEIGHT = config['gan_weight']
ngf = config['ngf']
ndf = config['ndf']
EPS = 1e-12
npy_path = pathlib.Path(config['data_path'])

config['synthetic_masks_path'] = config['data_path'] + config['synthetic_masks_path']
synthetic_masks_path = pathlib.Path(config['synthetic_masks_path'])

print(config)

checkpoint_dir = output_folder + '/training_checkpoints'
log_dir= output_folder + "/logs/"
out_dir = output_folder + "/output_images/"
 
train_files = glob.glob(str(npy_path / 'training_data/pairs/*.npy'))

inp, re = load_npy_sample(train_files[0])
input_shape = [inp.shape[0], inp.shape[1], config['output_channels'] + 1] # ex.: 128x128x10 + 1
target_shape = re.shape # 128x128x10
print('input_shape:', input_shape, np.min(inp), np.max(inp))
print('target_shape:', target_shape, np.min(re), np.max(re))

# Dataset items used for training
if 'buffer_size' in config:
  BUFFER_SIZE = config['buffer_size'] 
else:
  BUFFER_SIZE = len(train_files)

train_ds = tf.data.Dataset.from_tensor_slices(train_files)
train_ds = train_ds.map(lambda item: tuple(tf.compat.v1.numpy_function(load_npy_train, [item], [tf.float32,tf.float32])))
train_ds = train_ds.map(lambda img, label: set_shapes(img, label, input_shape, target_shape))
train_ds = train_ds.shuffle(BUFFER_SIZE)
train_ds = train_ds.batch(BATCH_SIZE)
print('[*] Train Dataset:')
print(train_ds.element_spec)
print(train_ds)

try:
  test_files = glob.glob(str(npy_path / 'testing_data/pairs/*.npy'))
except tf.errors.InvalidArgumentError:
  test_files = glob.glob(str(npy_path / 'validation_data/pairs/*.npy'))

test_ds = tf.data.Dataset.from_tensor_slices(test_files)
test_ds = test_ds.map(lambda item: tuple(tf.compat.v1.numpy_function(load_npy_test, [item], [tf.float32,tf.float32])))
test_ds = test_ds.map(lambda img, label: set_shapes(img, label, input_shape, target_shape))
test_ds = test_ds.batch(BATCH_SIZE)

print('[*] Test Dataset:')
print(test_ds.element_spec)
print(test_ds)


def resGenerator(input_shape=[256, 256, 3], ngf=64, last_act='tanh', n_residuals=9, summary=False, model_file=None, name='gan_g_'):

    init = tf.random_normal_initializer(0., 0.02)
    n_rows = input_shape[0]
    n_cols = input_shape[1]
    in_c_dims = input_shape[2]
    out_c_dims = input_shape[2]

    input_shape = (n_rows, n_cols, in_c_dims)
    input_layer = tf.keras.layers.Input(shape=input_shape, name=name+'_input')
    
    x = input_layer
    #(input_data, n_filters, k_size=3, strides=2, activation='relu', padding='same', SN=False, batchnorm=True, name='None')
    x = res_encoder_block(x, 1*ngf, k_size=7, strides=1, batchnorm=False, name=name+'_e1')
    x = res_encoder_block(x, 2*ngf, name=name+'e2') # rows/2, cols/2
    x = res_encoder_block(x, 4*ngf, name=name+'e3') # rows/4, cols/4

    for i in range(n_residuals):
        x = residual_block(x, n_kernels=4*ngf, name=name+str(i+1)+'_')  # rows/4, cols/4

    # (input_data, n_filters, k_size=3, strides=2, padding='same', name='None')
    x = res_decoder_block(x, 2*ngf, name=name+'d1') # rows/2, cols/2            
    x = res_decoder_block(x, 1*ngf, name=name+'d2') # rows, cols
    x = tf.keras.layers.Conv2D(OUTPUT_CHANNELS, 7, padding='same',  kernel_initializer=init, name=name+'d_out')(x)   # rows, cols

    output = tf.keras.layers.Activation(last_act, name=name+last_act)(x)

    model = tf.keras.Model(inputs=[input_layer], outputs=[output], name='Generator'+name[-3:])
    if (summary):
        model.summary()
    return model


def Generator(input_shape=[256, 256, 3], ngf=64, residual=False, n_residuals=3, drop_blocs=0):
  inputs = tf.keras.layers.Input(shape=[input_shape[0], input_shape[1], input_shape[2]])

  down_stack = [
    downsample(ngf, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64) 64, 64, 32 
    downsample(ngf * 2, 4),  # (batch_size, 64, 64, 128) 32, 32, 64
    downsample(ngf * 4, 4),  # (batch_size, 32, 32, 256) 16, 16, 128
    downsample(ngf * 8, 4),  # (batch_size, 16, 16, 512) 8, 8, 256
    downsample(ngf * 8, 4),  # (batch_size, 8, 8, 512) 4, 4, 256
    downsample(ngf * 8, 4),  # (batch_size, 4, 4, 512) 2, 2, 256
    downsample(ngf * 8, 4),  # (batch_size, 2, 2, 512) 1, 1, 256
    downsample(ngf * 8, 4),  # (batch_size, 1, 1, 512) 1, 1, 256
  ]

  up_stack = [
    upsample(ngf * 8, 4, apply_dropout=True),  # (batch_size, 2, 2, 512) -> 1024 
    upsample(ngf * 8, 4, apply_dropout=True),  # (batch_size, 4, 4, 512) -> 1024
    upsample(ngf * 8, 4, apply_dropout=True),  # (batch_size, 8, 8, 512) -> 1024
    upsample(ngf * 8, 4),  # (batch_size, 16, 16, 512) -> 1024
    upsample(ngf * 4, 4),  # (batch_size, 32, 32, 256) -> 512
    upsample(ngf * 2, 4),  # (batch_size, 64, 64, 128) -> 256
    upsample(ngf, 4),  # (batch_size, 128, 128, 64) ->  128
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x = inputs

  if input_shape[0] == 128:
    # removendo uma camada do encoder e uma do decoder
    down_stack = down_stack[:-1]
    up_stack = up_stack[1:]

  if residual:

    if drop_blocs > 0:
      down_stack = down_stack[:-drop_blocs]
      up_stack = up_stack[drop_blocs:]
    nb = min(2**(7-drop_blocs),8)
    print('nb:', nb)
    residual_b = residual_block(ngf * nb, 4)
    for down in down_stack:
      x = down(x)
    for i in range(n_residuals):
      x_old = x
      x = residual_b(x)
      x = tf.keras.layers.Add()([x, x_old])
    for up in up_stack:
      x = up(x)

  else:

    # Downsampling through the model
    skips = []
    for down in down_stack:
      x = down(x)
      skips.append(x)

    skips = reversed(skips[:-1]) # last layer is connected directly to the decoders

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
      x = up(x)
      x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


if config['residual_generator']:
  generator = resGenerator(input_shape, ngf)
else:
  generator = Generator(input_shape, ngf, config['residual_generator'],
                      config['number_residuals'], config['drop_blocs'])
print(generator.summary())
gen_output = generator(inp[tf.newaxis, ...], training=False)
fig = plt.figure()
plt.imshow(gen_output[0].numpy()[:,:,config['debug_channels']]*0.5 + 0.5)
fig.savefig(output_folder + '/gen_output.png')
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
  #gan_loss = tf.reduce_mean(-tf.math.log(disc_generated_output + EPS)) # affine-layer
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
  total_gen_loss = (GAN_WEIGHT * gan_loss) + (LAMBDA * l1_loss)
  return total_gen_loss, gan_loss, l1_loss


def resDiscriminator(input_shape=[256, 256, 3], target_shape=[256, 256, 3], ndf=64, name='d'):
  init = tf.random_normal_initializer(0., 0.02)
  inp = tf.keras.layers.Input(shape=input_shape, name='input_image')
  tar = tf.keras.layers.Input(shape=target_shape, name='target_image')
  d = tf.keras.layers.concatenate([inp, tar], axis=3)
  
  d = res_encoder_block(d, 1*ndf, k_size=4, activation='LReLU', SN=False, batchnorm=False, name=name+'_1')
  d = res_encoder_block(d, 2*ndf, k_size=4, activation='LReLU', SN=False, batchnorm=False, name=name+'_2')
  d = res_encoder_block(d, 4*ndf, k_size=4, activation='LReLU', SN=False, batchnorm=False, name=name+'_3')

  d = tf.keras.layers.ZeroPadding2D()(d)
  d = res_encoder_block(d, 8*ndf, k_size=4, activation='LReLU', strides=1, SN=False, batchnorm=False, padding='valid', name=name+'_4')
  d = tf.keras.layers.ZeroPadding2D()(d)
  logits = tf.keras.layers.Conv2D(1, (4,4), padding='valid', kernel_initializer=init, name=name+'_conv2D_5')(d)
  out = tf.keras.layers.Activation('sigmoid', name=name+'_act_sigmoid')(logits)

  model = tf.keras.Model(inputs=[inp, tar], outputs=[out, logits])
  return model


def Discriminator(input_shape=[256, 256, 3], target_shape=[256, 256, 3], ndf=64):
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=input_shape, name='input_image')
  tar = tf.keras.layers.Input(shape=target_shape, name='target_image')

  x = tf.keras.layers.concatenate([inp, tar], axis=3)  # (batch_size, 256, 256, channels*2)

  # layer_1
  x = tf.keras.layers.ZeroPadding2D()(x)
  down1 = downsample(ndf, 4, apply_batchnorm=False, strides=2, padding_mode='valid')(x)  # (batch_size, 128, 128, 64) 64, 64, 32
  # layer_2
  down1_pad = tf.keras.layers.ZeroPadding2D()(down1)
  down2 = downsample(ndf * 2, 4, padding_mode='valid')(down1_pad)  # (batch_size, 64, 64, 128) 32, 32, 64
  # layer_3
  down2_pad = tf.keras.layers.ZeroPadding2D()(down2)
  down3 = downsample(ndf * 4, 4, padding_mode='valid')(down2_pad)  # (batch_size, 32, 32, 256) 16, 16, 128
  # layer_4
  down3_pad = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256) 18, 18, 128
  conv = tf.keras.layers.Conv2D(ndf * 8, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(down3_pad)  # (batch_size, 31, 31, 512) 15, 15, 256
  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
  leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)(batchnorm1)
  # layer_5
  down4_pad = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512) 17, 17, 256
  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(down4_pad)
                                #activation='sigmoid')(down4_pad) # affine-layer
  
  return tf.keras.Model(inputs=[inp, tar], outputs=last)


if config['residual_generator']:
  discriminator = resDiscriminator(input_shape, target_shape, ndf)
else:
  discriminator = Discriminator(input_shape, target_shape, ndf)
disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False)
fig = plt.figure()
print(disc_out.shape, np.min(disc_out), np.max(disc_out))
plt.imshow(disc_out[0].numpy()[..., -1]*255, vmin=-20, vmax=20, cmap='RdBu_r')
plt.colorbar()
fig.savefig(output_folder + '/disc_out.png')
print(discriminator.summary())


def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
  total_disc_loss = real_loss + generated_loss
  #total_disc_loss = tf.reduce_mean(-(tf.math.log(disc_real_output + EPS) + tf.math.log(1 - disc_generated_output + EPS))) # affine-layer
  return total_disc_loss

generator_optimizer = tf.keras.optimizers.Adam(config['lr'], beta_1=config['beta1'])
discriminator_optimizer = tf.keras.optimizers.Adam(config['lr'], beta_1=config['beta1'])

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

for example_input, example_target in test_ds.take(1):
  generate_images(generator, example_input, example_target, output_folder + '/sample.png')


def cross_entropy_loss(labels, logits):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    return loss

def lsgan_loss(labels, logits):
        loss = tf.reduce_mean(tf.math.squared_difference(logits, labels))
        return loss 
    
def l1_loss(a, b):
    loss = tf.reduce_mean(tf.math.abs(a - b))
    return loss

os.makedirs(out_dir, exist_ok=True)
summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"))


@tf.function
def res_train_step(input_image, target, step):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True) 
    disc_real_output, disc_real_logits = discriminator([input_image, target], training=True)
    disc_generated_output, disc_fake_logits = discriminator([input_image, gen_output], training=True)
    # discriminator loss
    d_loss_real = lsgan_loss(tf.ones_like(disc_real_output), disc_real_logits)
    d_loss_fake = lsgan_loss(tf.zeros_like(disc_real_output), disc_fake_logits)
    disc_loss = (d_loss_real + d_loss_fake) / 2.0
    # reconstruction loss
    gen_l1_loss = LAMBDA * l1_loss(target, gen_output)
    # generator loss
    gen_gan_loss = lsgan_loss(tf.ones_like(disc_generated_output), disc_fake_logits)
    gen_total_loss =  gen_gan_loss + gen_l1_loss

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
    tf.summary.scalar('disc_loss', disc_loss, step=step//1000)
  
  return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss

@tf.function
def train_step(input_image, target, step):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True) 

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
    tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

  return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss


def plot_imgs(generator, test_ds, out_dir, counter):
  i = 0
  plot_list = []
  for inp, tar in test_ds.take(3):
    prediction = generate_images(generator, inp, tar)
    chans = [0, 1, 3, 10, 11, 13]
    plot_list.append(cv2.cvtColor(inp[0].numpy()[:,:,chans[:3]], cv2.COLOR_BGR2RGB))
    plot_list.append(cv2.cvtColor(inp[0].numpy()[:,:,-1], cv2.COLOR_BGR2RGB))
    plot_list.append(cv2.cvtColor(tar[0].numpy()[:,:,chans[:3]], cv2.COLOR_BGR2RGB))
    plot_list.append(cv2.cvtColor(prediction.numpy()[:,:,chans[:3]], cv2.COLOR_BGR2RGB))
    i+=1
  fig = plt.figure()
  columns = 4
  rows = 3
  for i in range(0, columns*rows):
    fig.add_subplot(rows, columns, i + 1)
    plt.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
    plt.imshow(plot_list[i]  * 0.5 + 0.5)
  fig.tight_layout(pad=1)
  fig.savefig(out_dir + str(counter) + '.png')
  plt.close(fig)


def fit(train_ds, test_ds, config):
  # example_input, example_target = next(iter(test_ds.take(1)))
  start_time = time.time()
  steps = config['training_steps']
  counter = 0
  for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
    if (step) % 1000 == 0:
      if step != 0:
        print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')
        print('gen_total_loss:', gen_total_loss)
        print('gen_gan_loss:', gen_gan_loss)
        print('gen_l1_loss:', gen_l1_loss)
        print('disc_loss:', disc_loss)
      start = time.time()
      print(f"Step: {step//1000}k")
    if (step) % 5000 == 0:
      plot_imgs(generator, test_ds, out_dir, counter)
      counter +=5000

    if config['residual_generator']:
      gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = res_train_step(input_image, target, step)
    else:
      gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = train_step(input_image, target, step)

    # Training step
    if (step+1) % 10 == 0:
      print('.', end='', flush=True)

    if (step + 1) % config['checkpoint_steps'] == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)
  print(f'[*] Training took a total of: {time.time()-start_time:.2f} secs.')


if args.train:
  start = time.time()
  fit(train_ds, test_ds, config)
  print(f'[*] Time taken for training: {time.time()-start:.2f} sec\n')

  os.makedirs(output_folder + '/generated_plots_test/')
  synthetic_path = output_folder + '/synthetic_data_test/'

  counter = 0
  for inp, tar in test_ds:
    prediction = generate_images(generator, inp, tar, output_folder + '/generated_plots_test/' + str(counter) + '.png')
    save_synthetic_img(inp, prediction, synthetic_path, str(counter))
    counter+=1

  tr_input_files = glob.glob(str( synthetic_masks_path / 'pairs/*.npy'))
  if len(tr_input_files) > config['max_input_samples']:
    tr_input_files = tr_input_files[:config['max_input_samples']]
    print('Using the first', str(config['max_input_samples']), 'pairs.')
  pix2pix_input_ds = tf.data.Dataset.from_tensor_slices(tr_input_files)
  pix2pix_input_ds = pix2pix_input_ds.map(lambda item: tuple(tf.compat.v1.numpy_function(load_npy_test, [item], [tf.float32,tf.float32])))
  pix2pix_input_ds = pix2pix_input_ds.map(lambda img, label: set_shapes(img, label, input_shape, target_shape))
  pix2pix_input_ds = pix2pix_input_ds.batch(BATCH_SIZE)

  os.makedirs(output_folder + '/generated_plots_random/')
  synthetic_path = output_folder + '/synthetic_data_random/'

  counter = 0
  for inp, tar in pix2pix_input_ds:
    print(inp.shape, inp[0].shape) # ver de tirar esse [0]
    prediction = generate_images(generator, inp, tar, output_folder + '/generated_plots_random/' + str(counter) + '.png')
    save_synthetic_img(inp[0], prediction, synthetic_path, str(counter))
    counter+=1

if args.inference:
  # checkpoint_prefix = os.path.join(config['checkpoint_folder'], "ckpt")
  checkpoint.restore(tf.train.latest_checkpoint(config['checkpoint_folder']))

  tr_input_files = glob.glob(str( synthetic_masks_path / 'pairs/*.npy'))
  pix2pix_input_ds = tf.data.Dataset.from_tensor_slices(tr_input_files)
  pix2pix_input_ds = pix2pix_input_ds.map(lambda item: tuple(tf.compat.v1.numpy_function(load_npy_test, [item], [tf.float32,tf.float32])))
  pix2pix_input_ds = pix2pix_input_ds.map(lambda img, label: set_shapes(img, label, input_shape, target_shape))
  pix2pix_input_ds = pix2pix_input_ds.batch(BATCH_SIZE)

  os.makedirs(output_folder + '/generated_plots_random/')
  synthetic_path = output_folder + '/synthetic_data_random/'

  counter = 0
  for inp, tar in pix2pix_input_ds:
    print(inp.shape, inp[0].shape) # ver de tirar esse [0]
    prediction = generate_images(generator, inp, tar, output_folder + '/generated_plots_random/' + str(counter) + '.png')
    save_synthetic_img(inp[0], prediction, synthetic_path, str(counter))
    counter+=1
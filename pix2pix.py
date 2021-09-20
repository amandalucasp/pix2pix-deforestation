from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import pathlib
import imageio
import shutil
import glob
import yaml
import time
import os

from pix2pix_utils import *

print(tf.executing_eagerly())

stream = open('./config.yaml')
config = yaml.load(stream)

time_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
# output_folder = config['training_name'] + '/' + time_string
output_folder = config['data_path'].replace('samples_','train_') + '/' + time_string
os.makedirs(output_folder)
shutil.copy('./config.yaml', output_folder)

BATCH_SIZE = config['batch_size']
# Each image is 256x256 in size
IMG_WIDTH = config['image_width']
IMG_HEIGHT = config['image_height']
OUTPUT_CHANNELS = config['output_channels']
input_shape = [IMG_WIDTH, IMG_HEIGHT, 3]
# Generator Loss Term
LAMBDA = config['lambda']

npy_path = pathlib.Path(config['data_path'])
inp, re = load_npy(str(npy_path/'training_data/pairs/0.npy'))
print(np.unique(inp))
print(inp.shape, re.shape)

checkpoint_dir = output_folder + '/training_checkpoints'
log_dir= output_folder + "/logs/"
out_dir = output_folder + "/output_images/"
 
train_files = glob.glob(str(npy_path / 'training_data/pairs/*.npy'))

# Dataset items used for training
if 'buffer_size' in config:
  BUFFER_SIZE = config['buffer_size'] 
else:
  BUFFER_SIZE = len(train_files)

train_ds = tf.data.Dataset.from_tensor_slices(train_files)
train_ds = train_ds.map(lambda item: tuple(tf.compat.v1.numpy_function(load_npy_train, [item], [tf.float32,tf.float32])))
train_ds = train_ds.map(lambda img, label: set_shapes(img, label, input_shape))
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
test_ds = test_ds.map(lambda img, label: set_shapes(img, label, input_shape))
test_ds = test_ds.batch(BATCH_SIZE)

print('[*] Test Dataset:')
print(test_ds.element_spec)
print(test_ds)


def Generator(input_shape=[256, 256, 3]):
  inputs = tf.keras.layers.Input(shape=[input_shape[0], input_shape[1], input_shape[2]])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


generator = Generator()
gen_output = generator(inp[tf.newaxis, ...], training=False)
fig = plt.figure()
plt.imshow(gen_output[0, ...])
fig.savefig(output_folder + '/gen_output.png')

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
  total_gen_loss = gan_loss + (LAMBDA * l1_loss)
  return total_gen_loss, gan_loss, l1_loss


def Discriminator(input_shape=[256, 256, 3], target_shape=[256, 256, 3]):
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=input_shape, name='input_image') # T1 + 
  tar = tf.keras.layers.Input(shape=target_shape, name='target_image') # T2

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)

discriminator = Discriminator()

disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False)
fig = plt.figure()
plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
plt.colorbar()
fig.savefig(output_folder + '/disc_out.png')

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
  total_disc_loss = real_loss + generated_loss
  return total_disc_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

for example_input, example_target in test_ds.take(1):
  generate_images(generator, example_input, example_target, output_folder + '/sample.png')


os.makedirs(out_dir, exist_ok=True)
summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

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


def fit(train_ds, test_ds, config):
  example_input, example_target = next(iter(test_ds.take(1)))
  start = time.time()
  steps = config['training_steps']
  counter = 0
  for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
    if (step) % 1000 == 0:
      # display.clear_output(wait=True)
      if step != 0:
        print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')
      start = time.time()
      # if (step) % 5000 == 0: 
      i = 0
      plot_list = []
      for inp, tar in test_ds.take(3):
        prediction = generate_images(generator, inp, tar)
        plot_list.append(inp[0])
        plot_list.append(tar[0])
        plot_list.append(prediction)
        i+=1
      fig = plt.figure(figsize=(15, 15))
      title = ['Input Image', 'Ground Truth', 'Predicted Image']
      columns = 3
      rows = 3
      for i in range(0, columns*rows):
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(plot_list[i]  * 0.5 + 0.5)
      fig.savefig(out_dir + str(counter) + '.png')
      plt.title(title)
      plt.close(fig)
      counter +=1000

      print(f"Step: {step//1000}k")

    train_step(input_image, target, step)

    # Training step
    if (step+1) % 10 == 0:
      print('.', end='', flush=True)

    if (step + 1) % config['checkpoint_steps'] == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

start = time.time()
fit(train_ds, test_ds, config)
print(f'[*] Time taken for training: {time.time()-start:.2f} sec\n')

os.makedirs(output_folder + '/generated_plots/')
os.makedirs(config['data_path'] + '/synthetic_data_' + time_string + '/')
counter = 0
for inp, tar in test_ds:
  prediction = generate_images(generator, inp, tar, output_folder + '/generated_plots/' + str(counter) + '.png')
  imageio.imwrite(config['data_path'] + '/synthetic_data_' + time_string + '/' + str(counter) + '.png', prediction)
  np.save(config['data_path'] + '/synthetic_data_' + time_string + '/' + str(counter) + '.npy', prediction)
  counter+=1
import tensorflow as tf
import os
import pathlib
import time
import datetime
from matplotlib import pyplot as plt
from IPython im
import numpy as np
import pathlib
from generator import *
from discriminator import *
from helpers import *

# The facade training set consist of 400 images
BUFFER_SIZE = 400
NUM_GENERATED_IMAGES = 5 # images generated on test
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1
# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256
# Number of output channels for the Generator
OUTPUT_CHANNELS = 3
# Paths
PATH = './data'
OUTPUT_PATH './output'
log_dir="logs/"
checkpoint_dir = '/training_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
# Lambda term from the Loss Function
LAMBDA = 100

# sample_image_from_np = np.load(str(PATH / '0.npy'))
# print(sample_image_from_np.shape, sample_image_from_np.dtype)

inp, re = load_npy_as_image(str(path_npy / '0.npy'))
print(inp.shape, inp.dtype, re.shape, re.dtype)

fig = plt.figure(figsize=(6, 6))
for i in range(4):
  rj_inp, rj_re = random_jitter(inp, re)
  plt.subplot(2, 2, i + 1)
  plt.imshow(rj_inp / 255.0)
  plt.axis('off')
fig.savefig(OUTPUT_PATH/'preprocessad_sample_input.png')


"""## Build an input pipeline with `tf.data`"""

train_dataset = tf.data.Dataset.list_files(str(PATH / 'train/*.npy'))
train_dataset = train_dataset.map(load_npy_train, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

try:
  test_dataset = tf.data.Dataset.list_files(str(PATH / 'test/*.npy'))
except tf.errors.InvalidArgumentError:
  test_dataset = tf.data.Dataset.list_files(str(PATH / 'val/*.npy'))
test_dataset = test_dataset.map(load_npy_test)
test_dataset = test_dataset.batch(BATCH_SIZE)


"""Visualize the generator model architecture:"""

generator = Generator()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64, to_file=OUTPUT_PATH/'generator_model.png')

"""Test the generator:"""

gen_output = generator(inp[tf.newaxis, ...], training=False)
fig = plt.figure(figsize=(6, 6))
plt.imshow(gen_output[0, ...])
fig.savefig(OUTPUT_PATH/'generator_output.png')

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

"""Visualize the discriminator model architecture:"""

discriminator = Discriminator()
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64, to_file=OUTPUT_PATH/'discriminator_model.png')

"""Test the discriminator:"""

disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False)
fig = plt.figure(figsize=(6, 6))
plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
plt.colorbar()
fig.savefig(OUTPUT_PATH/'discriminator_output.png')

"""
## Define the optimizers and a checkpoint-saver
"""

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


"""Finally, run the training loop:"""

fit(train_dataset, test_dataset, steps=40000)

""" Restore the latest checkpoint and test the network """

# Restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

"""## Generate some images using the test set"""

# Run the trained model on a few examples from the test set
counter = 0
os.makedirs(OUTPUT_PATH/'generated_images/', exist_ok=True)
for inp, target in test_dataset.take(NUM_GENERATED_IMAGES):
  filename = OUTPUT_PATH/'generated_images/' + str(counter) + '.png'
  generate_images(generator, inp, target, filename)
  counter+=1
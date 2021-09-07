# import gdal
import tensorflow as tf
import tensorflow_io as tfio

image_file = '../Sentinel2/' + '2018_20m_b5678a1112.tif'
image = tf.io.read_file(image_file)
image = tfio.experimental.image.decode_tiff(image)
image = tf.cast(image, tf.float32)
print(image)

def load_tif_image(patch):
    gdal_header = gdal.Open(patch)
    img = gdal_header.ReadAsArray()
    img = np.moveaxis(img, 0, -1)
    print(type(img), img.shape)
    return img


# def load_tiff(image_file):
#   # Read and decode an image file to a uint8 tensor


#   # Split each image tensor into two tensors:
#   # - one with a real building facade image
#   # - one with an architecture label image 
#   w = tf.shape(image)[1]
#   w = w // 2
#   input_image = image[:, w:, :]
#   real_image = image[:, :w, :]

  # Convert both images to float32 tensors


#   return input_image, real_image
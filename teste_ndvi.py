import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def get_ndvi(s2_image):
    # s2_image bands:
    # 0 - blue - 10m 
    # 1 - green - 10m 
    # 2 - red - 10m ---------> b4
    # 3 - NIR - 10m ---------> b8
    # 4 - vegetation red edge 
    # 5 - vegetation red edge 
    # 6 - vegetation red edge 
    # 7 - vegetation red edge 
    # 8 - SWIR (1.610 um)
    # 9 - SWIR (2.190um)

    # NDVI = (B8-B4)/(B8+B4)
    nir = s2_image[:,:, 3].copy().astype(np.float32) * 0.5 + 0.5 # [-1,+1] -> [0, 1]
    red = s2_image[:,:, 2].copy().astype(np.float32) * 0.5 + 0.5 # [-1,+1] -> [0, 1]
    ndvi = np.where( (nir==0.) | (red ==0.), -1, np.where((nir+red)==0., 0, (nir-red)/(nir+red)))
    return ndvi


def histeq(im,nbr_bins=3):
   #get image histogram
   imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True)
   cdf = imhist.cumsum() #cumulative distribution function
   cdf = cdf / cdf[-1] #normalize
   #use linear interpolation of cdf to find new pixel values
   im2 = np.interp(im.flatten(),bins[:-1],cdf)
   return im2.reshape(im.shape), cdf


def to_nan(array, value=-10):
    array[array == value] = np.NaN
    return array


def plot_image(plot_list, columns, rows, title, filename=None, pad=1):
  fig = plt.figure()
  for i in range(0, columns*rows):
    fig.add_subplot(rows, columns, i + 1)
    plt.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
    plt.imshow(plot_list[i]  * 0.5 + 0.5)
  fig.tight_layout(pad=pad)
  if filename:
    fig.savefig(filename)
  plt.close(fig)


def discretize_forest_region(image, mask, num_bins=2, s=''):

    deforestation_mask = mask == 0 #new_deforestation_mask = mask == 0
    old_deforestation_mask = mask == 1
    #deforestation_mask = (new_deforestation_mask | old_deforestation_mask)

    new_image = np.zeros_like(image)
    new_image[deforestation_mask] = image[deforestation_mask]

    #cv2.imwrite('./image.png', im)

    # todo: equalize image (all bands)
    ndvi = get_ndvi(new_image) # where mask is 0, will have the value of 0
    #print('ndvi:', np.unique(ndvi))

    # plot
    #result = (ndvi.copy() * 0.5 + 0.5) * 127.5
    #cv2.imwrite('./ndvi.png', result)

    ndvi = ndvi.astype(np.float32)
    num_bins = num_bins - 1
    ndvi_eq = np.floor(ndvi*num_bins+0.5)/num_bins
    values = np.unique(ndvi_eq)
    #print('values:', values)

    # plot
    result = (ndvi_eq *0.5 + 0.5) * 127.5
    result[~deforestation_mask] = 0
    #cv2.imwrite('./ndvi_eq_ops_1.png', result)

    h, w, channels = new_image.shape
    #out_list = []
    
    # for each channel
    for channel in np.arange(channels):
        print('>> channel:', channel)
        # get channel
        current_channel = new_image[:,:,channel].copy()
        
        # for each ndvi equalized value
        for value in values:
            print('>> value:', value)
            # get ndvi bin mask
            current_bin_mask = (ndvi_eq == value)
            print('current_bin_mask:', np.count_nonzero(current_bin_mask))
            # get indices
            current_bin_idx = np.stack(np.nonzero(current_bin_mask), axis=-1)

            # mask pixels that are not from this bin
            current_bin_pixels = current_channel*current_bin_mask.astype('float32') 
            # mask forest pixels from mean
            current_bin_pixels[~deforestation_mask] = -10
            print(np.count_nonzero(current_bin_pixels == -10))
            current_bin_pixels = to_nan(current_bin_pixels)
            print(np.count_nonzero(np.isnan(current_bin_pixels)))
            # get mean
            current_mean = np.nanmean(current_bin_pixels)
            print('current_mean:', current_mean)
            # assign mean value to all positions
            current_channel[current_bin_mask] = current_mean
            print(np.count_nonzero(current_channel == current_mean))

        new_image[:,:,channel] = current_channel.copy()

    # combine 
    new_image[~deforestation_mask] = image[~deforestation_mask]
    

    chans = [0, 1, 2]
    image_vis = image.copy()
    image_vis = image_vis[:,:,chans]
    new_image_vis = new_image[:,:,chans]
    plot_list = [image_vis, new_image_vis]

    plot_image(plot_list, columns=2, rows=1, title=[], filename='./sample_'+str(s)+'.png')
    plt.close('all')
    #cv2.imwrite('./new_image_vis_'+str(s)+'.png', new_image_vis*127.5)
    return new_image


for i in np.random.randint(0,500,15):
    i = str(i)
    im = np.load('D:\\amandalucs\\Samples\\change_detection_true\\training_data\\imgs\\' + i + '.npy')
    h, w, c = im.shape
    image = im[:,:,c//2:]
    mask = np.load('D:\\amandalucs\\Samples\\change_detection_true\\training_data\\masks\\' + i + '.npy')

    image_eq = discretize_forest_region(image, mask, 3, i)

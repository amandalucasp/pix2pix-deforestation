import gdal
import skimage
import scipy.misc
import numpy as np
from PIL import Image
import sys, os, platform
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.util import view_as_windows
from sklearn.preprocessing import StandardScaler, MinMaxScaler

only_bgr_channels = True
only_segmentation_problem = True
patch_size = 256 # 128. mudei pra 256 pelo tamanho da entrada da pix2pix
stride = int(patch_size / 2)
minipatch_size = 32
mini_stride = int(minipatch_size/4)
min_percentage = 5 # minimum deforestation required per patch
root_path = './/' # dataset images path
output_path = '/' # dataset output path
lim_x = 17000 # 1000
lim_y = 9200 # 7000
CHANNELS = [1, 2, 6]

tiles_tr = [1,3,5,7,8,10,11,13,14,16,18,20,4,6,19]
tiles_val = [2, 9, 12] # [4,6,19]
tiles_ts = (list(set(np.arange(20)+1)-set(tiles_tr)-set(tiles_val)))

def retrieve_idx_percentage(reference, patches_idx_set, patch_size, pertentage = 5):
    # extract only patches with >= 2% of deforestation
    count = 0
    new_idx_patches = []
    reference_vec = reference.reshape(reference.shape[0]*reference.shape[1])
    for patchs_idx in patches_idx_set:
        patch_ref = reference_vec[patchs_idx]
        class1 = patch_ref[patch_ref==1]
        if len(class1) >= int((patch_size**2)*(pertentage/100)):
            count = count + 1
            new_idx_patches.append(patchs_idx)
    return np.asarray(new_idx_patches)


def extract_patches_(image, reference, patch_size=256, stride=128):

  border_size = patch_size
  image = np.pad(image,((border_size,border_size),(border_size,border_size),(0,0)), mode='reflect')
  transformed_ref = np.pad(reference,((border_size,border_size),(border_size,border_size)), mode='reflect')
  #
  # print("Image shape:",image.shape,"\n Reference shape:",transformed_ref.shape)

  image_list = []
  ref_list = []
  num_y = image.shape[0]
  num_x = image.shape[1]

  # counter = 0
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
      # counter+=1

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


def extract_patches(input_image, reference, patch_size, stride):
    window_shape = patch_size
    window_shape_array = (window_shape, window_shape, input_image.shape[2])
    window_shape_ref = (window_shape, window_shape)
    patches_array = np.array(view_as_windows(input_image, window_shape_array, step = stride))
    patches_ref = np.array(view_as_windows(reference, window_shape_ref, step = stride))
    num_row,num_col,p,row,col,depth = patches_array.shape
    patches_array = patches_array.reshape(num_row*num_col,row,col,depth)
    patches_ref = patches_ref.reshape(num_row*num_col,row,col)
    return patches_array, patches_ref


def patch_tiles(tiles, mask_amazon, image_array, image_ref, patch_size, stride):
    '''Extraction of image patches and labels '''
    patches_out = []
    label_out = []
    for num_tile in tiles:
        rows, cols = np.where(mask_amazon == num_tile)
        x1 = np.min(rows)
        y1 = np.min(cols)
        x2 = np.max(rows)
        y2 = np.max(cols)

        tile_img = image_array[x1:x2 + 1, y1:y2 + 1, :]
        tile_ref = image_ref[x1:x2 + 1, y1:y2 + 1]
        patches_img, patch_ref = extract_patches(tile_img, tile_ref, patch_size, stride)
        # print(patches_img.shape)
        # print(patch_ref.shape)
        patches_out.append(patches_img)
        label_out.append(patch_ref)

    patches_out = np.concatenate(patches_out)
    label_out = np.concatenate(label_out)
    # print(patches_out.shape)
    # print(label_out.shape)

    # exit()
    return patches_out, label_out


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


def load_tif_image(patch):
    gdal_header = gdal.Open(patch)
    img = gdal_header.ReadAsArray()
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


def normalization(image, norm_type=1):
    image_reshaped = image.reshape((image.shape[0] * image.shape[1]), image.shape[2])
    if (norm_type == 1):
        scaler = StandardScaler()
    if (norm_type == 2):
        scaler = MinMaxScaler(feature_range=(0, 1))
    if (norm_type == 3):
        scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(image_reshaped)
    image_normalized = scaler.fit_transform(image_reshaped)
    image_normalized1 = image_normalized.reshape(image.shape[0], image.shape[1], image.shape[2])
    return image_normalized1


def create_mask(size_rows, size_cols, grid_size=(6, 3)):
    num_tiles_rows = size_rows // grid_size[0]
    num_tiles_cols = size_cols // grid_size[1]
    print('Tiles size: ', num_tiles_rows, num_tiles_cols)
    patch = np.ones((num_tiles_rows, num_tiles_cols))
    mask = np.zeros((num_tiles_rows * grid_size[0], num_tiles_cols * grid_size[1]))
    count = 0
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            count = count + 1
            mask[num_tiles_rows * i:(num_tiles_rows * i + num_tiles_rows),
            num_tiles_cols * j:(num_tiles_cols * j + num_tiles_cols)] = patch * count
    # plt.imshow(mask)
    print('Mask size: ', mask.shape)
    return mask


################### READ TIF IMAGES


# Load images
if not only_segmentation_problem:
    sent2_2018_1 = load_tif_image(root_path + '2018_10m_b2348.tif').astype('float32')
    sent2_2018_2 = load_tif_image(root_path + '2018_20m_b5678a1112.tif').astype('float32')
    # Resize bands of 20m
    sent2_2018_2 = resize_image(sent2_2018_2.copy(), sent2_2018_1.shape[0], sent2_2018_1.shape[1])
    # print('sent2_2018_1.shape', sent2_2018_1.shape)
    # print('sent2_2018_2.shape', sent2_2018_2.shape)
    sent2_2018 = np.concatenate((sent2_2018_1, sent2_2018_2), axis=-1)
    del sent2_2018_1, sent2_2018_2

sent2_2019_1 = load_tif_image(root_path + '2019_10m_b2348.tif').astype('float32')
sent2_2019_2 = load_tif_image(root_path + '2019_20m_b5678a1112.tif').astype('float32')

# Resize bands of 20m
sent2_2019_2 = resize_image(sent2_2019_2.copy(), sent2_2019_1.shape[0], sent2_2019_1.shape[1])
# print('sent2_2019_1.shape', sent2_2019_1.shape)
# print('sent2_2019_2.shape', sent2_2019_2.shape)
sent2_2019 = np.concatenate((sent2_2019_1, sent2_2019_2), axis=-1)
del sent2_2019_1, sent2_2019_2

if only_bgr_channels:
    print('Using only NIR-G-B channels.')
    if not only_segmentation_problem:
        sent2_2018 = sent2_2018[:, :, CHANNELS]       
    sent2_2019 = sent2_2019[:, :, CHANNELS]

 
# Filter outliers
if not only_segmentation_problem:
    sent2_2018 = filter_outliers(sent2_2018.copy())
sent2_2019 = filter_outliers(sent2_2019.copy())

if only_segmentation_problem:
    print('Deforestation/Forest Segmentation.')
    image_stack = sent2_2019
    del sent2_2019
else:
    image_stack = np.concatenate((sent2_2018, sent2_2019), axis=-1)
    del sent2_2018, sent2_2019

print('Image stack:', image_stack.shape)

final_mask = np.load(root_path+'final_mask_label.npy').astype('float32')
# 0: forest
# 1: new deforestation
# 2: old deforestation
print('final_mask unique values:', np.unique(final_mask), len(final_mask[final_mask == 1]))
# change into only forest and deforestation:
if only_segmentation_problem:
    final_mask[final_mask == 2] = 1
    print('final_mask unique values:', np.unique(final_mask), len(final_mask[final_mask == 1]))

final_mask = final_mask[:lim_x, :lim_y]
image_stack = image_stack[:lim_x, :lim_y, :]
h_, w_, channels = image_stack.shape
print('image stack size: ', image_stack.shape)

# Normalization
type_norm = 1
print(np.min(image_stack), np.max(image_stack))
image_array = normalization(image_stack.copy(), type_norm)
print(np.min(image_array), np.max(image_array))
del image_stack

# os.makedirs(output_path, exist_ok=True)
scipy.misc.imsave('image_array.png', image_array)
scipy.misc.imsave('final_mask.png', final_mask)

################### EXTRACT PATCHES

# Print pertengate of each class (whole image)
print('Total no-deforestaion class is {}'.format(len(final_mask[final_mask==0])))
print('Total deforestaion class is {}'.format(len(final_mask[final_mask==1])))
print('Total past deforestaion class is {}'.format(len(final_mask[final_mask==1])))
print('Percentage of deforestaion class is {:.2f}'.format((len(final_mask[final_mask==1])*100)/len(final_mask[final_mask==0])))

print("EXTRACTING PATCHES")

trn_out_path = output_path + '/training_data'
val_out_path = output_path + '/validation_data'
tst_out_path = output_path + '/testing_data'

os.makedirs(output_path, exist_ok=True)
os.makedirs(trn_out_path + '/imgs', exist_ok=True)
os.makedirs(trn_out_path + '/masks', exist_ok=True)
os.makedirs(val_out_path + '/imgs', exist_ok=True)
os.makedirs(val_out_path + '/masks', exist_ok=True)
os.makedirs(tst_out_path + '/imgs', exist_ok=True)
os.makedirs(tst_out_path + '/masks', exist_ok=True)

# Create tile mask
mask_tiles = create_mask(final_mask.shape[0], final_mask.shape[1], grid_size=(5, 4))
image_array = image_array[:mask_tiles.shape[0], :mask_tiles.shape[1],:]
final_mask = final_mask[:mask_tiles.shape[0], :mask_tiles.shape[1]]

# Define tiles for training, validation, and test sets
print('[*]Tiles for Training:', tiles_tr)
print('[*]Tiles for Validation:', tiles_val)
print('[*]Tiles for Testing:', tiles_ts)

mask_tr_val = np.zeros((mask_tiles.shape)).astype('float32')
# Training and validation mask
for tr_ in tiles_tr:
    mask_tr_val[mask_tiles == tr_] = 1

for val_ in tiles_val:
    mask_tr_val[mask_tiles == val_] = 2

mask_amazon_ts = np.zeros((mask_tiles.shape)).astype('float32')
for ts_ in tiles_ts:
    mask_amazon_ts[mask_tiles == ts_] = 1


def discard_patches_by_percentage(patches, patches_ref, percentage = 2):
    patches_ = []
    patches_ref_ = []
    for i in range(len(patches)):
        patch = patches[i]
        patch_ref = patches_ref[i]
        class1 = patch_ref[patch_ref == 1]
        per = int((patch_size ** 2) * (percentage / 100))
        # print(len(class1), per)
        if len(class1) >= per:
            patches_.append(patch)
            patches_ref_.append(patch_ref)
    patches_ = np.array(patches_)
    patches_ref_ = np.array(patches_ref_)
    return patches_, patches_ref_


print('[*]Patch size:', patch_size)
print('[*]Stride:', stride)
patches_trn, patches_trn_ref = patch_tiles(tiles_tr, mask_tiles, image_array, final_mask, patch_size, stride)
patches_val, patches_val_ref = patch_tiles(tiles_val, mask_tiles, image_array, final_mask, patch_size, stride)
patches_tst, patches_tst_ref = patch_tiles(tiles_ts, mask_tiles, image_array, final_mask, patch_size, stride)
patches_trn, patches_trn_ref = discard_patches_by_percentage(patches_trn, patches_trn_ref, min_percentage)
patches_val, patches_val_ref = discard_patches_by_percentage(patches_val, patches_val_ref, min_percentage)
patches_tst, patches_tst_ref = discard_patches_by_percentage(patches_tst, patches_tst_ref, min_percentage)

del image_array, final_mask

print('Filtered the patches by a minimum of', str(min_percentage), '% of deforestation.')
print('[*] Training patches:', patches_trn.shape)
print('[*] Validation patches:', patches_val.shape)
print('[*] Testing patches:', patches_tst.shape)

# saving patches
counter = 0
print('Saving training patches...')
for i in range(patches_trn.shape[0]):
    np.save(trn_out_path + '/imgs/' + str(i) + '.npy', patches_trn[i])
    np.save(trn_out_path + '/masks/' + str(i) + '.npy', patches_trn_ref[i])
    counter += 1

counter = 0
print('Saving validation patches...')
for i in range(patches_val.shape[0]):
    np.save(val_out_path + '/imgs/' + str(i) + '.npy', patches_val[i])
    np.save(val_out_path + '/masks/' + str(i) + '.npy', patches_val_ref[i])
    counter += 1

counter = 0
print('Saving testing patches...')
for i in range(patches_tst.shape[0]):
    np.save(tst_out_path + '/imgs/' + str(i) + '.npy', patches_tst[i])
    np.save(tst_out_path + '/masks/' + str(i) + '.npy', patches_tst_ref[i])
    counter += 1

del patches_tst, patches_tst_ref

################### EXTRACT MINIPATCHES (FOREST AND DEFORESTATION)

print("EXTRACTING MINIPATCHES")
max_minipatches = None

os.makedirs(trn_out_path + '/texture_class_0', exist_ok=True)
os.makedirs(trn_out_path + '/texture_class_1', exist_ok=True)
os.makedirs(val_out_path + '/texture_class_0', exist_ok=True)
os.makedirs(val_out_path + '/texture_class_1', exist_ok=True)
# os.makedirs(tst_out_path + '/texture_class_0', exist_ok=True)
# os.makedirs(tst_out_path + '/texture_class_1', exist_ok=True)

os.makedirs(trn_out_path + '/texture_class_0_debug', exist_ok=True)
os.makedirs(trn_out_path + '/texture_class_1_debug', exist_ok=True)
os.makedirs(val_out_path + '/texture_class_0_debug', exist_ok=True)
os.makedirs(val_out_path + '/texture_class_1_debug', exist_ok=True)
# os.makedirs(tst_out_path + '/texture_class_0_debug', exist_ok=True)
# os.makedirs(tst_out_path + '/texture_class_1_debug', exist_ok=True)


def check_patch_class(patch):
    total_pixels_patch = patch.shape[0]*patch.shape[1]
    patch_class = patch[0][0]
    if int(patch_class) == 2:
        return None
    pixels_class_count = np.count_nonzero(patch == patch_class)
    # print('total_pixels_patch:', total_pixels_patch)
    # print('pixels_class_count:', pixels_class_count)
    # print('patch_class:', patch_class)
    if pixels_class_count == total_pixels_patch:
        return int(patch_class)
    else:
        return None


def extract_minipatches_from_patch_old(patch, patch_ref, minipatch_size, index):
    # only works with patch_size = 2*stride
    stride = int(minipatch_size/2)
    num_y = patch.shape[0]
    num_x = patch.shape[1]
    found_patch = [0, 0]
    patches = [None, None]
    patches_ref = [None, None]

    for posicao_y in range(stride, (num_y - (stride)), stride):
        for posicao_x in range(stride, (num_x - (stride)), stride):
            y1 = posicao_y - stride
            y2 = posicao_y + stride
            x1 = posicao_x - stride
            x2 = posicao_x + stride
            aux = patch[y1:y2, x1:x2]
            aux_ref = patch_ref[y1:y2, x1:x2]
            # aux_vis_1 = normalization(aux[:,:,:3], 2)
            # aux_vis_2 = normalization(aux[:, :, 10:13], 2)
            patch_class = check_patch_class(aux_ref)
            if patch_class is not None and (found_patch[patch_class] == 0):
                # print('index:', str(index), 'patch_class:', str(patch_class), found_patch)
                # np.save(output_path + '/texture_class_' + str(patch_class) + '/' + str(index) + '.npy', aux)
                # scipy.misc.imsave(output_path + '/texture_class_' + str(patch_class) + '_debug/' + str(index) + '_1.jpg', aux_vis_1)
                # scipy.misc.imsave(output_path + '/texture_class_' + str(patch_class) + '_debug/' + str(index) + '_2.jpg', aux_vis_2)
                # np.save(output_path + str(int(patch_class)) + '/references/patch_' + str(index) + '_' + str(counter[int(patch_class)]) + '.npy', aux_ref)
                patches[patch_class] = aux
                patches_ref[patch_class] = aux_ref
                # images_list.append(aux)
                # ref_list.append(aux_ref)
                found_patch[patch_class] = 1
                if found_patch == list([1, 1]):
                    # print('->found_patch:', found_patch)
                    return patches, patches_ref, found_patch

    return patches, patches_ref, found_patch


def get_only_texture_minipatches(all_patches_array, all_patches_ref, index, out_path):
    patches = [None, None]
    patches_ref = [None, None]
    found_patch = [0, 0]
    for i in range(len(all_patches_array)):
        patch_class = check_patch_class(all_patches_ref[i])
        if patch_class is not None and (found_patch[patch_class] == 0):
            # print('patch_class:', str(patch_class), found_patch)
            found_patch[patch_class] = 1
            patches[patch_class] = all_patches_array[i]
            patches_ref[patch_class] = all_patches_ref[i]
            # print(patches[patch_class].shape)
            aux_vis_1 = patches[patch_class]
            # aux_vis_2 = patches[patch_class]
            aux_vis_1 = normalization(aux_vis_1, 2)
            # aux_vis_2 = normalization(aux_vis_2[:, :, 3:], 2)
            # print(aux_vis_1.shape, aux_vis_2.shape)
            scipy.misc.imsave(out_path + '/texture_class_' + str(patch_class) + '_debug/' + str(index) + '_1.jpg', aux_vis_1)
            # scipy.misc.imsave(out_path + '/texture_class_' + str(patch_class) + '_debug/' + str(index) + '_2.jpg', aux_vis_2)
            if found_patch == list([1, 1]):
                return patches, patches_ref, found_patch
    return patches, patches_ref, found_patch


def extract_minipatches_from_patch(input_image, reference, minipatch_size, mini_stride, index, out_path):
    window_shape = minipatch_size
    window_shape_array = (window_shape, window_shape, input_image.shape[2])
    window_shape_ref = (window_shape, window_shape)
    # extract all possible patches from input patch
    patches_array = np.array(view_as_windows(input_image, window_shape_array, step = mini_stride))
    patches_ref = np.array(view_as_windows(reference, window_shape_ref, step = mini_stride))
    num_row,num_col,p,row,col,depth = patches_array.shape
    # print(num_row,num_col,p,row,col,depth)
    patches_array = patches_array.reshape(num_row*num_col,row,col,depth)
    patches_ref = patches_ref.reshape(num_row*num_col,row,col)
    # get only the one-class-only minipatches
    patches, patches_ref, found_patch = get_only_texture_minipatches(patches_array, patches_ref, index, out_path)
    # return minipatches
    return patches, patches_ref, found_patch

# training
counter = 0
counter_ = 0
print('[*] Saving training minipatches.')
for idx in range(patches_trn.shape[0]):
    # print('idx:', idx)
    patches, patches_ref, found_patch = extract_minipatches_from_patch(patches_trn[idx], patches_trn_ref[idx],
    minipatch_size, mini_stride, idx, trn_out_path)
    if found_patch == list([1, 1]): # save only patches in pairs
        np.save(trn_out_path + '/texture_class_0/' + str(idx) + '.npy', patches[0])
        np.save(trn_out_path + '/texture_class_1/' + str(idx) + '.npy', patches[1])
        counter_ +=1
    counter+=1
print('Training minipatches:', counter, counter_)

# validation
counter = 0
counter_ = 0
print('[*] Saving validation minipatches.')
for idx in range(patches_val.shape[0]):
    patches, patches_ref, found_patch = extract_minipatches_from_patch(patches_val[idx], patches_val_ref[idx], 
    minipatch_size, mini_stride, idx, val_out_path)
    if found_patch == list([1, 1]): # save only patches in pairs
        np.save(val_out_path + '/texture_class_0/' + str(idx) + '.npy', patches[0])
        np.save(val_out_path + '/texture_class_1/' + str(idx) + '.npy', patches[1])
        counter_ +=1
    counter+=1
print('Training minipatches:', counter, counter_)

print('[*] Preprocessing done.')

# for item in range(patches.shape[0]):
#
#     # deforestation_window = []
#     # forest_window = []
#
#     minimum_patch = 25
#     # set minimum number of pixels for each class in order to extract patch...
#
#     inds = np.asarray(np.column_stack(np.ma.where(patches_ref[item] > 0.)))
#     print(inds)
#
#     plt.figure()
#
#
#     # for i in range(patches.shape[0]):
#     #     for j in range(patches.shape[1]):
#     #         if patches_ref[item][i][j] == 0:
#     #             forest_window = patches_ref[]
#
#     # where is 0, get forest minipatch
#     # where is 1, get deforest minipatch
#     # add minipatches to list
#     counter+=1


# exit()
# for file in dir_list:
#     imgs2018 = []
#     if file.endswith('.tif') and file.find('2018')!=-1:
#     # read images from 2018 and reference


    # elif file.endswith('.jpg'):
    #     if os.path.isfile(path_mask + '/' + file.replace('.jpg','_segmentation.png')):
    #         count = count + 1
    #         print(count)
    #
    #         img_mask = ndimage.imread(path_mask + '/' + file.replace('.jpg','_segmentation.png'))
    #
    #         inds = np.asarray(np.column_stack(np.ma.where(img_mask > 0.)))
    #
    #         # print 'indexes: '
    #
    #         row_ = int(np.mean(inds[:, 0]))
    #         col_ = int(np.mean(inds[:, 1]))
    #
    #         if np.sum(img_mask[row_-texture_window/2:row_+texture_window/2,col_-texture_window/2:col_+texture_window/2]) > 0.95*texture_window*texture_window:
    #             img = ndimage.imread(path_image + '/' + file)
    #             lesion_window = img[row_ - texture_window / 2:row_ + texture_window / 2,
    #                             col_ - texture_window / 2:col_ + texture_window / 2,:]
    #             skin_window = img[row_ - texture_window / 2:row_ + texture_window / 2,
    #                             50:50 + texture_window,:]
    #
    #             img = ndimage.zoom(img,
    #                                [float(1024.) / img_mask.shape[0],
    #                                 float(1024.) / img_mask.shape[1], 1])
    #
    #             img_mask = ndimage.zoom(img_mask,
    #                                     [float(1024.) / img_mask.shape[0],
    #                                      float(1024.) / img_mask.shape[1]])
    #
    #             np.save(output_path + '/masks/' + file.split('/')[-1].replace('.jpg', '.npy'), img_mask)
    #             np.save(output_path + '/texture_lesion/' + file.split('/')[-1].replace('.jpg', '.npy'), lesion_window)
    #             np.save(output_path + '/texture_skin/' + file.split('/')[-1].replace('.jpg', '.npy'), skin_window)
    #             np.save(output_path + '/imgs/' + file.split('/')[-1].replace('.jpg', '.npy'), img)
    #         else:
    #             continue



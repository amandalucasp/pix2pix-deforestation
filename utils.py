import gdal
import skimage
import imageio
import cv2
import numpy as np
from PIL import Image
import sys, os, platform
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.util import view_as_windows
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def process_masks(rej_pairs_ref, config):
    if config['synthetic_input_mode'] == 0:
        return rej_pairs_ref


def save_image_pairs(patches_list, patches_ref_list, pairs_path, config, synthetic_input_pairs=False):
    os.makedirs(pairs_path + '/pairs', exist_ok=True)
    counter = 0
    h, w, c = patches_list[0].shape

    if config['change_detection']:
        if not synthetic_input_pairs:
            # T1 + T2 + 3-classes mask
            for i in range(patches_list.shape[0]):
                # combined will be: T1 - T2 - mask
                combined = np.zeros(shape=(h,w*3,c//2))
                combined[:,:w,:] = patches_list[i][:,:,:c//2]
                combined[:,w:w*2,:] = patches_list[i][:,:,c//2:]
                converted = cv2.cvtColor(patches_ref_list[i].copy(), cv2.COLOR_GRAY2BGR) # verificar se precisa msm, acho que n faz diferenca 
                if config['type_norm'] == 0:
                    # 0: forest, 1: new deforestation, 2: old deforestation
                    converted[converted == 1] = 255/2
                    converted[converted == 2] = 255
                combined[:,w*2:,:] = converted
                np.save(pairs_path + '/pairs/' + str(i) + '.npy', combined)
                if config['debug_mode']:
                    cv2.imwrite(pairs_path + '/pairs/' + str(i) + '_debug.jpg', combined)
                counter += 1
        else:
            # T1 + blank + 3-classes mask
            for i in range(patches_list.shape[0]):
                # combined will be: T1 - blank - mask
                combined = np.zeros(shape=(h,w*3,c//2))
                combined[:,:w,:] = patches_list[i][:,:,:c//2]
                converted = cv2.cvtColor(patches_ref_list[i].copy(), cv2.COLOR_GRAY2BGR) # verificar se precisa msm, acho que n faz diferenca 
                if config['type_norm'] == 0:
                    # 0: forest, 1: new deforestation, 2: old deforestation
                    converted[converted == 1] = 255/2
                    converted[converted == 2] = 255
                combined[:,w*2:,:] = converted
                np.save(pairs_path + '/pairs/' + str(i) + '.npy', combined)
                if config['debug_mode']:
                    cv2.imwrite(pairs_path + '/pairs/' + str(i) + '_debug.jpg', combined)
                counter += 1
    else:
        # T2 only
        for i in range(patches_list.shape[0]):
            combined = np.zeros(shape=(h,w*2,c))
            combined[:,:w,:] = patches_list[i]
            converted = cv2.cvtColor(patches_ref_list[i].copy(), cv2.COLOR_GRAY2BGR) # verificar se precisa msm, acho que n faz diferenca 
            if config['type_norm'] == 0: # image is [0,255]
                if config['two_classes_problem']:
                    # binary mask
                    converted = 255*converted
                else:
                    # 3-classes mask
                    converted[converted == 1] = 255/2
                    converted[converted == 2] = 255
            combined[:,w:,:] = converted
            np.save(pairs_path + '/pairs/' + str(i) + '.npy', combined)
            if config['debug_mode']:
                cv2.imwrite(pairs_path + '/pairs/' + str(i) + '_debug.jpg', combined)
            counter += 1


def get_dataset(config):
    """
    input: config - a dict file with processing parameters
    output: image_stack, final_mask
        - two_classes_problem = True:
            image_stack: only T2; final_mask: binary mask; 
        - two_classes_problem = False:
            image_stack: T1 + T2 (concat); final_mask: 3-classes mask;  
    """

    print('[*]Loading dataset')
    if config['change_detection']:
        # LOAD IMAGE T1
        sent2_2018_1 = load_tif_image(config['root_path'] + '2018_10m_b2348.tif').astype('float32')
        sent2_2018_2 = load_tif_image(config['root_path'] + '2018_20m_b5678a1112.tif').astype('float32')
        # Resize bands of 20m
        sent2_2018_2 = resize_image(sent2_2018_2.copy(), sent2_2018_1.shape[0], sent2_2018_1.shape[1])
        sent2_2018 = np.concatenate((sent2_2018_1, sent2_2018_2), axis=-1)
        del sent2_2018_1, sent2_2018_2

    # LOAD IMAGE T2
    sent2_2019_1 = load_tif_image(config['root_path'] + '2019_10m_b2348.tif').astype('float32')
    sent2_2019_2 = load_tif_image(config['root_path'] + '2019_20m_b5678a1112.tif').astype('float32')
    # Resize bands of 20m
    sent2_2019_2 = resize_image(sent2_2019_2.copy(), sent2_2019_1.shape[0], sent2_2019_1.shape[1])
    sent2_2019 = np.concatenate((sent2_2019_1, sent2_2019_2), axis=-1)
    del sent2_2019_1, sent2_2019_2
    
    sent2_2019 = sent2_2019[:, :, config['channels']]
    sent2_2019 = filter_outliers(sent2_2019.copy())
    image_stack = sent2_2019
    del sent2_2019

    if config['change_detection']:
        sent2_2018 = sent2_2018[:, :, config['channels']]  
        sent2_2018 = filter_outliers(sent2_2018.copy()) 
        image_stack = np.concatenate((sent2_2018, image_stack), axis=-1)
        del sent2_2018 #, sent2_2019

    final_mask = np.load(config['root_path']+'final_mask_label.npy').astype('float32')
    # 0: forest, 1: new deforestation, 2: old deforestation
    # change into only forest and deforestation:
    if config['two_classes_problem']:
        final_mask[final_mask == 2] = 1
    print('final_mask unique values:', np.unique(final_mask), len(final_mask[final_mask == 1]))

    final_mask = final_mask[:config['lim_x'], :config['lim_y']]
    image_stack = image_stack[:config['lim_x'], :config['lim_y'], :]
    h_, w_, channels = image_stack.shape
    print('image_stack size: ', image_stack.shape)

    return image_stack, final_mask


def check_patch_class(patch):
    total_pixels_patch = patch.shape[0]*patch.shape[1]
    patch_class = patch[0][0]
    if int(patch_class) == 2:
        return None
    pixels_class_count = np.count_nonzero(patch == patch_class)
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
            imageio.imwrite(out_path + '/texture_class_' + str(patch_class) + '_debug/' + str(index) + '_1.jpg', aux_vis_1)
            # scipy.misc.imsave(out_path + '/texture_class_' + str(patch_class) + '_debug/' + str(index) + '_2.jpg', aux_vis_2)
            if found_patch == list([1, 1]):
                return patches, patches_ref, found_patch
    return patches, patches_ref, found_patch


def write_patches_to_disk(patches, patches_ref, out_path):
    counter = 0
    for i in range(patches.shape[0]):
        np.save(out_path + '/imgs/' + str(i) + '.npy', patches[i])
        np.save(out_path + '/masks/' + str(i) + '.npy', patches_ref[i])
        counter += 1


def save_minipatches(patches_list, patches_ref_list, out_path, config):
    mini_stride = int(config['minipatch_size']/4)
    os.makedirs(out_path + '/texture_class_0', exist_ok=True)
    os.makedirs(out_path + '/texture_class_1', exist_ok=True)
    counter = 0
    counter_ = 0
    for idx in range(patches_list.shape[0]):
        patches, patches_ref, found_patch = extract_minipatches_from_patch(patches_list[idx], patches_ref_list[idx],
        config['minipatch_size'], mini_stride, idx, out_path)
        if found_patch == list([1, 1]): # save only patches in pairs
            np.save(out_path + '/texture_class_0/' + str(idx) + '.npy', patches[0])
            np.save(out_path + '/texture_class_1/' + str(idx) + '.npy', patches[1])
            counter_ +=1
        counter+=1


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


def discard_patches_by_percentage(patches, patches_ref, config, new_deforestation_pixel_value = 1):
    # 0: forest, 1: new deforestation, 2: old deforestation
    patch_size = config['patch_size']
    percentage = config['min_percentage']
    patches_ = []
    patches_ref_ = []
    rejected_patches_ = []
    rejected_patches_ref = []
    rejected_pixels_count = []
    for i in range(len(patches)):
        patch = patches[i]
        patch_ref = patches_ref[i]
        class1 = patch_ref[patch_ref == new_deforestation_pixel_value]
        per = int((patch_size ** 2) * (percentage / 100))
        # print(len(class1), per)
        if len(class1) >= per:
            patches_.append(i)
            patches_ref_.append(i)
        else:
            rejected_patches_.append(i)
            rejected_patches_ref.append(i) 
            rejected_pixels_count.append(len(class1))
    return patches[patches_], patches_ref[patches_ref_], patches[rejected_patches_], patches_ref[rejected_patches_ref], rejected_pixels_count


def discard_patches_by_percentage_deprecated(patches, patches_ref, config, new_deforestation_pixel_value = 1):
    # 0: forest, 1: new deforestation, 2: old deforestation
    patch_size = config['patch_size']
    percentage = config['min_percentage']
    patches_ = []
    patches_ref_ = []
    # rejected_patches_ = []
    # rejected_patches_ref = []
    for i in range(len(patches)):
        patch = patches[i]
        patch_ref = patches_ref[i]
        class1 = patch_ref[patch_ref == new_deforestation_pixel_value]
        per = int((patch_size ** 2) * (percentage / 100))
        # print(len(class1), per)
        if len(class1) >= per:
            patches_.append(patch)
            patches_ref_.append(patch_ref)
        # else:
            # rejected_patches_.append(patch)
            # rejected_patches_ref.append(patch_ref) 
    # print(type(patches), type(patches_))
    print('a')
    rejected_patches_ = np.array(list(set(patches.tolist())-set(patches_)))
    print('b')
    rejected_patches_ref = np.array(list(set(patches_ref.tolist())-set(patches_ref_)))
    print('c')
    patches_ = np.array(patches_)
    patches_ref_ = np.array(patches_ref_)

    # rejected_patches_ref = np.array(rejected_patches_ref)
    return patches_, patches_ref_, rejected_patches_, rejected_patches_ref


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
    counter = 0
    for num_tile in tiles:
        counter+=1
        # print(num_tile, str(counter))
        rows, cols = np.where(mask_amazon == num_tile)
        x1 = np.min(rows)
        y1 = np.min(cols)
        x2 = np.max(rows)
        y2 = np.max(cols)
        tile_img = image_array[x1:x2 + 1, y1:y2 + 1, :]
        tile_ref = image_ref[x1:x2 + 1, y1:y2 + 1]
        # print(tile_img.shape, tile_ref.shape)
        patches_img, patch_ref = extract_patches(tile_img, tile_ref, patch_size, stride)
        # print(patches_img.shape, patch_ref.shape)
        patches_out.append(patches_img)
        label_out.append(patch_ref)
        # print(len(patches_out), len(label_out))

    patches_out = np.concatenate(patches_out)
    label_out = np.concatenate(label_out)
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
    if (norm_type == 0):
        scaler = MinMaxScaler(feature_range=(0, 255))
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
    # print('Tiles size: ', num_tiles_rows, num_tiles_cols)
    patch = np.ones((num_tiles_rows, num_tiles_cols))
    mask = np.zeros((num_tiles_rows * grid_size[0], num_tiles_cols * grid_size[1]))
    count = 0
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            count = count + 1
            mask[num_tiles_rows * i:(num_tiles_rows * i + num_tiles_rows),
            num_tiles_cols * j:(num_tiles_cols * j + num_tiles_cols)] = patch * count
    # plt.imshow(mask)
    # print('Mask size: ', mask.shape)
    return mask

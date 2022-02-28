import os
from osgeo import gdal
import time
import skimage
import imageio
import cv2
import random
import numpy as np
from PIL import Image
import sys, os, platform
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.util import view_as_windows
from sklearn.preprocessing import StandardScaler, MinMaxScaler

np.random.seed(0)

def load_npy_files(files_list):
    npys_imgs = []
    npys_refs = []
    i = 0
    for img_file in files_list:
        start = time.time()
        npys_imgs.append(np.load(img_file))
        elapsed = time.time() - start
        print(os.path.basename(img_file), elapsed)

        start = time.time()
        ref_file = img_file.replace('_img', '_ref')
        npys_refs.append(np.load(ref_file))
        elapsed = time.time() - start
        print(os.path.basename(ref_file), elapsed)
        
    return np.concatenate((npys_imgs), axis=0), np.concatenate((npys_refs), axis=0)


def classify_masks(rej_pairs_ref):
    no_deforestation = []
    new_deforest = []
    old_deforest = []
    only_deforest = []
    all_classes = []
    only_old_deforest = []

    for i in range(len(rej_pairs_ref)):
        current_mask = rej_pairs_ref[i]
        unique = np.unique(current_mask, return_counts=False)
        if np.array_equal(unique, [0.]):
            # print('no_deforestation')
            no_deforestation.append(i)
        elif np.array_equal(unique, [2.]):
            # print('only_old_deforest')
            only_old_deforest.append(i)
        elif np.array_equal(unique, [0., 1.]):
            # print('new_deforest')
            new_deforest.append(i)
        elif np.array_equal(unique, [0., 2.]): 
            # print('old_deforest')
            old_deforest.append(i)
        elif np.array_equal(unique, [1., 2.]):
            # print('only_deforest')
            only_deforest.append(i)
        else:
            # print('all_classes')
            all_classes.append(i)

    print('[*] Total rejected patches:', len(rej_pairs_ref))
    print('[*] Patches with no deforestation:', len(no_deforestation))
    print('[*] Patches with only OLD deforestation:', len(only_old_deforest))
    print('[*] Patches with new deforestation and forest:', len(new_deforest))
    print('[*] Patches with old deforestation and forest:', len(old_deforest))
    print('[*] Patches with only new+old deforestation:', len(only_deforest))
    print('[*] Patches with new, old deforestation and forest:', len(all_classes))

    return no_deforestation, new_deforest, old_deforest, only_deforest, all_classes, only_old_deforest


def process_masks(rej_pairs, rej_pairs_ref, config):
    '''
    faz o processamento das mascaras que serao passadas como entrada (junto com T1) para o gerador treinado.
    rej_pairs_ref: mascaras dos pares que foram rejeitados durante o pre-processamento
    '''
    if config['synthetic_input_mode'] == 0:
        # faz nada, retorna os pares (img+mask) originais
        return rej_pairs, rej_pairs_ref

    # position arrays
    no_deforestation, new_deforest, old_deforest, only_deforest, all_classes, only_old_deforest = classify_masks(rej_pairs_ref)

    if config['synthetic_input_mode'] == 1:
        # ideia 1 - combinando patches que nao tem desmatamento com patches de desmatamento (mudando passado ---> novo)

        # usar apenas os patches que nao tem nada de desmatamento 
        # + com mascaras so de desmatamento novo (pode pegar mascara com dois desmatamentos e tirar o antigo)
        final_imgs = rej_pairs[no_deforestation]

        if len(new_deforest) != 0:
            # existem patches com desmatamento novo + floresta 
            final_refs = np.random.choice(new_deforest, len(final_imgs), replace=False)
        else:
            # nao existem patches com desmatamento novo + floresta, modifico patches de desmatamento antigo
            selected_patches = rej_pairs_ref[old_deforest]
            selected_patches[selected_patches == 2.] = 1 # desmatamento antigo ---> desmatamento novo
            selected_pos = np.random.choice(len(selected_patches), len(final_imgs), replace=False)
            final_refs = selected_patches[selected_pos]

    if config['synthetic_input_mode'] == 2:
        # ideia 2 pt 1 - adicionando desmatamento novo em mascaras com desmatamento passado
        # Selecionar patches apenas com floresta e desmatamento passado (Só 2 classes, sem desmatamento novo)
        if len(old_deforest) > config['max_input_samples']:
            old_deforest = random.sample(old_deforest, config['max_input_samples'])
        final_imgs = rej_pairs[old_deforest]
        selected_patches = rej_pairs_ref[old_deforest]
        # Na máscara: fazer dilate nas regiões de desmatamento passado para criar uma
        # camada ao redor de desmatamento novo (e chamar essa região ao redor gerada de desmatamento novo na máscara).
        final_refs = dilate_masks(selected_patches, config)

    if config['synthetic_input_mode'] == 3: # depois juntar com a de cima, 2
        # ideia 2 pt 2 - adicionando mais desmatamento novo em mascaras com baixa % desmatamento novo
        if len(all_classes) > config['max_input_samples']:
            all_classes = random.sample(all_classes, config['max_input_samples'])
        final_imgs = rej_pairs[all_classes]
        selected_patches = rej_pairs_ref[all_classes]
        final_refs = dilate_masks(selected_patches, config)
    
    #if config['synthetic_input_mode'] == 4:
        # Gerando mascaras inéditas a partir de colagens de regiões de desmatamento novo e antigo

    return final_imgs, final_refs


def dilate_masks(masks_list, config):
    dilated_masks = []
    i = 0
    for img_mask in masks_list:
        dilation = dilate_mask(img_mask, config)
        dilated_masks.append(dilation)
        print(i)
        i+=1
    return dilated_masks


def dilate_mask(img_mask, config):
    kernel = np.ones((5,5), np.uint8)
    if config['synthetic_input_mode'] == 2:  
        per_class_1 = 0. #calculate_percentage_of_class(img_mask)
        nb_interations = 1
        while per_class_1 <= config['goal_percentage']:
            dilation = cv2.dilate(img_mask, kernel, iterations = nb_interations) # borderValue deixou o processo mais lento
            diff = dilation - img_mask 
            dilation[diff == 2.] = 1.
            per_class_1 = calculate_percentage_of_class(dilation)
            nb_interations += 1
        print(nb_interations - 1, per_class_1)
        return dilation

    if config['synthetic_input_mode'] == 3:
        per_class_1 = 0. #calculate_percentage_of_class(img_mask)
        nb_interations = 1
        while per_class_1 < config['goal_percentage']:
            print('nb_interations:', nb_interations)
            dilation = cv2.dilate(img_mask, kernel, iterations = nb_interations) # borderValue deixou o processo mais lento
            soma = dilation + img_mask
            # operacao de soma: 
            # 0 -> 0: soma = 0 OK
            # 0 -> 1: soma = 1 OK
            # 0 -> 2: soma = 2 ---> 1 (nao posso add desmatamento velho)
            # 1 -> 0: nao ocorre
            # 1 -> 1: soma = 2 ---> 1
            # 1 -> 2: soma = 3 ---> 1 (nao posso add desmatamento velho)
            # 2 -> 0: nao ocorre
            # 2 -> 1: soma = 3 ---> 1* -> 2
            # 2 -> 2: soma = 4 ---> 2
            dilation[soma == 2] = 1 
            dilation[soma == 3] = 1
            dilation[img_mask == 2] = 2 # onde era desmatamento velho deve continuar o sendo
            per_class_1 = calculate_percentage_of_class(dilation)
            nb_interations += 1
        print(nb_interations - 1, per_class_1)      
        return dilation

   
def calculate_percentage_of_class(img_mask):
    per_class_1 = 0.
    unique, counts = np.unique(img_mask, return_counts=True)
    if 1 in unique:
        per_class_1 = counts[1]/np.sum(counts)
        # print('>>>', unique, counts[1], np.sum(counts), per_class_1)
    return per_class_1*100


def apply_mask(img, mask):
    # inverting reference
    current_def = (np.logical_not(mask.copy()))*1
    # number of channels
    channels = img.shape[-1]
    current_def_matrix = np.repeat(np.expand_dims(current_def, axis = -1), channels, axis=-1)
    # multiplying the image with the current deforestation mask
    masked_img = img * current_def_matrix
    return masked_img


def save_image_pairs(patches_list, patches_ref_list, pairs_path, config, synthetic_input_pairs=False):
    os.makedirs(pairs_path + '/pairs', exist_ok=True)
    counter = 0

    if len(patches_list) == 0:
        print('Empty list of image pairs')
        return

    h, w, c = patches_list[0].shape

    for i in range(patches_list.shape[0]):
        
        # combined will be: T1 - T2 - mask - masked T2 
        combined = np.zeros(shape=(h,w*4,c//2))

        t1_img = patches_list[i][:,:,:c//2] 
        t2_img = patches_list[i][:,:,c//2:]

        # mask
        mask = np.expand_dims(patches_ref_list[i].copy(), axis=-1) # 128x128 -> 128x128x1
        # replicando os canais da mascara ate atingir o numero de canais de t1 e t2 
        while mask.shape[2] != c//2:
            mask = np.concatenate((mask, np.expand_dims(patches_ref_list[i].copy(),axis=-1)), axis=-1)
        # apply mask to T2 
        masked_t2 = apply_mask(t2_img, mask)

        combined[:,:w,:] = t1_img
        combined[:,w:w*2,:] = t2_img
        combined[:,w*2:w*3,:] = mask
        combined[:,w*3:,:] = masked_t2
        np.save(pairs_path + '/pairs/' + str(i) + '.npy', combined)

        # salva imagens JPEG
        if config['debug_mode']:
            if len(config['channels']) > 3:
                combined = combined[:,:,config['debug_channels']]
            if config['type_norm'] != 0:
                combined = (combined + 1) * 127.5
                if synthetic_input_pairs:
                    combined[:,w:w*2,:] = 0
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
        sent2_2018_1 = load_tif_image(config['root_path'] + 'img/2018_10m_b2348.tif').astype('float32')
        sent2_2018_2 = load_tif_image(config['root_path'] + 'img/2018_20m_b5678a1112.tif').astype('float32')
        # Resize bands of 20m
        sent2_2018_2 = resize_image(sent2_2018_2.copy(), sent2_2018_1.shape[0], sent2_2018_1.shape[1])
        sent2_2018 = np.concatenate((sent2_2018_1, sent2_2018_2), axis=-1)
        del sent2_2018_1, sent2_2018_2

    # LOAD IMAGE T2
    sent2_2019_1 = load_tif_image(config['root_path'] + 'img/2019_10m_b2348.tif').astype('float32')
    sent2_2019_2 = load_tif_image(config['root_path'] + 'img/2019_20m_b5678a1112.tif').astype('float32')
    # Resize bands of 20m
    sent2_2019_2 = resize_image(sent2_2019_2.copy(), sent2_2019_1.shape[0], sent2_2019_1.shape[1])
    sent2_2019 = np.concatenate((sent2_2019_1, sent2_2019_2), axis=-1)
    del sent2_2019_1, sent2_2019_2
    
    print('Filtering outliers:')

    sent2_2019 = sent2_2019[:, :, config['channels']]
    sent2_2019 = filter_outliers(sent2_2019.copy())
    image_stack = sent2_2019
    del sent2_2019

    if config['change_detection']:
        sent2_2018 = sent2_2018[:, :, config['channels']]  
        sent2_2018 = filter_outliers(sent2_2018.copy()) 
        image_stack = np.concatenate((sent2_2018, image_stack), axis=-1)
        del sent2_2018 #, sent2_2019

    print('Loading reference:')

    final_mask = load_tif_image(config['root_path'] + 'ref/r10m_def_2019.tif').astype('int')
    final_mask = np.transpose(final_mask.copy(), (1, 0))
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
    # rejected_pixels_count = []
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
            # print('descartado')
            rejected_patches_.append(i)
            rejected_patches_ref.append(i) 
            # rejected_pixels_count.append(len(class1))
    return patches[patches_], patches_ref[patches_ref_], patches[rejected_patches_], patches_ref[rejected_patches_ref] #, rejected_pixels_count


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
    # row_steps, cols_steps = int((1-overlap) * patch_size[0]), int((1-overlap) * patch_size[1])
    # patches = skimage.util.view_as_windows(im_idx, patch_size, step=(row_steps, cols_steps))
    window_shape = patch_size
    window_shape_array = (window_shape, window_shape, input_image.shape[2])
    window_shape_ref = (window_shape, window_shape)
    patches_array = np.array(view_as_windows(input_image, window_shape_array, step = stride))
    patches_ref = np.array(view_as_windows(reference, window_shape_ref, step = stride))
    num_row,num_col,p,row,col,depth = patches_array.shape
    patches_array = patches_array.reshape(num_row*num_col,row,col,depth)
    patches_ref = patches_ref.reshape(num_row*num_col,row,col)
    return patches_array, patches_ref


def patch_tiles(tiles, mask_amazon, image_array, image_ref, stride, config, save_rejected=False):
    '''Extraction of image patches and labels '''
    patch_size = config['patch_size']
    rej_out_path = config['output_path'] + '/rejected_patches_npy/'
    os.makedirs(rej_out_path, exist_ok=True)
    patches_out = []
    label_out = []
    counter = 0
    for num_tile in tiles:
        print(num_tile)
        counter+=1
        # print(num_tile, str(counter))
        rows, cols = np.where(mask_amazon == num_tile)
        x1 = np.min(rows)
        y1 = np.min(cols)
        x2 = np.max(rows)
        y2 = np.max(cols)
        tile_img = image_array[x1:x2 + 1, y1:y2 + 1, :]
        tile_ref = image_ref[x1:x2 + 1, y1:y2 + 1]
        patches_img, patch_ref = extract_patches(tile_img, tile_ref, patch_size, stride)
        # descarta por %
        patches_img, patch_ref, rej_patches, rej_patches_ref = discard_patches_by_percentage(patches_img, patch_ref, config)
        # salva os patches rejeitados
        if save_rejected:
            np.save(rej_out_path + 'rej_patches_tile_' + str(num_tile) + '_img.npy', rej_patches) 
            np.save(rej_out_path + 'rej_patches_tile_' + str(num_tile) + '_ref.npy', rej_patches_ref)
        patches_out.append(patches_img)
        label_out.append(patch_ref)
    patches_out = np.concatenate(patches_out)
    label_out = np.concatenate(label_out)
    return patches_out, label_out


def save_npy_array(np_array, out_path):
    if len(np_array) != 0:
        np.save(out_path, np_array)
    else:
        print('Empty array')
        return



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


def normalize_img_array(image, norm_type = 1, scaler = None):
    if image.ndim == 4:
      image_reshaped = image.reshape((image.shape[0]*image.shape[1]*image.shape[2]),image.shape[3])
    if image.ndim == 3:
      image_reshaped = image.reshape((image.shape[0]*image.shape[1]),image.shape[2])

    # filtrar os pontos da mascara la 
    #imgn = np.where(img>0, img, np.nan)
    #img_norm = (img - np.nanmean(imgn,axis=(0,1)))/np.nanstd(imgn,axis=(0,1))

    if scaler != None:
      print('Fitting data to provided scaler...')
    else:
      print('No scaler was provided. Fitting scaler', str(norm_type), 'to data...')
      if (norm_type == 1):
          scaler = StandardScaler()
      if (norm_type == 2):
          scaler = MinMaxScaler(feature_range=(0,1))
      if (norm_type == 3):
          scaler = MinMaxScaler(feature_range=(-1,1))
      if (norm_type == 4):
          scaler = MinMaxScaler(feature_range=(0,2))
      if (norm_type == 5):
          scaler = MinMaxScaler(feature_range=(0,255))
      scaler = scaler.fit(image_reshaped)
    image_normalized = scaler.transform(image_reshaped)
    if image.ndim == 4:
      image_normalized1 = image_normalized.reshape(image.shape[0],image.shape[1],image.shape[2], image.shape[3])
    if image.ndim == 3:
      image_normalized1 = image_normalized.reshape(image.shape[0],image.shape[1],image.shape[2])
    return image_normalized1, scaler


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
    image_normalized = scaler.transform(image_reshaped)
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

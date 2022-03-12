
import tensorflow as tf
import joblib
import yaml
import os

from utils_unet import *
from utils import *

unet_path = 'C:/Users/amandalucs/Documents/Github/unet-results/2022_03_11_21_05_47_augmented'

seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

stream = open('./config.yaml')
config = yaml.load(stream, Loader=yaml.CLoader)

output_folder = unet_path + '/test/'
os.makedirs(output_folder, exist_ok=True)

if config['run_inference_on_cpu']:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

batch_size = config['batch_size_unet']
epochs = config['epochs_unet']
nb_filters = config['nb_filters']
number_runs = config['times']
patience_value = config['patience_value']
type_norm_unet = config['type_norm_unet']
number_class = 2 
tiles_ts = (list(set(np.arange(20)+1)-set(config['tiles_tr'])-set(config['tiles_val'])))

print('[*] Loading image array...')
image_array, final_mask, _ = get_dataset(config)
# normalize image to [-1, +1] with the same scaler used in preprocessing
print('> Loading provided scaler:', config['scaler_path'])
preprocessing_scaler = joblib.load(config['scaler_path'])
image_array, _ = normalize_img_array(image_array, config['type_norm'], scaler=preprocessing_scaler) # [-1, +1]
# u-net expects input to be  [0, 1]. [-1, +1] => [0, 1]:
image_array = image_array*0.5 + 0.5
print('> image_array:', np.min(image_array), np.max(image_array))

mask_tiles = create_mask(final_mask.shape[0], final_mask.shape[1], grid_size=(5, 4))
print('[*] Creating padded image...')
n_pool = 3
n_rows = 5
n_cols = 4
rows, cols = image_array.shape[:2]
pad_rows = rows - np.ceil(rows/(n_rows*2**n_pool))*n_rows*2**n_pool
pad_cols = cols - np.ceil(cols/(n_cols*2**n_pool))*n_cols*2**n_pool
npad = ((0, int(abs(pad_rows))), (0, int(abs(pad_cols))), (0, 0))
image1_pad = np.pad(image_array, pad_width=npad, mode='reflect')
h, w, c = image1_pad.shape
patch_size_rows = h//n_rows
patch_size_cols = w//n_cols
num_patches_x = int(h/patch_size_rows)
num_patches_y = int(w/patch_size_cols)
input_shape=(patch_size_rows,patch_size_cols, c)

weights = [0.2, 0.8]
loss = weighted_categorical_crossentropy(weights)

time_ts = []
for run in range(0, number_runs):
    current_model = unet_path + '/checkpoints/model_' + str(run)
    net = tf.keras.models.load_model(current_model, custom_objects={"loss": loss})
    # net.summary()
    # testing the model
    new_model = build_unet(input_shape, nb_filters, number_class)
    for l in range(1, len(net.layers)):
        new_model.layers[l].set_weights(net.layers[l].get_weights())
    print('Loaded weights for testing model', str(run))
    patch_t = []
    start_test = time.time()
    for i in range(0,num_patches_y):
        print('[' + str(i) + '/' + str(num_patches_y) + ']', end='\r')
        for j in range(0,num_patches_x):
            patch = image1_pad[patch_size_rows*j:patch_size_rows*(j+1), patch_size_cols*i:patch_size_cols*(i+1), :]
            predictions_ = new_model.predict(np.expand_dims(patch, axis=0)) 
            del patch 
            patch_t.append(predictions_[:,:,:,1])
            del predictions_
    end_test =  time.time() - start_test
    patches_pred = np.asarray(patch_t).astype(np.float32)
    print(patches_pred.shape)
    prob_recontructed = pred_reconctruct(h, w, num_patches_x, num_patches_y, patch_size_rows, patch_size_cols, patches_pred)
    np.save(output_folder +'/prob_'+str(run)+'.npy',prob_recontructed) 
    time_ts.append(end_test)
    del prob_recontructed, net, patches_pred

time_ts_array = np.asarray(time_ts)
np.save(output_folder + '/metrics_ts.npy', time_ts_array)


prob_rec = np.zeros((image1_pad.shape[0],image1_pad.shape[1], number_runs))
for run in range (0, number_runs):
    prob_rec[:,:,run] = np.load(output_folder+'/'+'prob_'+str(run)+'.npy').astype(np.float32)

mean_prob = np.mean(prob_rec, axis = -1)
np.save(output_folder + '/prob_mean.npy', mean_prob)

cmap = matplotlib.colors.ListedColormap(['black', 'white'])

# Plot mean map and reference
fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(121)
plt.title('Prediction')
ax1.imshow(mean_prob, cmap = cmap)
ax1.axis('off')

ref2 = final_mask.copy()
ax2 = fig.add_subplot(122)
plt.title('Reference')
ax2.imshow(ref2, cmap = cmap)
ax2.axis('off')
fig.savefig(output_folder + '/mean_map_and_ref.png')

mean_prob = mean_prob[:final_mask.shape[0], :final_mask.shape[1]]

mask_amazon_ts = np.zeros((mask_tiles.shape)).astype('float32')
for ts_ in tiles_ts:
    mask_amazon_ts[mask_tiles == ts_] = 1

ref1 = np.ones_like(final_mask).astype(np.float32)
TileMask = mask_amazon_ts * ref1
GTTruePositives = final_mask==1

Npoints = 10
Pmax = np.max(mean_prob[GTTruePositives * TileMask ==1])
ProbList = np.linspace(Pmax,0,Npoints)

metrics_ = matrics_AA_recall(ProbList, mean_prob, final_mask, mask_amazon_ts, 625)
np.save(output_folder+'/acc_metrics.npy',metrics_)

# Complete NaN values
metrics_copy = metrics_.copy()
metrics_copy = complete_nan_values(metrics_copy)

# Comput Mean Average Precision (mAP) score 
Recall = metrics_copy[:,0]
Precision = metrics_copy[:,1]
AA = metrics_copy[:,2]
    
DeltaR = Recall[1:]-Recall[:-1]
AP = np.sum(Precision[:-1]*DeltaR)
print(output_folder)
print('mAP:', AP)

#X -> Recall
#Y -> Precision
mAP_func = Area_under_the_curve(Recall, Precision)
print('mAP_func:', mAP_func)

# Plot Recall vs. Precision curve
plt.close('all')
fig = plt.figure(figsize=(15,10))
plt.plot(metrics_copy[:,0],metrics_copy[:,1])
plt.plot(metrics_copy[:,0],metrics_copy[:,2])
plt.grid()
fig.savefig(output_folder + '/roc.png')
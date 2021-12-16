import matplotlib.pyplot as plt
import numpy as np
import os

output_folder = '/share_alpha_2/amandalucas/cluster_analysis/exp_include_dif_False_use_only_dif_True/'
img_path = output_folder + 'kmeans_image.npy'

im = np.load(img_path)
im[im == 0] = 3
im[im == 1] = 0
im[im == 3] = 1

fig = plt.figure(figsize=(15,12))
plt.imshow(im*0.5, cmap='viridis')
plt.show()
plt.axis('off')
fig.savefig(output_folder + 'kmeans_image_fix.png')
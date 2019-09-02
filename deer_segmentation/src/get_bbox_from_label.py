#%%
import os
import sys
from glob import glob
import numpy as np
import skimage
from skimage.measure import label
import matplotlib.pyplot as plt
import cv2

#%%
def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return [rmin, rmax, cmin, cmax]

#%%
def tmp(mask):
    separated = skimage.measure.label(mask, background=0)
    max_label = np.max(separated)
    coords_dict = {}
    for i in range(1,max_label+1):
        lbl = np.zeros_like(mask)
        lbl[separated==i] = 1
        box = bbox(lbl)
        coords_dict[i] = {
            'xmin':box[2], 
            'xmax':box[3], 
            'ymin':box[0], 
            'ymax':box[1]
        }

    return coords_dict, separated

#%%
mask_dir = '/home/mitsuki/take/deer_segmentation_copy/camvid_data/label'
mask_paths = glob(os.path.join(mask_dir, 'label_*.png'))

save_dir = './bbox_tmp'
os.makedirs(save_dir, exist_ok=True)

for mask_path in mask_paths:
    save_path = os.path.join(save_dir, os.path.basename(mask_path))

    mask = cv2.imread(mask_path)[:,:,0]
    coords, separated = tmp(mask)

    fig, ax = plt.subplots()
    plt.imshow(separated*100)

    if len(coords.keys())>0:   
        for i in range(1, max(coords.keys())+1):
            ax.add_patch(plt.Rectangle(xy=(coords[i]['xmin'],coords[i]['ymin']),
                                        width=coords[i]['xmax']-coords[i]['xmin']+1,
                                        height=coords[i]['ymax']-coords[i]['ymin']+1,
                                        fill=False, 
                                        edgecolor='r', 
                                        linewidth=2))

    plt.savefig(save_path)
    plt.close()


#%%

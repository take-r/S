import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from label_vis import overlay_label, set_color_to_label


def visualize_results(save_path, 
                      image, 
                      gt, 
                      pred,
                      uncertainty=None,
                      image_weight=1,
                      label_weight=.8,
                      contour=False,
                      contour_width=1):

    if len(image.shape)>3:
        image = np.squeeze(image)

    gt = np.squeeze(gt)
    pred = np.squeeze(pred)

    lookup_table = {0:{'name':'background', 'R':0, 'G':0, 'B':0},
                    1:{'name':'deer', 'R':1, 'G':0, 'B':0}}

    gt = set_color_to_label(gt, lookup_table)
    pred = set_color_to_label(pred, lookup_table)

    gt_overlayed = overlay_label(image, 
                                 gt, 
                                 image_weight, 
                                 label_weight,
                                 contour,
                                 contour_width)

    pred_overlayed = overlay_label(image,
                                   pred, 
                                   image_weight,
                                   label_weight,
                                   contour,
                                   contour_width)

    image_list = [image, pred_overlayed, gt_overlayed]
    title_list = ['Input', 'Predicted', 'Ground Truth']

    if uncertainty is None:
        plt.figure(figsize=(15, 5))
        for i, (im, title) in enumerate(zip(image_list, title_list)):
            plt.subplot(1, 3, i+1)
            plt.imshow(im)
            plt.axis('off')
            plt.title(title)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    else:
        plt.figure(figsize=(20, 5))
        for i, (im, title) in enumerate(zip(image_list, title_list)):
            plt.subplot(1, 4, i+1)
            plt.imshow(im)
            plt.axis('off')
            plt.title(title)

        plt.subplot(1, 4, 4)
        plt.imshow(uncertainty, cmap='jet', vmin=0, vmax=1e-2) #NOTE: adhoc
        plt.axis('off')
        plt.title('Uncertainty')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

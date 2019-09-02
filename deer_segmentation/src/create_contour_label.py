import sys
import os
import cv2
import glob
import numpy as np

def extract_contour(label, iteration=1):
   
    kernel = np.ones((5,5),np.uint8)
    dilated = cv2.dilate(label, kernel, iterations=1)
    contour = dilated - label
    return contour

def main():
    label_dir = '/home/mitsuki/take/deer_segmentation_copy/camvid_data/label'
    label_files = glob.glob(os.path.join(label_dir, 'label_*.png'))

    save_dir = '/home/mitsuki/take/deer_segmentation_copy/camvid_data/label_contour'
    os.makedirs(save_dir, exist_ok=True)

    for label_file in label_files:
        save_filename = os.path.basename(label_file)

        label = cv2.imread(label_file)[:, :, 0]
        contour = extract_contour(label)
    
        # import matplotlib.pyplot as plt
        # plt.imshow(contour*100)
        # plt.show()

        cv2.imwrite(os.path.join(save_dir, save_filename), contour*100)

if __name__=='__main__':
    main()
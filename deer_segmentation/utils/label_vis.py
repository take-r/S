import os
import numpy as np
import pandas as pd
import cv2

def overlay_label(image, 
                  label, 
                  image_weight=1.0, 
                  label_weight=0.5, 
                  contour=False,
                  contour_width=0):

    '''overlay label on image following the given weight
    when image is not RGB image, convert to RGB before superimposing
    '''

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    image = image.astype('uint8')
    label = label.astype('uint8')

    if contour:
        label = gen_contour_label(label, contour_width)

    if label_weight==1:
        mask = np.sum(label, axis=2)
        image[mask!=0] = 0

    overlayed = cv2.addWeighted(image, image_weight, label, label_weight, gamma=0)
    return overlayed

def gen_contour_label(label, contour_width):

    '''generate contourized label
    '''

    contour_label = np.zeros_like(label)
    for i in range(int(np.max(label))):
        l = label == i+1
        c = extract_contour(l.astype(np.uint8), contour_width=contour_width)
        contour_label[c==1] = i+1
    label = contour_label
    return label

def extract_contour(label, contour_width):

    '''extract contour by subtracing given label from eroded one
    '''

    kernel = np.array([[1,1,1], [1,1,1], [1,1,1]])
    eroded = cv2.erode(label, kernel, iterations=contour_width)
    contour = label - eroded
    return contour

def get_lookup_table(lookup_table_path):

    '''return lookup table in dictionary format
    '''

    df = pd.read_csv(lookup_table_path)
    lookup_table = {}
    for i in range(len(df)):
        lookup_table[i] = {}
        lookup_table[i]['R'] = df['color_r'][i]
        lookup_table[i]['G'] = df['color_g'][i]
        lookup_table[i]['B'] = df['color_b'][i]
        lookup_table[i]['name'] = df['name'][i]

    return lookup_table


def set_color_to_label(label, lookup_table):

    '''set RGB color to label
    '''

    assert len(np.squeeze(label).shape)==2, 'input array has already been set RGB color.'
    max_label = np.max(label)
    label = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)

    for idx in range(max_label+1):
        if idx==0:
            color_label = label.copy()
        else:
            value = [lookup_table[idx]['R'], lookup_table[idx]['G'], lookup_table[idx]['B']]
            value = np.array(value)
            color_label = np.where(label==idx, value*255., color_label)
    color_label = color_label.astype(np.uint8)

    return color_label


if __name__=='__main__':
    # below is for debugging

    import matplotlib.pyplot as plt
    from PIL import Image

    # config
    image_weight = 1
    label_weight = .8
    contour = True
    contour_width = 1

    data_dir = '../data/all/170929'
    filename = 'Label_1.png'
    image = np.asarray(Image.open(os.path.join(data_dir, 'picture', filename)))
    label = np.asarray(Image.open(os.path.join(data_dir, 'labeled', filename)))

    lookup_table_path = 'lookup_table.csv'
    lookup_table = get_lookup_table(lookup_table_path)
    label = set_color_to_label(label, lookup_table)

    overlayed = overlay_label(image, 
                              label, 
                              image_weight, 
                              label_weight,
                              contour,
                              contour_width)
    
    plt.imsave('test_label.png', label)
    plt.imsave('test.png', overlayed)

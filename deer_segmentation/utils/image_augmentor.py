import os
import sys
import numpy as np
import cv2
from transform.affine_transforms import AffineTransform
from transform.non_affine_transforms import NonAffineTransform

#TODO: include color transformation (eg. gamma-transform)

class ImageAugmentor(object):
    def __init__(self,
                 seed,
                 horizontal_flip=False,                   # holizontal flip
                 zoom_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 rotation_range=0.,
                 elastic_transform_flag=False,           # elastic transform 
                 elastic_transform_alpha=2, 
                 elastic_transform_sigma=0.8,
                 random_erasing_flag=False,              # random erasing
                 random_erasing_size=(0.02, 0.1), 
                 random_erasing_ratio=(0.3, 1.0),
                 **kwargs): 

        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)

        self.horizontal_flip = horizontal_flip
        self.zoom_range = zoom_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.rotation_range = rotation_range        

        if np.isscalar(self.zoom_range):
            self.zoom_range = [1-self.zoom_range, 1+self.zoom_range]
        elif len(self.zoom_range) == 2:
            self.zoom_range = [self.zoom_range[0], self.zoom_range[1]]
        else:
            raise ValueError('`zoom_range` should be a float or '
                             'a tuple or list of two floats. '
                             'Received: %s' % (self.zoom_range,))

        self.affine = AffineTransform(seed=self.seed,
                                      horizontal_flip=self.horizontal_flip,
                                      zoom_range=self.zoom_range,
                                      width_shift_range=self.width_shift_range,
                                      height_shift_range=self.height_shift_range,
                                      shear_range=self.shear_range,
                                      rotation_range=self.rotation_range)


        self.elastic_transform_flag = elastic_transform_flag
        self.elastic_transform_alpha = elastic_transform_alpha
        self.elastic_transform_sigma = elastic_transform_sigma
        self.random_erasing_flag = random_erasing_flag
        self.random_erasing_size = random_erasing_size
        self.random_erasing_ratio = random_erasing_ratio
        self.nonaffine = NonAffineTransform(seed=self.seed,
                                            elastic_transform_flag=self.elastic_transform_flag,
                                            elastic_transform_alpha=self.elastic_transform_alpha,
                                            elastic_transform_sigma=self.elastic_transform_sigma,
                                            random_erasing_flag=self.random_erasing_flag,
                                            random_erasing_size=self.random_erasing_size,
                                            random_erasing_ratio=self.random_erasing_ratio)

    def _get_fill_mode(self, 
                       label,
                       interp,
                       borderMode,
                       borderValue):
        # determine interpolation mode
        if label:
            interp = cv2.INTER_NEAREST
        else:
            # default interpolation mode for image
            interp = cv2.INTER_LINEAR

        # determine border mode
        if borderValue is not None:
            borderMode = cv2.BORDER_CONSTANT
        else:
            if borderMode=='reflect':
                borderMode = cv2.BORDER_REFLECT
            elif borderMode=='replicate':
                borderMode = cv2.BORDER_REPLICATE
            elif borderMode=='wrap':
                borderMode = cv2.BORDER_WRAP
            else:
                raise ValueError('`borderMode` should be `reflect`, `replicate` or'
                                ' `wrap`. Received: %s' % (borderMode,))

        return interp, borderMode

    def _set_transform(self, img_shape):
        self.affine._set_transform(img_shape)
        self.nonaffine._set_transform()

    def augment(self, 
                x,
                y,
                interp='linear',
                borderMode='reflect',
                borderValue=None):

        img_interp, img_borderMode = self._get_fill_mode(False, interp, borderMode, borderValue)
        lbl_interp, lbl_borderMode = self._get_fill_mode(True, interp, borderMode, borderValue)

        for d in [x, y]:
            if len(d.shape)!=4:
                raise ValueError('image must be 4-dimensional data.'
                                '(input shape:{})'.format(d.shape))

        assert(x.shape[0]==y.shape[0])

        for i in range(x.shape[0]):

            # define transform
            self._set_transform(img_shape=(x.shape[1], x.shape[2]))

            # affine transformation
            tfm_x = self.affine.flow(x[i],
                                   interp=img_interp,
                                   borderMode=img_borderMode,
                                   borderValue=borderValue)
            tfm_y = self.affine.flow(y[i],
                                   interp=lbl_interp,
                                   borderMode=lbl_borderMode,
                                   borderValue=borderValue)

            # non-affine transformation
            tfm_x = self.nonaffine.flow(tfm_x,
                                      label=False,
                                      interp=img_interp,
                                      borderMode=img_borderMode,
                                      borderValue=borderValue)
            tfm_y = self.nonaffine.flow(tfm_y,
                                      label=True,
                                      interp=lbl_interp,
                                      borderMode=lbl_borderMode,
                                      borderValue=borderValue)

            # if len(tfm.shape)!=3:
            #     if len(tfm.shape)==2:
            #         tfm = np.expand_dims(tfm, axis=-1)

            x[i] = tfm_x
            y[i] = np.expand_dims(tfm_y, axis=-1)

        return x, y

if __name__=='__main__':

    # sanity check
    import matplotlib.pyplot as plt

    config = {
        'augmentation_params': {
            'augmentation': True,
            'seed':0,
            'horizontal_flip':True,
            'zoom_range':0.1,
            'width_shift_range':0.,
            'height_shift_range':0.,
            'shear_range':5.,
            'rotation_range':10,
            'elastic_transform_flag':False,
            'elastic_transform_alpha':2,
            'elastic_transform_sigma':.8,
            'random_erasing_flag':True,
            'random_erasing_size':(0.02, 0.05),
            'random_erasing_ratio':(0.5, 1.0)
                }
            }

    augmentor = ImageAugmentor(**config['augmentation_params'])

    nrow, ncol = 4, 2
    batch_size = 8
    X, y = [], []

    # for idx in range(0, 600, 600//batch_size):
    for idx in range(batch_size):
        X.append(plt.imread('../camvid_data/image/image_{:04d}.png'.format(idx)))
        y.append(plt.imread('../camvid_data/label/label_{:04d}.png'.format(idx)))
    
    X = np.array(X)
    y = np.array(y)
    y = np.expand_dims(y, axis=-1)

    augmented_X, augmented_y = augmentor.augment(X, y, borderMode='reflect')

    # print(augmented_X.shape)
    # print(augmented_y.shape)
    
    plt.figure(figsize=(10, 5))
    cnt = 0

    for aug_X, aug_y in zip(augmented_X, augmented_y):
        plt.subplot(nrow, ncol, cnt+1)
        concat = cv2.hconcat([aug_X, np.concatenate([aug_y, aug_y, aug_y], axis=-1)*100])
        plt.imshow(concat)
        cnt += 1

    plt.show()
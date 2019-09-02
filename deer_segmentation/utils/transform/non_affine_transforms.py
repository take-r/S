import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter
import copy

class NonAffineTransform(object):
    def __init__(self,
                 seed,
                 elastic_transform_flag,
                 elastic_transform_alpha,
                 elastic_transform_sigma,
                 random_erasing_flag,
                 random_erasing_size,
                 random_erasing_ratio):

        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)

        self.elastic_transform_flag = elastic_transform_flag
        self.elastic_transform_alpha = elastic_transform_alpha
        self.elastic_transform_sigma = elastic_transform_sigma
        self.random_erasing_flag = random_erasing_flag
        self.random_erasing_size = random_erasing_size
        self.random_erasing_ratio = random_erasing_ratio

        self.elastic_transform_prob = 0
        self.random_erasing_prob = 0

    def _set_transform(self):

        if self.elastic_transform_flag:
            self.elastic_transform_prob = self.random_state.rand()

        if self.random_erasing_flag:
            self.random_erasing_prob = self.random_state.rand()


    def flow(self, img, label, interp, borderMode, borderValue):
        #NOTE: elastic transform -> random erasing

        if (self.elastic_transform_prob > 0.5) * self.elastic_transform_flag:
            img = self.check_shape(img)
            img = self.elastic_transform(img, 
                                         interp=interp, 
                                         borderMode=borderMode,
                                         borderValue=borderValue)


        if not label:
            if (self.random_erasing_prob > 0.5) * self.random_erasing_flag:
                img = self.check_shape(img)
                img = self.random_erasing(img)
                # print('random_erasing was applied')

        return img

    def check_shape(self, img):

        if len(img.shape)==3:
            return img

        elif len(img.shape)==2:
            img = np.expand_dims(img, axis=-1)
            return img

        else:
            raise ValueError('size of image is invalid.(shape:{})'.format(img.shape))


    def elastic_transform(self, img, interp, borderMode, borderValue):

        h, w, c = img.shape

        alpha = w*self.elastic_transform_alpha
        sigma = w*self.elastic_transform_sigma

        dx = np.float32(gaussian_filter((self.random_state.rand(h, w)*2-1), sigma)*alpha)
        dy = np.float32(gaussian_filter((self.random_state.rand(h, w)*2-1), sigma)*alpha)
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        mapx = np.float32(x+dx)
        mapy = np.float32(y+dy)

        img = cv2.remap(img, mapx, mapy, 
                        interpolation=interp,
                        borderMode=borderMode,
                        borderValue=borderValue)
        return img

    def random_erasing(self, img):

        img = copy.deepcopy(img)
        h, w, c = img.shape

        mask_area = self.random_state.randint(h*w*self.random_erasing_size[0], h*w*self.random_erasing_size[1])
        mask_aspect_ratio = self.random_erasing_ratio[0] + self.random_state.rand()*self.random_erasing_ratio[1]

        mask_h = int(np.sqrt(mask_area/mask_aspect_ratio))
        mask_w = int(mask_aspect_ratio*mask_h)
        mask_h = min(mask_h, h-1)
        mask_w = min(mask_w, w-1)

        random_value = self.random_state.uniform(0, np.max(img), (mask_h, mask_w, c))

        top = self.random_state.randint(0, h-mask_h)
        left = self.random_state.randint(0, w-mask_w)
        img[top:top+mask_h, left:left+mask_w, :] = random_value
        return img


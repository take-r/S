import numpy as np
import cv2

class AffineTransform(object):
    def __init__(self,
                 seed,
                 horizontal_flip,
                 zoom_range,
                 width_shift_range,
                 height_shift_range,
                 shear_range,
                 rotation_range):

        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)

        self.horizontal_flip = horizontal_flip
        self.zoom_range = zoom_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.rotation_range = rotation_range


    def _set_transform(self, img_shape):
        self.mtx = self._get_random_transform_mtx(img_shape)
        

    def flow(self, img, interp, borderMode, borderValue):
        if len(img.shape)!=3:
            if len(img.shape)==2:
                img = np.expand_dims(img, axis=-1)
            else:
                raise ValueError('size of image is invalid.(shape:{})'.format(img.shape))

        h, w, c = img.shape
        transformed = cv2.warpAffine(img, 
                                     self.mtx[:2], 
                                     (w, h), 
                                     flags=interp, 
                                     borderMode=borderMode, 
                                     borderValue=borderValue)
        return transformed


    def _get_random_shift_mtx(self, tx, ty):
        mtx = np.array([[1, 0, tx],
                        [0, 1, ty],
                        [0, 0, 1]])
        return mtx

    def _get_scale_mtx(self, zx, zy):
        mtx = np.array([[zx, 0, 0],
                        [0, zy, 0],
                        [0, 0, 1]])
        return mtx

    def _get_shear_mtx(self, shear):
        shear = np.deg2rad(shear)
        mtx = np.array([[1, -np.sin(shear), 0],
                        [0, np.cos(shear), 0],
                        [0, 0, 1]])
        return mtx

    def _get_hflip_mtx(self, h):
        mtx = np.array([[-1, 0, h],
                        [0, 1, 0],
                        [0, 0, 1]])
        return mtx

    def _get_vflip_mtx(self, w):
        mtx = np.array([[1, 0, 0],
                        [0, -1, w],
                        [0, 0, 1]])
        return mtx

    def _get_rotate_mtx(self, theta):
        theta = np.deg2rad(theta)
        mtx = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])        
        return mtx

    #NOTE: the origin of rotation is the center of the image
    def _get_center_rotate_mtx(self, theta, h, w):
        mtx1 = np.array([[1, 0, -w//2],
                         [0, 1, -h//2],
                         [0, 0, 1]])
        theta = np.deg2rad(theta)
        mtx2 = np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])        
        mtx3 = np.array([[1, 0, w//2],
                         [0, 1, h//2],
                         [0, 0, 1]])
        mtx = np.dot(mtx2, mtx1)
        mtx = np.dot(mtx3, mtx)
        return mtx


    def _get_random_transform_mtx(self, img_shape):
        #NOTE: hflip-(vflip)->scale->shift->shear->rotate

        h, w = img_shape
        tfm_mtx = np.eye(3)

        #---------------- horizontal flip ----------------#
        if (self.random_state.rand() > 0.5) * self.horizontal_flip:
            mtx = self._get_hflip_mtx(w)
            tfm_mtx = np.dot(mtx, tfm_mtx)
        #-------------------------------------------------#

        #---------------- scale ----------------#
        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = self.random_state.uniform(self.zoom_range[0], self.zoom_range[1], 2)        
        mtx = self._get_scale_mtx(zx, zy)
        tfm_mtx = np.dot(mtx, tfm_mtx)
        #---------------------------------------#

        #---------------- shift ----------------#
        if self.width_shift_range:
            try:  # 1-D array-like or int
                tx = self.random_state.choice(self.width_shift_range)
                tx *= self.random_state.choice([-1, 1])
            except ValueError:  # floating point
                tx = self.random_state.uniform(-self.width_shift_range, self.width_shift_range)
            if np.max(self.width_shift_range) < 1:
                tx *= img_shape[0]
        else:
            tx = 0

        if self.height_shift_range:
            try:  # 1-D array-like or int
                ty = self.random_state.choice(self.height_shift_range)
                ty *= self.random_state.choice([-1, 1])
            except ValueError:  # floating point
                ty = self.random_state.uniform(-self.height_shift_range, self.height_shift_range)
            if np.max(self.height_shift_range) < 1:
                ty *= img_shape[1]
        else:
            ty = 0
        mtx = self._get_random_shift_mtx(tx, ty)
        tfm_mtx = np.dot(mtx, tfm_mtx)
        #---------------------------------------#

        #---------------- shear ----------------#
        if self.shear_range:
            shear = self.random_state.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        mtx = self._get_shear_mtx(shear)
        tfm_mtx = np.dot(mtx, tfm_mtx)
        #---------------------------------------#

        #-------------- rotation --------------#
        if self.rotation_range:
            theta = self.random_state.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        mtx = self._get_center_rotate_mtx(theta, h, w)
        # mtx = self._get_rotate_mtx(theta)
        tfm_mtx = np.dot(mtx, tfm_mtx)
        #--------------------------------------#


        return tfm_mtx

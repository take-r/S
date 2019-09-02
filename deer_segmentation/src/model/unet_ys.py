from math import ceil, floor

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout
from keras.layers import UpSampling2D, Cropping2D
from keras.layers.advanced_activations import PReLU, LeakyReLU, ELU
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate


class UNet():
    def __init__(self, input_shape, 
                 n_class, 
                 n_layer=4, 
                 activation='relu',
                 n_kernel=32, 
                 kernel_size=3, 
                 kernel_initializer='he_normal',
                 dropout=False,
                 encoder_drop_prob=0.2, 
                 decoder_drop_prob=0.2,
                 pool_size=2, 
                 batch_norm=True, 
                 **kwargs):

        self.input_shape = input_shape
        self.n_class = n_class
        self.n_layer = n_layer
        self.activation = activation
        self.n_kernel = n_kernel
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.dropout = dropout
        self.encoder_drop_prob = encoder_drop_prob
        self.decoder_drop_prob = decoder_drop_prob
        self.pool_size = pool_size
        self.batch_norm = batch_norm

    def my_activation(self, name):
        if name == 'prelu':
            return PReLU()
        elif name == 'lrelu':
            return LeakyReLU()
        elif name == 'elu':
            return ELU()
        else:
            return Activation(name)
    
    def contract(self, prev_layer, n_kernel, kernel_size=3, pool_size=2, activation='relu'):
        conv = Conv2D(n_kernel, kernel_size, padding='same', use_bias=True)(prev_layer)
        conv = self.my_activation(activation)(conv)
        n_kernel = n_kernel << 1
        conv = Conv2D(n_kernel, kernel_size, padding='same', use_bias=False)((conv))
        conv = self.my_activation(activation)(BatchNormalization()(conv))
        pool = MaxPooling2D(pool_size=pool_size, strides=pool_size)((conv))
    
        return conv, pool

    def expand(self, prev_layer, left_layer, n_kernel, kernel_size=3, 
               pool_size=2, activation='relu'):
        dx = (left_layer.shape[1].value-prev_layer.shape[1].value*pool_size)/2
        dy = (left_layer.shape[2].value-prev_layer.shape[2].value*pool_size)/2
        crop_size = ((floor(dx),ceil(dx)), (floor(dy),ceil(dy)))
        up = UpSampling2D(size=pool_size)(prev_layer)
        cropped = Cropping2D(crop_size)(left_layer)
        up = Concatenate(axis=3)([up, cropped])
        conv = Conv2D(n_kernel, kernel_size, padding='same', use_bias=True)(up)
        conv = self.my_activation(activation)(conv)
        n_kernel = n_kernel >> 1
        conv = Conv2D(n_kernel, kernel_size, padding='same', use_bias=False)((conv))
        conv = self.my_activation(activation)(BatchNormalization()(conv))
        return conv
        
    def create_model(self):
        inputs = Input(self.input_shape)
        conv1, pool1 = self.contract(inputs, self.n_kernel, self.kernel_size,
                                     self.pool_size, self.activation)
        self.n_kernel = conv1.shape[-1].value
        conv2, pool2 = self.contract(pool1, self.n_kernel, self.kernel_size,
                                     self.pool_size, self.activation)
        self.n_kernel = conv2.shape[-1].value
        conv3, pool3 = self.contract(pool2, self.n_kernel, self.kernel_size,
                                     self.pool_size, self.activation)
        self.n_kernel = conv3.shape[-1].value
        conv4, pool4 = self.contract(pool3, self.n_kernel, self.kernel_size,
                                     self.pool_size, self.activation)
        self.n_kernel = conv4.shape[-1].value
        conv5, pool5 = self.contract(pool4, self.n_kernel, self.kernel_size,
                                     self.pool_size, self.activation)
        self.n_kernel = conv5.shape[-1].value
    
        conv = self.expand(conv5, conv4, self.n_kernel, self.kernel_size,
                           self.pool_size, self.activation)
        self.n_kernel = conv.shape[-1].value
        conv = self.expand(conv, conv3, self.n_kernel, self.kernel_size,
                           self.pool_size, self.activation)
        self.n_kernel = conv.shape[-1].value
        conv = self.expand(conv, conv2, self.n_kernel, self.kernel_size,
                           self.pool_size, self.activation)
        self.n_kernel = conv.shape[-1].value
        conv = self.expand(conv, conv1, self.n_kernel, self.kernel_size,
                           self.pool_size, self.activation)
    
        output = Conv2D(self.n_class, 1, activation='sigmoid', padding='same')(conv)
    
        return Model(inputs=inputs, outputs=output)
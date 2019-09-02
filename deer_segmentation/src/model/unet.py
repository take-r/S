from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.layers import Activation, Dropout
from keras.layers.advanced_activations import PReLU, LeakyReLU, ELU
from keras.layers.normalization import BatchNormalization


class UNet():
    def __init__(self, input_shape, 
                 n_class, 
                 n_layer=4, 
                 activation='relu',
                 n_kernel=32, 
                 kernel_size=3, 
                 kernel_initializer='he_normal',
                 dropout=False,
                 bayesian_state=False,
                 encoder_drop_prob=0.2, 
                 decoder_drop_prob=0.2,
                 pool_size=2, 
                 batch_norm=True,
                 upsample_method='upsampling',
                 **kwargs):

        self.input_shape = input_shape
        self.n_class = n_class
        self.n_layer = n_layer
        self.activation = activation
        self.n_kernel = n_kernel
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.dropout = dropout
        self.bayesian_state = bayesian_state
        self.encoder_drop_prob = encoder_drop_prob
        self.decoder_drop_prob = decoder_drop_prob
        self.pool_size = pool_size
        self.batch_norm = batch_norm
        self.upsample_method = upsample_method

    def my_activation(self, name):
        if name == 'prelu':
            return PReLU()
        elif name == 'lrelu':
            return LeakyReLU()
        elif name == 'elu':
            return ELU()
        else:
            return Activation(name)

    def _conv_block(self, input_layer):
        """ create convolution block """

        conv = Conv2D(self.n_kernel, self.kernel_size, padding='same',
                      kernel_initializer=self.kernel_initializer)(input_layer)
        if self.batch_norm:
            conv = BatchNormalization()(conv)
        conv = self.my_activation(self.activation)(conv)

        conv = Conv2D(self.n_kernel, self.kernel_size, padding='same', 
                      kernel_initializer=self.kernel_initializer)(conv)
        if self.batch_norm:
            conv = BatchNormalization()(conv)
        conv = self.my_activation(self.activation)(conv)
        return conv
    
    def _encoder_block(self, input_layer):
        """ create encoder block """
        
        self.n_kernel = self.n_kernel << 1
        conv = self._conv_block(input_layer)
        pool = MaxPooling2D(pool_size=self.pool_size)(conv)
        if self.dropout:
            if self.bayesian_state:
                pool = Dropout(self.encoder_drop_prob)(pool, training=True)
            else:
                pool = Dropout(self.encoder_drop_prob)(pool)
        return conv, pool

    def _decoder_block(self, input1, input2):
        """ create decoder block """
        
        self.n_kernel = self.n_kernel >> 1
        
        if self.upsample_method=='upsampling':
            up = UpSampling2D(size=self.pool_size)(input1)
        else:
            up = Conv2DTranspose(self.n_kernel, (2, 2), strides=(2, 2), padding='same')(input1)
        up = concatenate([up, input2], axis=3)
        if self.dropout:
            if self.bayesian_state:
                up = Dropout(self.decoder_drop_prob)(up, training=True)
            else:
                up = Dropout(self.decoder_drop_prob)(up)
        conv = self._conv_block(up)
        return conv   
        
    def get_model(self):
        """ create UNet model """
        
        first = Input(shape=self.input_shape)
        
        # encoder
        conv_layers = []
        for i in range(self.n_layer-1): 
            if i==0:
                conv, pool = self._encoder_block(input_layer=first)
            else:
                conv, pool = self._encoder_block(input_layer=pool)
            self.n_kernel = conv.shape[-1].value
            conv_layers.append(conv)
        
        # bottom
        self.n_kernel = self.n_kernel << 1
        conv = self._conv_block(input_layer=pool)
        self.n_kernel = conv.shape[-1].value
        
        # decoder
        for j in range(self.n_layer-1):
            conv = self._decoder_block(input1=conv, input2=conv_layers[::-1][j])
            self.n_kernel = conv.shape[-1].value

        last = Conv2D(self.n_class, (1, 1), activation='sigmoid', padding='same')(conv)
        return Model(inputs=[first], outputs=[last])
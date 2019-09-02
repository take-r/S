from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.layers import Activation, Dropout, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU, ELU
from keras.layers.normalization import BatchNormalization

class UNet():
    def __init__(self,
                 input_shape, 
                 n_class, 
                 n_layer=5, 
                 activation='relu',
                 n_kernel=32, 
                 kernel_size=3, 
                 kernel_initializer='he_normal',
                 bayesian_state=False,
                 encoder_drop_rate=0.2, 
                 decoder_drop_rate=0.2,
                 pool_size=2, 
                 batch_norm=True, 
                 last_activation='sigmoid',
                 upsample_method='upsampling',
                 residual=False,
                 **kwargs):

        self.input_shape = input_shape
        self.n_class = n_class
        self.n_layer = n_layer
        self.activation = activation
        self.n_kernel = n_kernel
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.bayesian_state = bayesian_state
        self.encoder_drop_rate = encoder_drop_rate
        self.decoder_drop_rate = decoder_drop_rate
        self.pool_size = pool_size
        self.batch_norm = batch_norm
        self.last_act = last_activation
        self.upsample_method = upsample_method
        self.residual = False #NOTE: ad-hoc


    def my_activation(self, name):
        if name == 'prelu':
            return PReLU()
        elif name == 'lrelu':
            return LeakyReLU()
        elif name == 'elu':
            return ELU()
        else:
            return Activation(name)


    def _conv_block(self, 
                    input_layer,
                    n_kernel,
                    kernel_size, 
                    kernel_initializer='glorot_uniform',
                    activation='relu'):
        """ create convolution block """

        # Conv-BatchNormalization-(dropout)-relu
        conv = Conv2D(n_kernel, 
                      kernel_size, 
                      padding='same',
                      kernel_initializer=kernel_initializer)(input_layer)
        if self.batch_norm: 
            conv = BatchNormalization()(conv)
        conv = self.my_activation(activation)(conv)

        conv = Conv2D(n_kernel, 
                      kernel_size, 
                      padding='same',
                      kernel_initializer=kernel_initializer)(conv)
        if self.batch_norm: 
            conv = BatchNormalization()(conv)
        conv = self.my_activation(activation)(conv)
        
        return conv
    
    def _encoder_block(self, 
                       input_layer, 
                       n_kernel, 
                       kernel_size,
                       pool_size, 
                       kernel_initializer='glorot_uniform',
                       activation='relu'):
        """ create encoder block """

        conv = self._conv_block(input_layer, 
                                n_kernel, 
                                kernel_size, 
                                kernel_initializer, 
                                activation)

        if self.encoder_drop_rate:
            if self.bayesian_state:
                conv = Dropout(self.encoder_drop_rate)(conv, training=True)
            else:
                conv = Dropout(self.encoder_drop_rate)(conv)

        pool = MaxPooling2D(pool_size=pool_size)(conv)
        return conv, pool

    def _decoder_block(self, 
                       input_layer1, 
                       input_layer2,
                       n_kernel, 
                       kernel_size,
                       pool_size, 
                       kernel_initializer='glorot_uniform',
                       activation='relu'):
        """ create decoder block """

        if self.upsample_method=='upsampling':
            up = UpSampling2D(size=self.pool_size)(input_layer1)
        else:
            up = Conv2DTranspose(n_kernel, (2, 2), strides=(2, 2), padding='same')(input_layer1)

        up = concatenate([up, input_layer2], axis=3)
        conv = self._conv_block(up, 
                                n_kernel, 
                                kernel_size, 
                                kernel_initializer, 
                                activation)

        if self.decoder_drop_rate:
            if self.bayesian_state:
                conv = Dropout(self.decoder_drop_rate)(conv, training=True)
            else:
                conv = Dropout(self.decoder_drop_rate)(conv)

        return conv   

    def get_model(self):
        """ create model """
        # input layer
        first = Input(shape=self.input_shape)
        
        # encode layers
        conv1, pool1 = self._encoder_block(first, self.n_kernel, self.kernel_size, self.pool_size)
        conv2, pool2 = self._encoder_block(pool1, self.n_kernel*2, self.kernel_size, self.pool_size)
        conv3, pool3 = self._encoder_block(pool2, self.n_kernel*4, self.kernel_size, self.pool_size)
        conv4, pool4 = self._encoder_block(pool3, self.n_kernel*8, self.kernel_size, self.pool_size)
        # conv5, pool5 = self._encoder_block(pool4, self.n_kernel*16, self.kernel_size, self.pool_size)
        # conv6, _ = self._encoder_block(pool5, self.n_kernel*32, self.kernel_size, self.pool_size)

        # decode layers
        # conv = self._decoder_block(conv6, conv5, self.n_kernel*32, self.kernel_size, self.pool_size)
        # conv = self._decoder_block(conv5, conv4, self.n_kernel*16, self.kernel_size, self.pool_size)
        conv = self._decoder_block(conv4, conv3, self.n_kernel*8, self.kernel_size, self.pool_size)
        conv = self._decoder_block(conv, conv2, self.n_kernel*4, self.kernel_size, self.pool_size)
        conv = self._decoder_block(conv, conv1, self.n_kernel*2, self.kernel_size, self.pool_size)

        conv = self._conv_block(conv, self.n_kernel, self.kernel_size)
        conv = Conv2D(self.n_class, (1, 1), padding='same')(conv)
        last = Activation(self.last_act, name='output')(conv)

        model = Model(inputs=[first], outputs=[last])
        return model
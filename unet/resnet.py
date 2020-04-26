from keras.layers import Input
from keras.layers import BatchNormalization, Activation, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model




def build_generator(self):
    def residual_block(layer_input, filters):
        """Residual block described in paper"""
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
        d = Activation('relu')(d)
        d = BatchNormalization(momentum=0.8)(d)
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
        d = BatchNormalization(momentum=0.8)(d)
        d = Add()([d, layer_input])
        return d

    def deconv2d(layer_input):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
        u = Activation('relu')(u)
        return u

    # Low resolution image input
    img_lr = Input(shape=self.lr_shape)

    # Pre-residual block
    c1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
    c1 = Activation('relu')(c1)

    # Propogate through residual blocks
    r = residual_block(c1, self.gf)
    for _ in range(self.n_residual_blocks - 1):
        r = residual_block(r, self.gf)

    # Post-residual block
    c2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
    c2 = BatchNormalization(momentum=0.8)(c2)
    c2 = Add()([c2, c1])

    # Upsampling
    u1 = deconv2d(c2)
    u2 = deconv2d(u1)

    # Generate high resolution output
    gen_hr = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)

    return Model(img_lr, gen_hr)
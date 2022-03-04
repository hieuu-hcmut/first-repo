import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

def Conv_block(inp, channels, ksize ,st, apply_activation, apply_batchnorm):
    initializer = tf.random_normal_initializer(0., 0.02)
    out = Conv2D(channels, kernel_size=(ksize, ksize) ,strides = (st, st), padding = 'same', kernel_initializer=initializer)(inp)

    if apply_batchnorm:
        out = BatchNormalization()(out)
    
    if apply_activation:
        out = Activation('relu')(out)
    return out

def Residual_Dense_block(x_init, channels):
    ch = channels/4
    f0 = Conv_block(x_init, ch, 3 ,1, True, False)
    c0 = concatenate([x_init, f0])
    f1 = Conv_block(c0, ch, 3,1, True, False)
    c1 = concatenate([x_init, f0, f1])
    f2 = Conv_block(c1, ch, 3, 1, True, False)
    c2 = concatenate([x_init, f0, f1, f2])
    f3 = Conv_block(c2, ch, 3, 1, True, False)
    c3 = concatenate([x_init, f0, f1, f2, f3])
    
    c4 = Conv_block(c3, channels, 1, 1, False, False)
    return c4 + x_init

def Deconv_block(inp, channels,ksize ,st, apply_activation, apply_batchnorm):
    initializer = tf.random_normal_initializer(0., 0.02)
    out = Conv2DTranspose(channels, kernel_size=(ksize, ksize), strides = (st, st), padding = 'same', kernel_initializer=initializer)(inp)

    
    if apply_activation:
        out = LeakyReLU(alpha=0.2)(out)
    
    if apply_batchnorm:
        out = BatchNormalization()(out)
        
    return out

def Res_block(inp, channels):
    x = Conv_block(inp, channels, 3, 1, True, True)
    x = Conv_block(x, channels, 3, 1, True, False)
    return x + inp 

def Generator():
    inp = Input(shape=[256, 320, 3])
    x0 = Conv_block(inp, 32, 3, 2, True, True)
    x0 = Residual_Dense_block(x0, 32)
    x1 = Conv_block(x0, 64, 3, 2, True, True)
    x1 = Residual_Dense_block(x1, 64)
    x2 = Conv_block(x1, 128, 3, 2, True, True)
    x2 = Residual_Dense_block(x2, 128)
    x3 = Conv_block(x2, 256, 3, 2, True, True)
    x3 = Residual_Dense_block(x3, 256)
    x4 = Conv_block(x3, 512, 3, 2, True, True)
    x4 = Residual_Dense_block(x4, 512)
    x5 = Res_block(x4, 512)
    for _ in range(5):
        x5 = Res_block(x5, 512)
  
    x5 = Deconv_block(x5, 512, 3, 1, True, True)
    x5 = concatenate([x5, x4])
    x6 = Deconv_block(x5, 512, 3, 2, True, True)
    x6 = concatenate([x6, x3])
    x7 = Deconv_block(x6, 256, 3, 2, True, True)
    x7 = concatenate([x7, x2])
    x8 = Deconv_block(x7, 128, 3, 2, True, True)
    x8 = concatenate([x8, x1])
    x9 = Deconv_block(x8, 64, 3, 2, True, True)
    x9 = concatenate([x9, x0])
    x10 = Deconv_block(x9, 32, 3, 2, True, True)
    out = Conv2D(3, kernel_size=(3, 3), strides=(1, 1), padding='same')(x10)
    out = BatchNormalization()(out)
    out = Activation('tanh')(out)
    model = Model(inputs=inp, outputs=out)
    return model

def res_block_down(x_init, channels):
    initializer = tf.random_normal_initializer(0., 0.02)
    x0 = Conv2D(channels, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer = initializer)(x_init)
    x = LeakyReLU(alpha=0.2)(x0)
    x = Conv2D(channels, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer = initializer)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    
    return x + x0

def res_block(x_init, channels):
    initializer = tf.random_normal_initializer(0., 0.02)
    x = Conv2D(channels, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer = initializer)(x_init)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(channels, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer = initializer)(x)

    return x + x_init
def Discriminator():
    inp = Input(shape=[256, 320, 3])
    x = res_block_down(inp, 32)
    x = res_block_down(x, 64)
    x = res_block_down(x, 128)
    x = res_block_down(x, 256)
    x = res_block_down(x, 512)
    x = res_block(x, 512)
    x = LeakyReLU(alpha = 0.2)(x)
    x = Flatten()(x)
    x = Dense(units=128)(x)
    x = LeakyReLU(alpha=0.2)(x)
    out = Dense(units=1)(x)
    
    model = Model(inp, out)
    return model
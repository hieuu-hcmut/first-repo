import tensorflow as tf

from src.utils import standardize, SSIM

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False
layer_names = ['block1_pool', 'block2_pool', 'block3_pool','block4_pool', 'block5_pool']
outputs = [vgg.get_layer(name).output for name in layer_names]
vgg_model = tf.keras.Model([vgg.input], outputs)

def vgg19_preprocess(img):
    '''
    Args: An image that all pixels are in range [0, 1]
    Return: preprocessed image for VGG19 input
    '''
    return  tf.keras.applications.vgg19.preprocess_input(standardize(img))

def MAE(x, y):
    '''
    Calculate mean absolute error of x and y
    Return: |x - y|
    '''
    return tf.reduce_mean(tf.abs(x-y))

def perceptual_loss(gen, tar):
    '''
    Calculate content/perceptual loss of two feature maps
    Keyword: Neural Style Transfer

    Args: 
        - gen: generated image from Generator
        - tar: target/ground truth 

    Return:
        - Perceptual loss
    '''
    gen_features = vgg_model(vgg19_preprocess(gen))
    tar_features = vgg_model(vgg19_preprocess(tar))

    loss = 0
    for gen, tar in zip(gen_features, tar_features):
        (B, H, W, C) = gen.shape
        loss += MAE(gen, tar)/(H*W*C)
    return loss

def DSSIM(x, y):
    '''
    Loss function to maximize SSIM
    '''
    return 1 - SSIM(standardize(x), standardize(y))

def tv_loss(x):
    '''
    Calculate total variation loss 
    '''
    (B, H, W, C) = x.shape
    return tf.reduce_sum(tf.image.total_variation(x))/(H*W)

def G_RSGAN(d_fake, d_real):
    '''
    Calculate Relativistic GAN
    Args:
        - d_fake: output of Discriminator for fake image
        - d_real: output of Discriminator for real image 
    '''
    fake_logit = (d_fake - tf.reduce_mean(d_real))
    real_logit = (d_real - tf.reduce_mean(d_fake))
    
    fake_loss = tf.reduce_mean(loss_object(tf.ones_like(d_fake), fake_logit))
    real_loss = tf.reduce_mean(loss_object(tf.zeros_like(d_real), real_logit))

    return fake_loss + real_loss

def D_RSGAN(d_fake, d_real):
    '''
    Calculate Relativistic GAN
    Args:
        - d_fake: output of Discriminator for fake image
        - d_real: output of Discriminator for real image 
    '''
    real_loss = tf.reduce_mean(loss_object(tf.ones_like(d_real), d_real))
    fake_loss = tf.reduce_mean(loss_object(tf.zeros_like(d_fake), d_fake))

    return real_loss + fake_loss

def gradient_penalty(discriminator, real_images, LAMBDA=10):
    '''
    Calculate Gradient Penalty
    Keyword:
    '''
    # tแบกo interplated image
    shape = tf.shape(real_images)
    eps = tf.random.uniform(shape=shape, minval=0., maxval=1.)
    x_mean, x_var = tf.nn.moments(real_images, axes=[0, 1, 2, 3])
    x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region
    noise = 0.5 * x_std * eps  # delta in paper
    
    alpha = tf.random.uniform(shape=[shape[0], 1, 1, 1], minval=-1., maxval=1.)
    interpolated = tf.clip_by_value(real_images + alpha * noise, -1., 1.)
    
    with tf.GradientTape() as grad_tape:
        grad_tape.watch(interpolated)
        logit = discriminator(interpolated, training=True)

    grad = grad_tape.gradient(logit, [interpolated])[0]
    grad_norm = tf.norm(tf.keras.layers.Flatten()(grad), axis=1)
    gp = tf.reduce_mean(tf.square(grad_norm - 1.))
    return gp*LAMBDA

def total_g_loss(d_fake, d_real, gen, tar, loss_weights):
    '''
    Calculate total generator loss
    '''
    mae = MAE(gen, tar)
    tv = tv_loss(gen)
    p_loss = perceptual_loss(gen, tar)
    g_loss = G_RSGAN(d_fake, d_real)
    dssim = DSSIM(gen, tar)
    return mae*loss_weights['mae'] + tv*loss_weights['tv'] + p_loss*loss_weights['vgg'] + g_loss*loss_weights['g_loss'] + dssim*loss_weights['dssim']

def total_d_loss(d_fake, d_real, tar, dicriminator):
    '''
    Calculate total discriminator loss
    '''
    d_loss = D_RSGAN(d_fake, d_real)
    gp = gradient_penalty(dicriminator, tar)
    return d_loss + gp
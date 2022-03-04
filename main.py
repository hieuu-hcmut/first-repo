import os
import time
import tensorflow as tf
import argparse

from datetime import datetime
from src.network import Generator, Discriminator
from src.utils import visualization
from src.losses import total_g_loss, total_d_loss
from src.dataset import KAIST_DATASET

# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#   # Invalid device or cannot modify virtual devices once initialized.
#     pass

generator = Generator()
discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5, beta_2 = 0.999)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.999)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

@tf.function
def train_step(inp, tar, loss_weights):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen = generator(inp, training=True)

        d_fake = discriminator(gen, training = True)
        d_real = discriminator(tar, training = True)
        G_loss = total_g_loss(d_fake, d_real, gen, tar, loss_weights)
        D_loss = total_d_loss(d_fake, d_real, tar, discriminator)
    generator_gradients = gen_tape.gradient(G_loss, generator.trainable_variables) 
    discriminator_gradients = disc_tape.gradient(D_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    return G_loss, D_loss

def fit(args, train_ds, test_ds):
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    f =  open(f'logs/{now}.txt', 'w')

    for x in vars(args):
        line = f'{x}: {getattr(args, x)}\n'
        print(line)
        f.writelines([line])
    
    loss_weights = {'mae': args.mae, 'dssim': args.dssim, 'tv': args.tv, 'g_loss': args.g_loss, 'vgg': args.vgg}

    for epoch in range(args.epochs):
        print(f'Start training epochs {epoch}')
        
        for iter, (inp, tar) in train_ds.enumerate():
            start = time.time()
            G_loss, D_loss = train_step(inp, tar, loss_weights)
            end = time.time()

            line = 'Epoch: %d, iter: %d, G_loss: %.4f, D_loss: %.4f, %.2f sec/it\n' % (epoch, iter, G_loss, D_loss, end-start)
            # if iter*args.batchsize % 500 == 0: 
            print(line)

            f.writelines([line])

        for inp, tar in test_ds.take(1):
            visualization(generator, inp, tar, epoch)

        checkpoint.save(file_prefix = checkpoint_prefix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=1, type=int, help='Batch size')
    parser.add_argument('--epochs', default=3, type=int, help='Number of epoch')
    parser.add_argument('--gp_lambda', default=10, type=int, help='Gradient penalty lambda')
    parser.add_argument('--vgg', default=1000, type=float, help='Weight for perceptual loss')
    parser.add_argument('--mae', default=1, type=float, help='Weight for mae loss')
    parser.add_argument('--g_loss', default=0.01, type=float, help='Weight for g loss')
    parser.add_argument('--tv', default=1, type=float, help='Weight for tv loss')
    parser.add_argument('--dssim', default=1, type=float, help='Weight for dssim')
    parser.add_argument('--height', default=256, type=int, help='Image height')
    parser.add_argument('--width', default=320, type=int, help='Image width')
    parser.add_argument('--seed', default=4, type=int, help='Seed for reproducibility')
    parser.add_argument('--train_file', default='train.txt', type=str, help='text file of training images path')
    parser.add_argument('--test_file', default='test.txt', type=str, help='text file of testing images path')
    args = parser.parse_args()

    assert args.height >= 256, 'Image height must not be lower than 256'
    assert args.width >= 256, 'Image width must not be lower than 256'

    tf.random.set_seed(args.seed)

    print('===========================')
    print('THERMAL INFRARED COLORIZATION')
    print('Author: Quang Tran Huynh Duy')
    print('===========================')

    dataset = KAIST_DATASET(args.batchsize, args.height, args.width, args.train_file, args.test_file)
    train_ds, test_ds = dataset.export_data()
    fit(args, train_ds, test_ds)
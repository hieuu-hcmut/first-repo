import tensorflow as tf
import matplotlib.pyplot as plt 

def standardize(img):
    '''
    Args: An image that all pixels are in range [-1, 1]
    Return: An image that all pixels are in range [0, 1] -> For visualization
    '''
    return img*0.5 + 0.5

def SSIM(x, y):
    '''
    Calculate Structural Similarity Index Measurement of two images, x and y
    '''
    SSIM = tf.reduce_mean(
        tf.image.ssim(
            standardize(x), standardize(y),
            max_val = 1.0, filter_size = 11, filter_sigma = 1.5,
            k1 = 0.01, k2 = 0.03
        )
    )
    return SSIM

def visualization(model, test_input, tar, epoch):
    '''
    Plot 1 sample images at every epoch
    '''
    prediction = model.predict(test_input)
    display_list = [test_input[0], tar[0], prediction[0]]

    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    plt.figure(figsize=(15, 15))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(standardize(display_list[i]))
        plt.axis('off')

    plt.savefig(f'results/Epoch_{epoch}')
    plt.close()
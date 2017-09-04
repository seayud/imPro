"""
 Implementation DEMO

 Applied tflearn high-level library to demo alexnet
 References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.
    - 17 Category Flower Dataset. Maria-Elena Nilsback and Andrew Zisserman.
Links:
    - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
    - [Flower Dataset (17)](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)
 Date: 2017-08-20
"""

from __future__ import division, print_function, absolute_import
from PIL import Image
import tflearn
import tensorflow as tf
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

##import tflearn.datasets.oxflower17 as oxflower17

#X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))

dataset_file ="/home/hp/rice/files.txt"

""" Resize an image.
Arguments:
in_image: `PIL.Image`. The image to resize.
new_width: `int`. The image new width.
new_height: `int`. The image new height.
out_image: `str`. If specified, save the image to the given path.
resize_mode: `PIL.Image.mode`. The resizing mode.
Returns:
`PIL.Image`. The resize image.
"""
def load_image(path):
    image = Image.open(path)
    return image

def resize_image(in_image, new_width, new_height, out_image=None, resize_mode=Image.ANTIALIAS):
    img = in_image.resize((new_width, new_height), resize_mode)
    if out_image:
        img.save(out_image)
        return img
    else:
        return img


""" Convert a PIL.Image to numpy array. """
def pil_to_nparray(pil_image):
    pil_image.load()
    return np.asarray(pil_image, dtype='float32')


# Build the preloader array, resize images to 128x128
from tflearn.data_utils import image_preloader
X, Y = image_preloader(dataset_file, image_shape=(224, 224),   mode='file', categorical_labels=True, normalize=True)

print('Load Data done')
#print(tf.shape(img))
print(tf.shape(X[0]))

# Building 'AlexNet'
network = input_data(shape=[None, 224, 224, 3])
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 3, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training

model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=2)
print(u'开始加载模型')
model.fit(X, Y, n_epoch=200, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=32, snapshot_step=200,
          snapshot_epoch=False, run_id='alexnet_rice3')

model.save('Alexnet_Rice.tflearn')



#Testing
"""
model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=2)
model.load('Alexnet_Rice.tflearn')

print(u'开始预测')
result = model.predict_label(imgs)
print(result)
"""

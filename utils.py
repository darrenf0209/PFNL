# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import cv2
from os.path import exists
import os
from tensorflow.python.layers.convolutional import Conv2D, conv2d
from tensorflow.python.layers.pooling import AveragePooling2D, average_pooling2d
import functools, inspect
import tensorflow.compat.v1 as tf


# This function implements the non-local residual block to compute temporal correlations between a given frame and
# all others
def NonLocalBlock(input_x, out_channels, sub_sample=1, nltype=0, is_bn=False, scope='NonLocalBlock'):
    '''https://github.com/nnUyi/Non-Local_Nets-Tensorflow'''
    batchsize, height, width, in_channels = input_x.get_shape().as_list()
    typedict = {0: 'embedded_gaussian', 1: 'gaussian', 2: 'dot_product', 3: 'concat'}
    with tf.variable_scope(scope) as sc:
        if nltype <= 2:
            with tf.variable_scope('g') as scope:
                g = conv2d(input_x, out_channels, 1, strides=1, padding='same', name='g')
                if sub_sample > 1:
                    g = average_pooling2d(g, pool_size=sub_sample, strides=sub_sample, name='g_pool')

            with tf.variable_scope('phi') as scope:
                if nltype == 0 or nltype == 2:
                    phi = conv2d(input_x, out_channels, 1, strides=1, padding='same', name='phi')
                elif nltype == 1:
                    phi = input_x
                if sub_sample > 1:
                    phi = average_pooling2d(phi, pool_size=sub_sample, strides=sub_sample, name='phi_pool')

            with tf.variable_scope('theta') as scope:
                if nltype == 0 or nltype == 2:
                    theta = conv2d(input_x, out_channels, 1, strides=1, padding='same', name='theta')
                elif nltype == 1:
                    theta = input_x

            g_x = tf.reshape(g, [batchsize, -1, out_channels])
            theta_x = tf.reshape(theta, [batchsize, -1, out_channels])

            # theta_x = tf.reshape(theta, [batchsize, out_channels, -1])
            # theta_x = tf.transpose(theta_x, [0,2,1])
            phi_x = tf.reshape(phi, [batchsize, -1, out_channels])
            phi_x = tf.transpose(phi_x, [0, 2, 1])
            # phi_x = tf.reshape(phi_x, [batchsize, out_channels, -1])

            f = tf.matmul(theta_x, phi_x)
            if nltype <= 1:
                # f_softmax = tf.nn.softmax(f, -1)
                f = tf.exp(f)
                f_softmax = f / tf.reduce_sum(f, axis=-1, keepdims=True)
            elif nltype == 2:
                f = tf.nn.relu(f)  # /int(f.shape[-1])
                f_mean = tf.reduce_sum(f, axis=[2], keepdims=True)
                # print(f.shape,f_mean.shape)
                f_softmax = f / f_mean
            y = tf.matmul(f_softmax, g_x)
            y = tf.reshape(y, [batchsize, height, width, out_channels])
            with tf.variable_scope('w') as scope:
                w_y = conv2d(y, in_channels, 1, strides=1, padding='same', name='w')
                # if is_bn:
                #     w_y = slim.batch_norm(w_y)
            z = w_y  # input_x + w_y
            return z

# TensorFlow wrapper
def tf_scope(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        func_args = inspect.getcallargs(f, *args, **kwargs)
        with tf.variable_scope(func_args.get('scope'), reuse=tf.AUTO_REUSE) as scope:
            return f(*args, **kwargs)
    return wrapper


# Create a directory according to the path if it does not already exist
def automkdir(path):
    if not exists(path):
        os.makedirs(path)


# Stepped learning rate schedule for training
def end_lr_schedule(step):
    if step < 150e3:
        return 1e-4
    elif 150e3 <= step < 170e3:
        return 0.5e-4
    elif 170e3 <= step < 190e3:
        return 0.25e-4
    elif 190e3 <= step < 210e3:
        return 0.1e-4
    elif 210e3 <= step < 230e3:
        return 0.05e-4
    elif 230e3 <= step < 250e3:
        return 0.025e-4
    elif 250e3 <= step < 270e3:
        return 0.01e-4
    elif 270e3 <= step < 290e3:
        return 0.005e-4
    else:
        return 0.0025e-4


# Gaussian kernel for blurring inputs
def gkern(kernlen=13, nsig=1.6):
    import scipy.ndimage.filters as fi
    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen // 2, kernlen // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, nsig)


# Create the blurred filter for input to the network
BLUR = gkern(13, 1.6)  # 13 and 1.6 for x4
BLUR = BLUR[:, :, np.newaxis, np.newaxis].astype(np.float32)

# Downsample an image according to the scale and filter
def DownSample(x, h, scale=4):
    ds_x = tf.shape(x)

    x = tf.reshape(x, [ds_x[0] * ds_x[1], ds_x[2], ds_x[3], 3])

    # Reflect padding
    W = tf.constant(h)

    filter_height, filter_width = 13, 13
    pad_height = filter_height - 1
    pad_width = filter_width - 1

    # When pad_height (pad_width) is odd, we pad more to bottom (right),
    # following the same convention as conv2d().
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_array = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]

    depthwise_F = tf.tile(W, [1, 1, 3, 1])
    # Applies the convolutional filters to each input channel and then concatenates the results
    # Output has input_channels * channel_multiplier channels
    y = tf.nn.depthwise_conv2d(tf.pad(x, pad_array, mode='REFLECT'), depthwise_F, [1, scale, scale, 1], 'VALID')

    ds_y = tf.shape(y)
    y = tf.reshape(y, [ds_x[0], ds_x[1], ds_y[1], ds_y[2], 3])
    return y

# Downsample an image according to the scale and filter. Accommodates a batch size.
def DownSample_4D(x, h, scale=4):
    ds_x = tf.shape(x)

    # Reflect padding
    W = tf.constant(h)

    filter_height, filter_width = 13, 13
    pad_height = filter_height - 1
    pad_width = filter_width - 1

    # When pad_height (pad_width) is odd, we pad more to bottom (right),
    # following the same convention as conv2d().
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_array = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]

    depthwise_F = tf.tile(W, [1, 1, 3, 1])
    y = tf.nn.depthwise_conv2d(tf.pad(x, pad_array, mode='REFLECT'), depthwise_F, [1, scale, scale, 1], 'VALID')

    ds_y = tf.shape(y)
    y = tf.reshape(y, [ds_x[0], ds_y[1], ds_y[2], 3])
    print("Shape of Downsample is {}".format(y.shape))
    return y


# Convert an image from RGB to YCBCR formatting
def _rgb2ycbcr(img, maxVal=255):
    O = np.array([[16],
                  [128],
                  [128]])
    T = np.array([[0.256788235294118, 0.504129411764706, 0.097905882352941],
                  [-0.148223529411765, -0.290992156862745, 0.439215686274510],
                  [0.439215686274510, -0.367788235294118, -0.071427450980392]])

    if maxVal == 1:
        O = O / 255.0

    t = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    t = np.dot(t, np.transpose(T))
    t[:, 0] += O[0]
    t[:, 1] += O[1]
    t[:, 2] += O[2]
    ycbcr = np.reshape(t, [img.shape[0], img.shape[1], img.shape[2]])

    return ycbcr


# Convert to uint8
def to_uint8(x, vmin, vmax):
    x = x.astype('float32')
    x = (x - vmin) / (vmax - vmin) * 255  # 0~255
    return np.clip(np.round(x), 0, 255)


# Initialise weight distribution
he_normal_init = tf.keras.initializers.VarianceScaling(
    scale=2.0, mode='fan_in', distribution='truncated_normal', seed=None
)


# Rearranges data from depth into blocks of spatial data
def depth_to_space_3D(x, block_size):
    ds_x = tf.shape(x)
    x = tf.reshape(x, [ds_x[0] * ds_x[1], ds_x[2], ds_x[3], ds_x[4]])

    y = tf.depth_to_space(x, block_size)

    ds_y = tf.shape(y)
    x = tf.reshape(y, [ds_x[0], ds_x[1], ds_y[1], ds_y[2], ds_y[3]])
    return x


# Save image with RGB formatting
def cv2_imsave(img_path, img):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)


# Load image with RGB formatting
def cv2_imread(img_path):
    img = cv2.imread(img_path)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    return img



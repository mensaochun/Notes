"""
This source code is for convolutinal neural network
Author:Mensaochun
Date:2017.01.16
"""
# import used packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# load mnist data for testing
mnist_data = input_data.read_data_sets("/home/pisme/mensaochun/mnistdata", one_hot=True)
x_train = mnist_data.train.images  # shape:[55000,784]
y_train = mnist_data.train.labels  # shape:[55000,10]
x_test = mnist_data.test.images  # shape:[10000,784]
y_test = mnist_data.test.labels  # shape:[10000,10]

# Parameters setting
img_height = 28
img_width = 28
img_flatten = img_width * img_height
n_classes = 10
learning_rate = 0.01
n_iterations = 10000
batch_size = 1


# Difine function add_conv_layer() for adding convolutional layer(pooling layer may be contained)
def add_conv_layer(input, filter_shape, pooling=True, activation_function=tf.nn.relu):
    """
    Parameters description:
    input shape:[batch, in_height, in_width, in_channels]
    filter_shape:[filter_height, filter_width, in_channels, out_channels]
    In conv2d module:
    strides and padding are set to be fixed as [1, 1, 1, 1],"SAME"
    In pooling module:
    ksize and strides
    """
    conv_strides = [1, 1, 1, 1]
    conv_padding = "SAME"
    pooling_strides = [1, 2, 2, 1]
    pooling_ksize = [1, 2, 2, 1]
    pooling_padding = "SAME"
    # Define weight and bias
    weight = tf.Variable(tf.random_normal(filter_shape, 0, 0.1), dtype=tf.float32)
    bias = tf.Variable(tf.random_normal([filter_shape[3]], dtype=tf.float32))
    layer = activation_function(
        tf.nn.conv2d(input=input, filter=weight, strides=conv_strides, padding=conv_padding) + bias)
    # Pooling
    if pooling is True:
        output = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=pooling_padding)
    else:
        output = layer
    return output, weight


def flatten_layer(input):
    """
    Parameters description:
     input shape:[batch, height, width, channels]

    """
    shape = tf.shape(input)
    height, width, channels = shape[1:4]
    output = tf.reshape(input, [-1, height * width * channels])
    return output


def add_fully_connect_layer(input, out_size, activation_function=None):
    """
    Parameters description:
    input shape:[batch, height*width*channels]

    """
    in_size = tf.shape(input)[1]
    weight = tf.Variable(tf.random_normal([in_size, out_size], stddev=0.1), dtype=tf.float32)
    bias = tf.Variable(tf.constant(0.1, shape=[out_size]), dtype=tf.float32)
    output = tf.matmul(input, weight) + bias
    if activation_function is None:
        return output
    else:
        return activation_function(output)


def accuracy(softmax, y_true):
    is_equal = tf.equal(tf.arg_max(softmax, dimension=1), tf.arg_max(y_true, dimension=1))
    is_equal_flaot32 = tf.cast(is_equal, tf.float32)
    return tf.reduce_mean(is_equal_flaot32)


# Define placeholder and reshape it to Input shape:[batch, in_height, in_width, in_channels]
x_p = tf.placeholder(dtype=tf.float32, shape=[None, img_flatten])
x_p = tf.reshape(x_p, [-1, img_height, img_width, 1])
y_p = tf.placeholder(dtype=tf.float32, shape=[None, n_classes])

# Construct convNet
layer1 = x_p
layer2 = add_conv_layer(layer1, [5, 5, 1, 16])
layer3 = add_conv_layer(layer2, [5, 5, 16, 32])
layer3_flatten = flatten_layer(layer3)
layer4 = add_fully_connect_layer(layer3_flatten, 128, activation_function=tf.nn.sigmoid)
layer5 = add_fully_connect_layer(layer4, 10, activation_function=None)

# Define cost
softmax = tf.nn.softmax(layer5)
cost = tf.reduce_mean(tf.reduce_sum(-y_p * tf.log(softmax), axis=1))

# Difine optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Init all variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(n_iterations):
        x_batch, y_batch = mnist_data.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x_p: x_batch, y_p: y_batch})
        if i % 200 == 0:
            print "iteration", i, "loss:", sess.run(cost, feed_dict={x_p: x_train, y_p: y_train})
    print "Optimizer done!"
    print("Test accuracy:", sess.run(accuracy, feed_dict={x_p: x_test, y_p: y_test}))

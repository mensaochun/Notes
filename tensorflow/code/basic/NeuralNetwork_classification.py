"""
This script is for Neural Network classification
google's deep learning frame tensorflow is used
Author:Mensaochun
Date:2017.01.15
"""
# Import some used packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# Load mnist data, here only training and test data is used
data = input_data.read_data_sets("/home/pisme/mensaochun/mnistdata", one_hot=True)
x_train = data.train.images
y_train = data.train.labels
x_test = data.test.images
y_test = data.test.labels

# Define parameters
learning_rate = 0.1
n_iterations = 1000000
image_size = 28
image_flaten_size = 28 * 28
n_classes = 10
batch_size = 50

# Define add_layer
def add_layer(input, input_size, output_size, activation_function=None):
    W = tf.Variable(tf.random_normal([input_size, output_size]))
    b = tf.Variable(tf.random_normal([output_size]))
    if activation_function is None:
        return tf.matmul(input, W) + b
    else:
        return activation_function(tf.matmul(input, W) + b)

# Define placeholder
x_p = tf.placeholder(dtype=tf.float32, shape=[None, image_flaten_size])
y_p = tf.placeholder(dtype=tf.float32, shape=[None, n_classes])

# Construct layers
layer1 = x_p
layer2 = add_layer(layer1, image_flaten_size, 100, tf.nn.sigmoid)
layer3 = add_layer(layer2, 100, 100, tf.nn.sigmoid)
layer4 = add_layer(layer3, 100, 100, tf.nn.sigmoid)
layer5 = add_layer(layer4, 100, 100, tf.nn.sigmoid)
layer6 = add_layer(layer5, 100, n_classes)

# Define cost
softmax = tf.nn.softmax(layer6)
cross_entropy = tf.reduce_mean(tf.reduce_sum(-y_p * tf.log(softmax), axis=1))
cost = cross_entropy

# Define optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Difine accuracy
def accuracy(softmax_prediction, y_true):
    Is_correct = tf.equal(tf.argmax(softmax_prediction, axis=1), tf.argmax(y_true, axis=1))
    accuracy = tf.reduce_mean(tf.cast(Is_correct, tf.float32))
    return accuracy

# Init all variables
init = tf.global_variables_initializer()

# Training
loss=[]
with tf.Session() as sess:
    sess.run(init)
    for i in range(n_iterations):
        batch_x, batch_y = data.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x_p: batch_x, y_p: batch_y})
        if i % 500 == 0:
            loss_n_stepSize=sess.run(cost, feed_dict={x_p: x_train, y_p: y_train})
            loss.append(loss_n_stepSize)
            print "Iteration:", i, "Loss:", loss_n_stepSize
    print "Training Done!"
    print "Training accuracy:",sess.run(accuracy(softmax,y_train),feed_dict={x_p: x_train,y_p:y_train})
    print "Test accuracy:    ", sess.run(accuracy(softmax,y_test),feed_dict={x_p: x_test,y_p:y_test})
    print"Plot loss:"
plt.plot(loss)
plt.show()
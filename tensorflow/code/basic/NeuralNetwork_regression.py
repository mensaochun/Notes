"""
This python script is for constructing neural_network_regression
Google's deep learning frame tensorflow is used here
Author:Mensaochun
Date:2017.01.15
"""
# import relative packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
learning_rate = 0.01
n_iterations = 50000
# Ceate a toy data example
x = np.linspace(-10, 10, 1000)[:, np.newaxis]
noise = np.random.normal(loc=0.0, scale=0.1, size=x.shape)
y = np.sin(x) + noise

# Visualize the data
plt.plot(x, y)
plt.show()

# Define add_layer
def add_layer(input, input_size, output_size, activation_function=None):
    W = tf.Variable(tf.random_normal([input_size, output_size]))
    b = tf.Variable(tf.random_normal([output_size]))
    if activation_function is None:
        return tf.matmul(input, W) + b
    else:
        return activation_function(tf.matmul(input, W) + b)


# Define placeholder
x_p = tf.placeholder(dtype=tf.float32, shape=[None, 1])
y_p = tf.placeholder(dtype=tf.float32, shape=[None, 1])

# Construct layers
layer1 = x_p
layer2 = add_layer(layer1, 1, 100, tf.nn.sigmoid)
layer3 = add_layer(layer2, 100, 100, tf.nn.sigmoid)
layer4 = add_layer(layer3, 100, 100, tf.nn.sigmoid)
layer5 = add_layer(layer4, 100, 100, tf.nn.sigmoid)
prediction = layer6 = add_layer(layer5, 100, 1)

# Define cost
cost = tf.reduce_mean((prediction - y_p) ** 2)

# Define optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Init all variables
init = tf.global_variables_initializer()

# Training
with tf.Session() as sess:
    sess.run(init)
    for i in range(n_iterations):
        sess.run(optimizer, feed_dict={x_p: x, y_p: y})
        if i % 100 == 0:
            print "Iteration:", i, "Loss:", sess.run(cost, feed_dict={x_p: x, y_p: y})
    # Visualize the result
    plt.plot(x, y)
    plt.plot(x, sess.run(prediction, feed_dict={x_p: x, y_p: y}))
    plt.show()

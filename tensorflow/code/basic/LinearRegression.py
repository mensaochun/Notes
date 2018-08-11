"""
This script is for linear regression
google's deep learning frame tensorflow is used
Author:Mensaochun
Date:2017.01.15
"""

# load relative packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set parameters
learning_rate = 0.01
n_iterations = 1000

# Create a toy data example
x = np.linspace(-10, 10, num=300)[:, np.newaxis]
noise = np.random.normal(loc=0, scale=2, size=[300, 1])
y = -2 * x + noise

# Plot the created line
plt.figure()
plt.plot(x, y)
plt.show()
plt.close()

# Define placeholder
xp = tf.placeholder(dtype=tf.float32, shape=[None, 1])
yp = tf.placeholder(dtype=tf.float32, shape=[None, 1])

# Define weights and bias
weights = tf.Variable(initial_value=tf.random_normal([1, 1], mean=0, stddev=0.1, dtype=tf.float32))
bias = tf.Variable(initial_value=tf.random_normal([1, 1], mean=0, stddev=0.1), dtype=tf.float32)

# define cost
prediction = xp * weights + bias
cost = tf.reduce_mean((prediction - yp) ** 2)

# define optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# define init
init = tf.global_variables_initializer()

# Training
with tf.Session() as sess:
    sess.run(init)
    for i in range(n_iterations):
        sess.run(optimizer, feed_dict={xp: x, yp: y})
        if i % 50 == 0:
            print "Iteration", i, "Training loss:", sess.run(cost, feed_dict={xp: x, yp: y})
    print("Traing done!\nNow plot:")
    plt.figure()
    plt.plot(x, y)
    plt.plot(x, sess.run(prediction, feed_dict={xp: x, yp: y}))
    plt.show()

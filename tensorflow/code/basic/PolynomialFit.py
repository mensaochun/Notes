"""
This script is for polynomic fitting
tensorflow is used
Author:Mensaochun
Date:2017.01.15
"""

# import used packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set parameters
learning_rate = 0.0001
n_orders = 3  # number of polynomic orders
n_iterations = 30000
n_datapoints = 400  # the number of data points in the created data
lamb = 0  # penalty on weights

# # Create a toy data1 example
# x = np.linspace(-2, 2, n_datapoints)[:,np.newaxis]
# noise = np.random.normal(loc=0, scale=0.1, size=x.shape)
# y = 20*(np.sin(5*x) + noise)

# Create a toy data2 example
x = np.linspace(-5, 5, n_datapoints)[:, np.newaxis]
noise = np.random.normal(loc=0, scale=0.1, size=x.shape)
y = x ** 2 + x + 5 * noise

# Visualize the created data
plt.figure()
plt.plot(x, y)
plt.show()
plt.close()

# Prepare data:x's shape:[None,1]==>x_new's shape:[None,n_orders]
def modify_data(x, n_orders):
    x = np.reshape(x, [n_datapoints, ])
    x_new = np.zeros((n_datapoints, n_orders))
    for i in range(n_orders):
        x_new[:, i] = np.power(x, i + 1)
    return x_new

# Transform x to x_new
x_new = modify_data(x, n_orders)

# Difine placeholder
xp = tf.placeholder(dtype=tf.float32, shape=[None, n_orders])
yp = tf.placeholder(dtype=tf.float32, shape=[None, 1])

# Define weight and bias
weight = tf.Variable(initial_value=tf.random_normal([n_orders, 1], mean=0, stddev=0.1, dtype=tf.float32))
bias = tf.Variable(initial_value=tf.constant(1, dtype=tf.float32))

# Define cost
prediction = tf.matmul(xp, weight) + bias
cost = tf.reduce_mean((prediction - yp) ** 2) + lamb * tf.reduce_sum(weight ** 2)

# Init all variables
init = tf.global_variables_initializer()

# Define optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Training
with tf.Session() as sess:
    sess.run(init)
    for i in range(n_iterations):
        sess.run(optimizer, feed_dict={xp: x_new, yp: y})
        if i % 100 == 0:
            print "Iteration", i, "loss:", sess.run(cost, feed_dict={xp: x_new, yp: y})
    print "Training Done!"
    # plot original data and polynomic fitting result
    plt.figure()
    plt.plot(x, y)
    plt.plot(x, sess.run(prediction, feed_dict={xp: x_new}))
    plt.show()
    plt.close

"""neural network for regression prediction"""

# load relative packages
from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# prepare data
# data coming from matlab must be transformed to python numpy type
filename = u'/home/pisme/mensaochun/data2.mat'
data = sio.loadmat(filename)
# Note:transform original uint8 datatype to float32 type
# x_train
X_train = data["train_x"]
X_train = X_train.tolist()
X_train = np.array(X_train, dtype='float32')
X_train = X_train.reshape((67, -1))
# y_train
y_train = data["train_label"]
y_train = y_train.tolist()
y_train = np.array(y_train, dtype='float32')

# x_test
X_test = data["test_x"]
X_test = X_test.tolist()
X_test = np.array(X_test, dtype='float32')
X_test = X_test.reshape((10, -1))

# y_test
y_test = data["test_label"]
y_test = y_test.tolist()
y_test = np.array(y_test, dtype='float32')


def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 765])
ys = tf.placeholder(tf.float32, [None, 1])
# add 2 hidden layers
l1 = add_layer(xs, 765, 10, activation_function=tf.sigmoid)
l2 = add_layer(l1, 10, 10, activation_function=tf.sigmoid)
# add output layer
prediction = add_layer(l2, 10, 1, activation_function=None)

# the error between prediciton and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# define init
init = tf.global_variables_initializer()

# start graph
# init variable,containing weights and biases
sess = tf.Session()
sess.run(init)
write_loss = []
# iteration,to update weights and biases
for i in range(10000):
    # training
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train})
    if i % 100 == 0:
        write_loss.append(sess.run(loss, feed_dict={xs: X_train, ys: y_train}))
        # to see the step improvement
        print("Iteration", i, "Training loss:", sess.run(loss, feed_dict={xs: X_train, ys: y_train}))
print("Training is done!")

result = sess.run(prediction, feed_dict={xs: X_test})
print("Prediction values:\n", result.T)
print("Real values:\n", y_test.T)
print("plot:")
plt.plot(write_loss,'o')
plt.xlabel("100*num_of_iterations")
plt.ylabel("loss")
plt.show()
sess.close

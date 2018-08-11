import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

# load data
from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets("/home/pisme/mensaochun/mnistdata", one_hot=True)
X_train = data.train.images  # 55000*784
y_train = data.train.labels  # 55000*10
X_test = data.test.images
y_test = data.test.labels

# set parameters
batch_size = 100
learning_rate = 0.05
n_iterations = 50000

# define placeholder
Xp = tf.placeholder(dtype=tf.float32, shape=[None, 784])
yp = tf.placeholder(dtype=tf.float32, shape=[None, 10])

# define variable
Weights = tf.Variable(tf.random_normal([784, 10], mean=0.0, stddev=0.1))
biases = tf.Variable(tf.random_normal(shape=[1, 10], mean=0.0, stddev=0.1))

# define model
Xw_add_b = tf.matmul(Xp, Weights) + biases

# define loss
softmax = tf.nn.softmax(Xw_add_b)
cross_entropy = -tf.reduce_sum(yp * tf.log(softmax), axis=1)
cost = tf.reduce_mean(cross_entropy)

# define optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# define init
init = tf.global_variables_initializer()

# define accuracy
Is_correct = tf.equal(tf.argmax(softmax, axis=1), tf.argmax(yp, axis=1))
accuracy = tf.reduce_mean(tf.cast(Is_correct, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for i in range(n_iterations):
        x_batch, y_batch = data.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={Xp: x_batch, yp: y_batch})
        if i % 200 == 0:
            print("iteration", i,
                  "training accuracy:", sess.run(accuracy, feed_dict={Xp: X_train, yp: y_train}),
                  "loss:", sess.run(cost, feed_dict={Xp: X_train, yp: y_train})
                  )
    print "Optimizer done!"
    print("Test accuracy:", sess.run(accuracy, feed_dict={Xp: X_test, yp: y_test}))
    w = sess.run(Weights)
# plot weights
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.title(i)
    plt.imshow(np.reshape(w[:, i], [28, 28]),cmap='seismic')
plt.show()

import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# define neural network
def layer(X,in_size,out_size, activation_function=None):
    with tf.name_scope("layer"):
        with tf.name_scope("Weights"):
            W=tf.Variable(tf.random_normal((in_size,out_size)))
        with tf.name_scope("biases"):
            b=tf.Variable(tf.zeros((1,out_size))+0.1)
        if activation_function is None:
            output=tf.matmul(X,W)+b
        else:
            output=activation_function(tf.matmul(X,W)+b)
        return output
# define placeholder
with tf.name_scope("Inputs"):
    with tf.name_scope("x_inputs"):
        xs = tf.placeholder(dtype=tf.float32, shape=[None, 1],name="Encoded_molecular_one-hot_type")
    with tf.name_scope("y_inputs"):
        ys = tf.placeholder(dtype=tf.float32, shape=[None, 1],name="Conductance")

# construct neural network
l1=layer(xs,1,10,activation_function=tf.nn.sigmoid)
l2=layer(l1,10,10,activation_function=tf.nn.sigmoid)
l3=layer(l2,10,10,activation_function=tf.nn.sigmoid)
#l4=layer(l3,10,10,activation_function=tf.nn.sigmoid)
with tf.name_scope("outputs"):
    with tf.name_scope("prediction"):
        prediction=layer(l3,10,1,activation_function=None)
    with tf.name_scope("loss"):
        loss=tf.reduce_mean(tf.reduce_sum(tf.square(prediction-ys),axis=1))

# define init
init=tf.global_variables_initializer()
# start graph
sess=tf.Session()
sess.run(init)

# tf.train.SummaryWriter soon be deprecated, use following
writer = tf.summary.FileWriter("/home/pisme/mensaochun/", sess.graph)
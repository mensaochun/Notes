import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import matplotlib.pyplot as plt

W=tf.get_variable(name='W',shape=[1,1], dtype=tf.float32,initializer=tf.random_normal_initializer())
b=tf.get_variable(name='b',shape=[1], dtype=tf.float32,initializer=tf.constant_initializer())

x_data=np.arange(0,100).reshape([100,1])
y_data=2*x_data+1+np.random.randn(100,1)*5
plt.plot(x_data,y_data,c='b')
plt.scatter(x_data,y_data,c='r',s=3)
# plt.show()

x=tf.placeholder(dtype=tf.float32,shape=[100,1])
y=tf.placeholder(dtype=tf.float32,shape=[100,1])
out=tf.matmul(x,W)+b
loss=tf.reduce_mean(tf.square(out-y))

optimizer=tf.train.GradientDescentOptimizer(0.0002).minimize(loss)

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess=tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.run(init)
    for i in range(10000):
        sess.run(optimizer,feed_dict={x:x_data,y:y_data})
        loss_=sess.run(loss,feed_dict={x:x_data,y:y_data})
        if i%100==0:
            print "loss", loss_
    out_=sess.run(out,feed_dict={x:x_data,y:y_data})
    plt.plot(x_data,out_, c='b')
    plt.scatter(x_data, out_, c='r', s=3)
    plt.show()




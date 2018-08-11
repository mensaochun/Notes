import tensorflow as tf
import numpy as np
c1=tf.constant(((1,2),(3,4)))
sess=tf.Session()
c2=sess.run(c1)
print c2

#Variable
v=tf.Variable(0)

for _ in range(3):
    v=v+1

init=tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)
print sess.run(v)

sess.close()

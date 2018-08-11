import tensorflow as tf

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(learning_rate=starter_learning_rate,
                                           global_step=global_step,
                                           decay_steps=1000,
                                           decay_rate=0.9,
                                           staircase=True)
# y=x**2+2x+1
x = tf.get_variable('x', shape=[1], initializer=tf.constant_initializer(value=2))
y = tf.square(x) + tf.multiply(2., x) + 1.
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss=y, global_step=global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        sess.run(optimizer)
        x_, y_, learning_rate_, global_step_ = sess.run([x, y, learning_rate, global_step])
        print 'iter', i, 'x', x_, "y:", y_, "learning_rate:", learning_rate_, 'global step:', global_step_

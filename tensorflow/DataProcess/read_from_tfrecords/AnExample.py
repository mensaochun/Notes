# coding=utf-8
import numpy
import tensorflow as tf

writer = tf.python_io.TFRecordWriter('test.tfrecord')
for i in range(0, 2):
    a = 0.618 + i
    b = [2016 + i, 2017 + i]
    c = numpy.array([[0, 1, 2], [3, 4, 5]]) + i
    c = c.astype(numpy.uint8)
    c_raw = c.tostring()
    example = tf.train.Example(
        features=tf.train.Features(feature={'a': tf.train.Feature(float_list=tf.train.FloatList(value=[a])),
                                            'b': tf.train.Feature(int64_list=tf.train.Int64List(value=b)),
                                            'c': tf.train.Feature(bytes_list=tf.train.BytesList(value=[c_raw]))}))
    serialized = example.SerializeToString()
    writer.write(serialized)
writer.close()

# output file name string to a queue
filename_queue = tf.train.string_input_producer(['test.tfrecord'], num_epochs=None)
# create a reader from file queue
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
# get feature from serialized example

features = tf.parse_single_example(serialized_example, features={'a': tf.FixedLenFeature([], tf.float32),
                                                                 'b': tf.FixedLenFeature([2], tf.int64),
                                                                 'c': tf.FixedLenFeature([], tf.string)})

a_out = features['a']
b_out = features['b']
c_raw_out = features['c']
c_out = tf.decode_raw(c_raw_out, tf.uint8)
c_out = tf.reshape(c_out, [2, 3])

a_batch, b_batch, c_batch = tf.train.shuffle_batch([a_out, b_out, c_out], batch_size=3,
                                                   capacity=200, min_after_dequeue=100, num_threads=2)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

tf.train.start_queue_runners(sess=sess)
a_val, b_val, c_val = sess.run([a_batch, b_batch, c_batch])


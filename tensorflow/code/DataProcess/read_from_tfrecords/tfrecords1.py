# coding=utf-8
import os

import tensorflow as tf
from PIL import Image


def create_tfrecords_cls(data_dir, class_names, path_to_save_tfrecords):
    """
    Description:
        This func for classification
    Args:
        cwd: data folder, with multiple class folders
        classes: class names
        tfrecords_path_to_save: path to save tfrecords
    Return:
        None
    """
    writer = tf.python_io.TFRecordWriter(path_to_save_tfrecords)
    for index, name in enumerate(class_names):
        class_path = os.path.join(data_dir, name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            print 'img_name:', img_name
            img = Image.open(img_path)
            # convert img to binary file
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
    writer.close()
    return


def create_tfrecords_reg(data_dir, class_names, path_to_save_tfrecords):
    """
    Description:
        This func for regression
    Args:
        cwd: data folder, with multiple class folders
        classes: class names
        tfrecords_path_to_save: path to save tfrecords
    Return:
        None
    """
    # TODO
    writer = tf.python_io.TFRecordWriter(path_to_save_tfrecords)
    for index, name in enumerate(class_names):
        class_path = os.path.join(data_dir, name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            print 'img_name:', img_name
            img = Image.open(img_path)
            # convert img to binary file
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            tf.train.Feature()
            writer.write(example.SerializeToString())
    writer.close()
    return


def decode(tfrecords_path, swd):
    filename_queue = tf.train.string_input_producer([tfrecords_path])  # 读入流中
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # 取出包含image和label的feature对象
    # tf.decode_raw可以将字符串解析成图像对应的像素数组
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [200, 200, 3])
    label = tf.cast(features['label'], tf.int64)
    with tf.Session() as sess:  # 开始一个会话
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # 启动多线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(40):
            example, l = sess.run([image, label])  # 在会话中取出image和label
            img = Image.fromarray(example, 'RGB')  # 这里Image是之前提到的
            img.save(swd + '/' + str(i) + '_''Label_' + str(l) + '.jpg')  # 存下图片
            print 'read_records:', l
            # print(example, l)
        coord.request_stop()
        coord.join(threads)


def get_nextbatch(swd, tfrecords_path, image_size):
    """
    Args:
        swd:
        tfrecords_path:
        image_size:
    Returns:

    """
    filename_queue = tf.train.string_input_producer([tfrecords_path])  # 读入流中
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # 取出包含image和label的feature对象
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, image_size)
    label = tf.cast(features['label'], tf.int64)

    # 组合batch
    batch_size = 4
    mini_after_dequeue = 100
    capacity = mini_after_dequeue + 3 * batch_size

    example_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, capacity=capacity)

    with tf.Session() as sess:  # 开始一个会话
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(10):  # 10个batch
            example, l = sess.run([example_batch, label_batch])  # 取出一个batch
            for j in range(batch_size):  # 每个batch内4张图
                sigle_image = Image.fromarray(example[j], 'RGB')
                sigle_label = l[j]
                print (swd + '/batch_' + str(i) + '_' + 'size' + str(j) + '_' + 'Label_' + str(sigle_label) + '.jpg')
                sigle_image.save(swd + '/batch_' + str(i) + '_' + 'size' + str(j) + '_' + 'Label_' + str(
                    sigle_label) + '.jpg')  # 存下图片
                # print(example, l)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    # data path
    cwd = './data'
    # classes
    classes = {'forky': 1, 'knifey': 2}
    tfrecords_path_to_save = '/home/pi/stone/Notes/tensorflow/TFRecords/tfrecords/mydata.tfrecords'
    create_tfrecords_cls(data_dir=cwd, class_names=classes, path_to_save_tfrecords=tfrecords_path_to_save)
    swd = '/home/pi/stone/Notes/tensorflow/TFRecords/img2'
    # decode(tfrecords_path=tfrecords_path_to_save, swd=swd)
    get_nextbatch(swd=swd, tfrecords_path=tfrecords_path_to_save, image_size=[200, 200, 3])

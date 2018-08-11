# coding=utf-8
import os

import tensorflow as tf
from PIL import Image


def create_records(cwd, classes, record_path_to_save):
    writer = tf.python_io.TFRecordWriter(record_path_to_save)
    for index, name in enumerate(classes):
        class_path = os.path.join(cwd, name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path)
            img = img.resize((128, 128))
            # convert image to bytes
            img_raw = img.tobytes()
            # plt.imshow(img)
            # plt.show()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
    writer.close()


def read_and_decode(filename, cwd):
    # read records, and generate a queue
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    # return filename and file
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={'label': tf.FixedLenFeature([], tf.int64),
                                                                     'img_raw': tf.FixedLenFeature([], tf.string), })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [128, 128, 3])
    # 在流中抛出img张量
    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    # 在流中抛出label张量
    label = tf.cast(features['label'], tf.int32)
    with tf.Session() as sess:  # 开始一个会话
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(1):
            example, l = sess.run([img, label])  # 在会话中取出image和label
            img = Image.fromarray(example, 'RGB')  # 这里Image是之前提到的
            img.save(cwd + '/' + str(i) + '_''Label_' + str(l) + '.jpg')  # 存下图片
            print(example, l)
        coord.request_stop()
        coord.join(threads)
        # return img, label


if __name__ == '__main__':
    cwd = './data/knifey-spoony'
    classes = ['forky', 'knifey', 'spoony']
    record_path_to_save = "./data/knifey-spoony/tfrecord/knifey-spoony.tfrecords"

    create_records(cwd=cwd, classes=classes, record_path_to_save=record_path_to_save)
    read_and_decode(filename=record_path_to_save, cwd='./data/knifey-spoony/save_img')

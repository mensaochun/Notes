#  tensorflow笔记

[TOC]

## tensorflow程序的pipeline

以下以图片为例子。

数据部分：

1. 考虑网络的输入是否是相同大小的图像，如果是，要进行图像的长宽比分布的调查，然后确定最终需要resize的大小。
2. 数据resize到一定的大小之后，进行数据增强。（先resize后增强还是先增强再resize这个需要再考虑。）
3. 数据预处理：计算图像的均值（一般是三通道的均值），将图片减均值。
4. 储存数据：三种方式，1.存在内存中。2.存在硬盘中，batch读取。3.制备成tfrecord的数据。根据具体情况而定。
5. 制作批量读取数据的工具。

模型部分：

1. 定义模型的输入，也就是placeholder。
2. 定义inference的模型，也就是前向模型。
3. 定义训练模型，也就是定义损失函数和优化器。
4. 定义validation。

> 定义模型的时候需要注意一些细节：
>
> a. Batchnorm需要区分训练和测试。在测试的时候需要用到滑动平均和滑动variance，这个不是模型参数。测试的时候怎么弄？恢复模型的时候怎么弄？
>
> b. Dropout需要区分训练和测试。
>
> c. 学习率、batch_norm有一个global_step参数，需要将其进行保存，恢复训练的时候需要使用。目前只知道通过ckpt来进行恢复，还有其他办法吗？
>
> d. 注意区分train variable（模型参数）和global variable。







## 安装

**cuda** 

目前安装cuda8.0。详细安装教程看caffe-install。

**cudann**

使用cudann v6。

tensorflow 1.3以上就只能支持cudann v6，因此目前安装cudann v6。

到英伟达官网下载cudann v6。

下载玩之后，解压缩。可以看到一个include和lib问价夹，里面分别装了头文件和动态库文件。

头文件和动态库直接复制到相应位置：

~~~bash
cd cuda/include/
sudo cp cudnn.h /usr/local/cuda/include/
cd ../lib64
sudo cp lib* /usr/local/cuda/lib64/
~~~

改变头文件和动态库的权限。

~~~bash
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
~~~

建立软链接。

需要根据不同的cudann版本进行相应的修改。实际上，只有libcudnn.so.6.0.21是动态库文件。其他两个都是软链接。

~~~bash
cd /usr/local/cuda/lib64/
sudo rm -rf libcudnn.so libcudnn.so.6
sudo ln -s libcudnn.so.6.0.21 libcudnn.so.6
sudo ln -s libcudnn.so.6 libcudnn.so
sudo ldconfig
~~~

## docker

1. 下载tensorflow镜像，并且创建自己的容器。

   ~~~shell
   # -it：在命令行中运行，否则会在后台运行，不会在shell中显示。bash是指定了在bash运行，也可以在notebook上运行。
   nvidia-docker run --name yourc_tf -it tensorflow/tensorflow:latest-gpu bash
   # 注意nvidia-docker是gpu版本的。

   # 以下命令，从飞哥的镜像中直接创建容器。而且挂载了宿主的目录到容器中。
   nvidia-docker run --name yourc_tf -it -v /home/yourc/docker:/root/docker zhuwf/tf/faster_rcnn:0 bash

   # 如果想要带有tensorboard，那么就需要进行端口映射。将docker中的6006端口映射到k80主机上的6006端口。
   nvidia-docker run --name yourc_tf2 -p 6006:6006 -it -v /home/yourc/docker:/root/docker zhuwf/tf/faster_rcnn:0 bash 

   # 在docker中运行tensorboard命令。
   tensorboard --logdir /root/stone
   # 在本地浏览器中输入k80的ip地址和端口
   192.168.16.6:6006

   # 安装pytorch
   nvidia-docker run --name yourc_pytorch2 -p 6006:6006 -it -v /home/yourc/docker:/root/docker nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04 bash  
   ~~~

2. 查看镜像和容器

   ~~~shell
   docker ps -a(包括没有启动的容器)
   docker images
   ~~~

3. 启动容器，关闭容器，删除容器

   ~~~shell
    # 是否需要用nvidia-docker?
    docker start -i yourc_tf(加上-i才能在shell上运行)
    docker stop 
    docker rm
   ~~~

4. 进入已经启动的容器

   ~~~shell
   docker exec -it yourc_tf bash
   ~~~

5. 建立软链接

   ~~~shell
   ln -s source dist        # 建立软连接
   ~~~

6. 在docker中安装软件。

   在使用docker容器时，有时候里边没有安装vim，敲vim命令时提示说：vim: command not found，这个时候就需要安装vim，可是当你敲apt-get install vim命令时，提示：

   ```shell
   Reading package lists... Done
   Building dependency tree       
   Reading state information... Done
   E: Unable to locate package vim
   ```
    这时候需要敲：apt-get update，这个命令的作用是：同步 /etc/apt/sources.list 和 /etc/apt/sources.list.d 中列出的源的索引，这样才能获取到最新的软件包。等更新完毕以后再敲命令：apt-get install vim命令即可。

7. 将主机的目录挂载到容器内，实现数据共享。

   `Docker`可以支持把一个宿主机上的目录挂载到镜像里。
   命令如下:

   ~~~shell
   docker run -it -v /home/dock/Downloads:/usr/Downloads ubuntu64 /bin/bash
   ~~~

   通过-v参数，冒号前为宿主机目录，必须为绝对路径，冒号后为镜像内挂载的路径。

8. 拷贝数据

   但是对这三种方法我都不太喜欢，无意间看到另位一种方法供大家参考：

   从主机复制到容器`sudo docker cp host_path containerID:container_path`

   从容器复制到主机`sudo docker cp containerID:container_path host_path`

   容器ID的查询方法想必大家都清楚:`docker ps -a`



## 复制变量

https://stackoverflow.com/questions/33717772/how-can-i-copy-a-variable-in-tensorflow

## flags

http://blog.csdn.net/lyc_yongcai/article/details/73456960

1. 命令行参数通过如下定义。

   ~~~python
   # 比如定义checkpoint_path的路径。
   tf.app.flags.DEFINE_string(
       'checkpoint_path', None,
       'The path to a checkpoint from which to fine-tune.')
   ~~~

2. 命令行参数获取

   ~~~python
   FLAGS = tf.app.flags.FLAGS
   FLAGS.checkpoint_path
   ~~~

3. 通过tf.app.run()解析命令行参数，调用main 函数。注意：1.命令行参数只能在这个main函数里才能进行调用。2.这里需要自己去定义一个main函数。

   ~~~python
   def main(unused_argv):  
       pass
   ~~~

   ​

## 数据准备

tensorflow读取数据有三种方式：

- Feeding: Python code provides the data when running each step（提供数据文件的路径，在线读取）
- Reading from files: an input pipeline reads the data from files at the beginning of a TensorFlow graph（制作成tfrecoder）
- Preloaded data: a constant or variable in the TensorFlow graph holds all the data (for small data sets)(载入到内存中再进行读取)

### 1.直接读进内存

### 

### 2.TFRecord

http://blog.csdn.net/best_coder/article/details/70146441

TensorFlow高效读取数据之tfrecord详细解读:http://blog.csdn.net/qq_16949707/article/details/53483493

是否是list才要指定具体的长度？

Tensorflow中使用tfrecord方式读取数据：http://blog.csdn.net/u010358677/article/details/70544241

Tfrecords Guide：https://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/



- 另外一个系列：



TensorFlow 队列与多线程的应用：http://blog.csdn.net/chaipp0607/article/details/72924572 

TensorFlow TFRecord数据集的生成与显示：http://blog.csdn.net/chaipp0607/article/details/72960028

TensorFlow 组合训练数据（batching）：http://blog.csdn.net/chaipp0607/article/details/73016068

根据这个系列整理的代码

~~~python
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
    create_records(cwd=cwd, classes=classes, tfrecords_path_to_save=tfrecords_path_to_save)
    swd = '/home/pi/stone/Notes/tensorflow/TFRecords/img2'
    # decode(tfrecords_path=tfrecords_path_to_save, swd=swd)
    get_nextbatch(swd=swd, tfrecords_path=tfrecords_path_to_save)
~~~

## Saver类

### Saver的背景介绍

我们经常在训练完一个模型之后希望保存训练的结果，这些结果指的是模型的参数，以便下次迭代的训练或者用作测试。Tensorflow针对这一需求提供了Saver类。Saver类提供了向checkpoints文件保存和从checkpoints文件中恢复变量的相关方法。Checkpoints文件是一个二进制文件，它把变量名映射到对应的tensor值 。只要提供一个计数器，当计数器触发时，Saver类可以自动的生成checkpoint文件。这让我们可以在训练过程中保存多个中间结果。例如，我们可以保存每一步训练的结果。为了避免填满整个磁盘，Saver可以自动的管理Checkpoints文件。例如，我们可以指定保存最近的N个Checkpoints文件。

### Saver的实例

下面以一个例子来讲述如何使用Saver类

~~~python
import tensorflow as tf  
import numpy as np  
  
x = tf.placeholder(tf.float32, shape=[None, 1])  
y = 4 * x + 4  
  
w = tf.Variable(tf.random_normal([1], -1, 1))  
b = tf.Variable(tf.zeros([1]))  
y_predict = w * x + b  
  
loss = tf.reduce_mean(tf.square(y - y_predict))  
optimizer = tf.train.GradientDescentOptimizer(0.5)  
train = optimizer.minimize(loss)  
  
isTrain = False  
train_steps = 100  
checkpoint_steps = 50  
checkpoint_dir = ''  
  
saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b  
x_data = np.reshape(np.random.rand(10).astype(np.float32), (10, 1))  
  
with tf.Session() as sess:  
    sess.run(tf.initialize_all_variables())  
    if isTrain:  
        for i in xrange(train_steps):  
            sess.run(train, feed_dict={x: x_data})  
            if (i + 1) % checkpoint_steps == 0:  
                saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=i+1)  # 这个可以恢复？
    else:  
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  
        if ckpt and ckpt.model_checkpoint_path:  
            saver.restore(sess, ckpt.model_checkpoint_path)  
        else:  
            pass  
        print(sess.run(w))  
        print(sess.run(b))  
~~~


- isTrain：用来区分训练阶段和测试阶段，True表示训练，False表示测试

- train_steps：表示训练的次数，例子中使用100
- checkpoint_steps：表示训练多少次保存一下checkpoints，例子中使用50
- checkpoint_dir：表示checkpoints文件的保存路径，例子中使用当前路径

### 训练阶段

使用Saver.save()方法保存模型：

- sess：表示当前会话，当前会话记录了当前的变量值
- checkpoint_dir + 'model.ckpt'：表示存储的文件名
- global_step：表示当前是第几步

训练完成后，当前目录底下会多出5个文件。

![1](./pics/1.png)

打开名为“checkpoint”的文件，可以看到保存记录，和最新的模型存储位置

![2](/home/pi/stone/Notes/tensorflow/note/pics/2.png)

### 测试阶段

测试阶段使用saver.restore()方法恢复变量：

1. sess：表示当前会话，之前保存的结果将被加载入这个会话
2. ckpt.model_checkpoint_path：表示模型存储的位置，不需要提供模型的名字，它会去查看checkpoint文件，看看最新的是谁，叫做什么。

运行结果如下图所示，加载了之前训练的参数w和b的结果

![3](./pics/3.png)



### 恢复部分模型

恢复部分模型，需要在定义saver的时候就指定需要恢复的变量。如下所示：

~~~python
# 获取全部变量
self.all_variables = tf.global_variables()
# 定义restorer，默认恢复所有变量。这里指定只恢复部分变量。
self.restorer = tf.train.Saver(self.all_variables[:-6], max_to_keep=None)
# 定义saver，默认恢复所有变量。这里指定恢复全部变量，也可以恢复部分变量。
self.saver = tf.train.Saver(self.all_variables, max_to_keep=None)
~~~



参考：https://stackoverflow.com/questions/42217320/restore-variables-that-are-a-subset-of-new-model-in-tensorflow



##  tf.train.batch和tf.train.shuffle_batch的理解

tf.train.batch和tf.train.shuffle_batch的理解：http://blog.csdn.net/ying86615791/article/details/73864381

How tf.train.shuffle_batch works：https://stackoverflow.com/questions/45203872/how-tf-train-shuffle-batch-works

What's going on in tf.train.shuffle_batch and `tf.train.batch?：https://stackoverflow.com/questions/43028683/whats-going-on-in-tf-train-shuffle-batch-and-tf-train-batch

## deconvolution

反卷积请看:http://blog.csdn.net/mao_xiao_feng/article/details/71713358

反卷积是需要指定out_shape的,因为需要通过same和valid来指定.

https://github.com/vdumoulin/conv_arithmetic

## conv Valid 和Same

参考博客：[http://www.cnblogs.com/willnote/p/6746668.html](http://www.cnblogs.com/willnote/p/6746668.html)

### 图示说明

![padding操作图示](https://upload-images.jianshu.io/upload_images/5723738-b556db7333087ed0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 用一个3x3的网格在一个28x28的图像上做切片并移动
- 移动到边缘上的时候，如果不超出边缘，3x3的中心就到不了边界
- 因此得到的内容就会缺乏边界的一圈像素点，只能得到26x26的结果
- 而可以越过边界的情况下，就可以让3x3的中心到达边界的像素点
- 超出部分的矩阵补零

问题：1. 如果卷积核的大小是偶数，这个时候卷积核的中心还怎么定？这个时候进行same的padding怎么算？不过这种情况很少，暂时不考虑这个。

注意：这里的same不是指padding之后，再进行卷积得到的图像的大小和之前一样。

padding的数目的计算方法：卷积核/2，结果向下取整。

计算卷积过后feature map的大小

1. 先对feature map进行padding
2. 将宽w-F，也就是预留出一个卷积核大小的位置。
3. （W-F）/s计算出来一个数值，这个数值可能为整数，这个时候（W-F）/s+1就是结果，如果这个数值是小数，将这个小数向下取整，得到（W-F）/s+1向下取整的结果。

## reshape

reshape这个函数，会将tensor按照最后一个维度最快的方式进行展开成一维。

然后再将这个一维的向量以最后一个维度最快的方式进行排列。

### 代码说明

根据tensorflow中的conv2d函数，我们先定义几个基本符号

- 输入矩阵 W×W，这里只考虑输入宽高相等的情况，如果不相等，推导方法一样，不多解释。
- filter矩阵 F×F，卷积核
- stride值 S，步长
- 输出宽高为 new_height、new_width

在Tensorflow中对padding定义了两种取值：VALID、SAME。下面分别就这两种定义进行解释说明。

### VALID

```
new_height = new_width = (W – F + 1) / S  #结果向上取整
```

- 含义：new_height为输出矩阵的高度
- 说明：VALID方式不会在原有输入矩阵的基础上添加新的值，输出矩阵的大小直接按照公式计算即可

### SAME

```
new_height = new_width = W / S    #结果向上取整
```

- 含义：new_height为输出矩阵的高度
- 说明：对W/S的结果向上取整得到W"包含"多少个S

```
pad_needed_height = (new_height – 1)  × S + F - W
```

- 含义：pad_needed_height为输入矩阵需要补充的高度
- 说明：因为new_height是向上取整的结果，所以先-1得到W可以完全包裹住S的块数，之后乘以S得到这些块数的像素点总和，再加上filer的F并减去W，即得到在高度上需要对W补充多少个像素点才能满足new_height的需求

```
pad_top = pad_needed_height / 2    #结果取整
```

- 含义：pad_top为输入矩阵上方需要添加的高度
- 说明：将上一步得到的pad_needed_height除以2作为矩阵上方需要扩充0的像素点数

```
pad_bottom = pad_needed_height - pad_top
```

- 含义：pad_bottom为输入矩阵下方需要添加的高度
- 说明：pad_needed_height减去pad_top的剩余部分补充到矩阵下方

以此类推，在宽度上需要pad的像素数和左右分别添加的像素数为

```
pad_needed_width = (new_width – 1)  × S + F - W
pad_left = pad_needed_width  / 2    #结果取整
pad_right = pad_needed_width – pad_left
```



## tensorboard

1.网络可视化

通过name_scope或者variable_scope将一些数据包起来，作为一个模块来显示。

命令其实很简单，只需要两句

~~~python
sess=tf.Session()
# 将session中的图进行保存
writer = tf.summary.FileWriter(logdir="/home/mensaochun/mensaochun/graph", sess.graph)
~~~

通过log_dir指定event存放的位置，在运行python文件，再terminal中输入以下命令就可以进行可视化了。

~~~shell
tensorboard --logdir="/home/mensaochun/mensaochun/graph/"
~~~

注意，在新版本的tensorflow中`--log_dir`已经改成了`--logdir`

2.查看数据

精简为四步骤：1.添加这个要观察的变量。2.合并这些要观察的变量。3.建立writer，用于写入结果。4.运行这些变量，得到结果。5.将结果写入到writer中。

- 查看image信息

比如我们以手写字体为例子。下面代码定义了输入。

~~~python
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
~~~

然后，进行reshape之后就可以进行可视化了。如果是三通道的图片，把最后一个维度改为3就好了。10表示指定需要可视化多少张图片。

~~~python
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)
~~~

- 查看scalar信息，histogram信息和distribution信息

可以写一个函数来封装需要看的tensor的相关参数信息，比如：scalar信息，histogram信息和distribution信息等。。。

~~~python
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      # 计算参数的均值，并使用tf.summary.scaler记录
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      # 计算参数的标准差
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      # 用直方图记录参数的分布
      tf.summary.histogram('histogram', var)
~~~

- 合并信息

只要调用一个merge命令就可以合并所有信息

~~~python
# summaries合并
merged = tf.summary.merge_all()
~~~

- 建立writer

~~~python
# 写到指定的磁盘路径中
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir + '/test')
~~~

- 运行summary，将运行结果加入到writer中。

在建立了writer之后，要添加观察的变量就要通过add



~~~python
for i in range(max_steps):
    if i % 10 == 0:  # 记录测试集的summary与accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:  # 记录训练集的summary
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
  train_writer.close()
  test_writer.close()
~~~







- 使用tf.summary.scalar记录标量

http://blog.csdn.net/sinat_33761963/article/details/62433234

## 远程可视化

远程可视化，需要在服务器和客户端都安装OpenSSH.

windows安装：在这里下载https://www.mls-software.com/opensshd.html

安装一路回车。通过在命令行中输入ssh查看是否安装好。

linux安装openssh-server

~~~shell
sudo apt-get install openssh-server
~~~

在客户端连接服务器的时候，输入

~~~shell
ssh -L 16006:127.0.0.1:6006 username@server_ip
~~~

然后在客户端的浏览器中输入

~~~python
http://127.0.0.1:16006/
~~~

就可以了。

## tf.Variable(), tf.get_variable()

参考这篇文章，可以说讲得非常详细了：http://blog.csdn.net/qq_22522663/article/details/78729029

有两种方式可以创建变量，一种是tf.variable()，另外一种是tf.get_variable()

前者不需要指定变量名字，系统会自动安排；后者则要手动指定名字。当在同一个variable_scope下存在

这时候问题就来了，若在函数中通过tf.get_variable()定义了一个variable，当多次调用该函数的时候就出现问题，因为variable的名字就只有一个。而tf.variable()应该不存在这个问题，每次调用函数的时候，变量的名字都会重新命名。

但是，这样又有一个问题，变量怎么重用？

若是tf.variable()，好像没办法重复使用，但是tf.get_variable()通过定义tf.variable_scope()可以指定变量范围，比如model/weights,再指定重用就好了。

详细见官网，讲得很详细。https://www.tensorflow.org/programmers_guide/variables

## tf.name_scope, tf.variable_scope

几个注意的问题：

1.variable.name和variable.op.name有什么区别？

参考链接：https://stackoverflow.com/questions/34727792/whats-difference-between-variable-name-and-variable-op-name

2.name_scope和variable_scope的区别？

主要有两点：

- name_scope 一定影响op的名字，一定影响tf.variable，不影响tf.get_variable。


- variable_scope 影响tf.variable和tf.get_variable的名字，同时影响op的名字。

tf.get_variable()会忽略name_scope的作用，而tf.variable()则不会。看下面的例子：

~~~python
with tf.variable_scope("foo"):
    with tf.name_scope("bar"):
        v=tf.Variable(1.0,name='v')
        x = 1.0 + v
print v.name
print x.op.name

#输出的结果
foo/bar/v:0
foo/bar/add

with tf.variable_scope("foo"):
    with tf.name_scope("bar"):
        v = tf.get_variable("v", [1])
        # v=tf.Variable(1.0,name='v')
        x = 1.0 + v

print v.name
print x.op.name
#输出的结果
foo/v:0
foo/bar/add
~~~

3.关于变量重用

http://blog.csdn.net/qq_19918373/article/details/69499091



## padding中的valid和same

参考这篇文章：

http://blog.csdn.net/wuzqchom/article/details/74785643



##  tf.nn.sparse_softmax_cross_entropy_with_logits

http://blog.csdn.net/john_xyz/article/details/61211422




## tensorflow1.4安装出错

import tensorflow 出错libcudnn.so.6: cannot open shared object file: No such file or directory

解决方法：

https://hk.saowen.com/a/f567da5f37bab03970cad4dc11d366036779269a0cf6510868797623c5ac3845

cudnn下载：

https://developer.nvidia.com/rdp/cudnn-download

选择：Download cuDNN v6.0 (April 27, 2017), for CUDA 8.0→cuDNN v6.0 Library for Linux

**Note**:是否与caffe发生冲突？

## 设置GPU设备

**在Python代码中指定GPU**

~~~python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
~~~

如果有多块GPU,直接设置

~~~python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
~~~

参考文章:http://blog.csdn.net/guvcolie/article/details/77164230

## 自适应学习率

在模型的初期的时候，往往设置为较大的学习速率比较好，因为距离极值点比较远，较大的学习速率可以快速靠近极值点；而，后期，由于已经靠近极值点，模型快收敛了，此时，采用较小的学习速率较好，较大的学习速率，容易导致在真实极值点附近来回波动，就是无法抵达极值点。

在tensorflow中，提供了一个较为友好的API,

~~~python
tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)
~~~

其数学表达式是这样的：

$$decayed\_learning\_rate = learning\_rate \times decay\_rate ^{ (global\_step / decay\_steps)}$$

先解释API中的参数的意思，第一个参数`learning_rate`即初始学习速率，第二个参数，是用来计算步骤的，每调用一次优化器，即自增1，第三个参数`decay_steps`通常设为一个常数，如数学公式中所示，与第五个参数配合使用效果较好，第五个参数`staircase`如果设置为`True`，那么指数部分就会采用整除策略，表示每`decay_step`，学习速率变为原来的`decay_rate`，至于第四个参数`decay_rate`表示的是学习速率的下降倍率。

```python
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
```

意思就是，初始的学习速率是0.1，每经过1000次训练后，学习速率变为原来的0.9

参考文章: https://www.cnblogs.com/crackpotisback/p/7105748.html



## Dropout

dropout在训练的时候设置为(0,1)之间的数字,但是一般在测试阶段,是不进行dropout的,因此设置为0?因此,最好将dropout设置为一个placeholder,作为可变的网络输入.可以这么做:

~~~python
keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
~~~





## Debug

ref：

官网：https://www.tensorflow.org/programmers_guide/debugger

民间：https://www.jianshu.com/p/9fd237c7fda3

### 打印变量

~~~python
def eval_variable(self, layer_name, var_name):
        with tf.variable_scope(layer_name, reuse=True):
            var = tf.get_variable(var_name)
            out = self.sess.run(var, feed_dict={self.model.X: self.batch_X, self.model.Y: self.batch_Y})
        if out.ndim == 2:
            for i in range(out.shape[0]):
                for j in range(out.shape[1]):
                    print out[i, j], " "
        if out.ndim == 3:
            print out
            # return out
~~~

## Batch normalization

Batch Normalization在tensorflow中有各种版本，请看stackoverflow上的一个总结：

[What is right batch normalization function in Tensorflow?](https://stackoverflow.com/questions/48001759/what-is-right-batch-normalization-function-in-tensorflow/48006315#48006315)

[tf.layers.batch_normalization](https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization)这个op用的比较多。但是在用的时候注意坑。

1. [TensorFlow的batch normalization层中存在的坑](https://byjiang.com/2017/11/26/TensorFlow_BN_Layer/)
2. [tf.layers.batch_normalization large test error](https://stackoverflow.com/questions/43234667/tf-layers-batch-normalization-large-test-error)

The operations which `tf.layers.batch_normalization` adds to update mean and variance don't automatically get added as dependencies of the train operation - so if you don't do anything extra, they never get run. (Unfortunately, the documentation doesn't currently mention this. I'm opening an issue about it.)

Luckily, the update operations are easy to get at, since they're added to the `tf.GraphKeys.UPDATE_OPS` collection. Then you can either run the extra operations manually:

```
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
sess.run([train_op, extra_update_ops], ...)
```

Or add them as dependencies of your training operation, and then just run your training operation as normal:

```
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_op = optimizer.minimize(loss)
...
sess.run([train_op], ...)
```

在tensorflow/models中的做法:

~~~python
def batch_norm(inputs, training, data_format):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  return tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)
~~~





## 出错

1.error when import tensorflow:ImportError: libcudnn.so.5: cannot open shared object file: No such file or directory

Had the same problem it was solved by :

~~~shell
sudo ldconfig /usr/local/cuda/lib64
~~~

ref：https://github.com/tensorflow/tensorflow/issues/7522
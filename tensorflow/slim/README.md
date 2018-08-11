# slim

参考：

1. tensorflow/models

   https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim

2. tensorflow/contrib...

   https://github.com/tensorflow/models/tree/master/research/slim

3. 使用TensorFlow-Slim进行图像分类

   https://blog.csdn.net/lijiancheng0614/article/details/77727445

## 简要概括

slim主要包括以下一些模块。

> - [arg_scope](https://www.tensorflow.org/code/tensorflow/contrib/framework/python/ops/arg_scope.py): 除name_scope和variable_scope之外建立的名字空间，arg就是arguments的意思。在这个参数scope之内，特定的函数可以采用某些默认的参数。
> - [data](https://www.tensorflow.org/code/tensorflow/contrib/slim/python/slim/data/): contains TF-slim's [dataset](https://www.tensorflow.org/code/tensorflow/contrib/slim/python/slim/data/dataset.py) definition, [data providers](https://www.tensorflow.org/code/tensorflow/contrib/slim/python/slim/data/data_provider.py), [parallel_reader](https://www.tensorflow.org/code/tensorflow/contrib/slim/python/slim/data/parallel_reader.py), and [decoding](https://www.tensorflow.org/code/tensorflow/contrib/slim/python/slim/data/data_decoder.py) utilities.
> - [evaluation](https://www.tensorflow.org/code/tensorflow/contrib/slim/python/slim/evaluation.py): contains routines for evaluating models.
> - [layers](https://www.tensorflow.org/code/tensorflow/contrib/layers/python/layers/layers.py): contains high level layers for building models using tensorflow.
> - [learning](https://www.tensorflow.org/code/tensorflow/contrib/slim/python/slim/learning.py): contains routines for training models.
> - [losses](https://www.tensorflow.org/code/tensorflow/contrib/losses/python/losses/loss_ops.py): contains commonly used loss functions.
> - [metrics](https://www.tensorflow.org/code/tensorflow/contrib/metrics/python/ops/metric_ops.py): contains popular evaluation metrics.
> - [nets](https://www.tensorflow.org/code/tensorflow/contrib/slim/python/slim/nets/): contains popular network definitions such as [VGG](https://www.tensorflow.org/code/tensorflow/contrib/slim/python/slim/nets/vgg.py) and [AlexNet](https://www.tensorflow.org/code/tensorflow/contrib/slim/python/slim/nets/alexnet.py) models.
> - [queues](https://www.tensorflow.org/code/tensorflow/contrib/slim/python/slim/queues.py): provides a context manager for easily and safely starting and closing QueueRunners.
> - [regularizers](https://www.tensorflow.org/code/tensorflow/contrib/layers/python/layers/regularizers.py): contains weight regularizers.
> - [variables](https://www.tensorflow.org/code/tensorflow/contrib/framework/python/ops/variables.py): provides convenience wrappers for variable creation and manipulation. 包装了模型变量和非模型变量。

## 定义模型

### 1.变量

原生的tensorflow中变量可以分为几种，regular变量，local变量。regular变量可以保存到硬盘当中，regular变量不能（注意这里变量不一定是tf中的variable）。

TF中的变量更加细分：模型变量，非模型变量，比如global step、moving average。

在代码中的区别：

~~~python
# model变量可以训练，regular变量不可以训练。
# Model Variables
weights = slim.model_variable('weights',
                              shape=[10, 10, 3 , 3],
                              initializer=tf.truncated_normal_initializer(stddev=0.1),
                              regularizer=slim.l2_regularizer(0.05),
                              device='/CPU:0')
model_variables = slim.get_model_variables()

# Regular variables
my_var = slim.variable('my_var',
                       shape=[20, 1],
                       initializer=tf.zeros_initializer())
regular_variables_and_model_variables = slim.get_variables()
~~~

### 2.layers

slim提供了不同功能的layer，比如卷积就有不同类型的卷积，使调用更简单。

> 重复层可以使用slim.repeat。
>
> 重复层，但是层的参数不一样的话，那么可以用slim.stack。

### 3.scope

除了name_scope, variable_scope，slim中还加入了一种更好用的，叫做arg_scope。这个可以减少相同函数的参数。在arg_scope下，参数可以覆盖。

## 训练模型

### 1.loss

多任务loss可以直接通过`get_total_loss`来获得。

如果有自己定义的loss也很方便的加到总的loss中。

~~~python
pose_loss = MyCustomLossFunction(pose_predictions, pose_labels)
slim.losses.add_loss(pose_loss) # Letting TF-Slim know about the additional loss.
~~~

### 2.训练循环

一旦定义了模型，损失和优化器，就可以通过`slim.learning.create_train_op` 和`slim.learning.train`

进行优化。

~~~python
g = tf.Graph()

# Create the model and specify the losses...
...
total_loss = slim.losses.get_total_loss()
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

# create_train_op ensures that each time we ask for the loss, the update_ops
# are run and the gradients being computed are applied too.
train_op = slim.learning.create_train_op(total_loss, optimizer)
logdir = ... # Where checkpoints are stored.

slim.learning.train(
    train_op,
    logdir,
    number_of_steps=1000,
    save_summaries_secs=300,
    save_interval_secs=600):
~~~

## 微调模型

### 1.在模型的名字相同的情况下微调

在tensorflow中进行微调的做法：1. 定义一个`restorer=tf.train.Saver()`，这个是对所有的变量都进行微调。也可以通过`restorer = tf.train.Saver([v1, v2])`进行微调部分变量。2.将session和ckpt传进去。`restorer.restore(sess, "/tmp/model.ckpt")`，这个session计算进行了初始化，可以继续训练了。

~~~python
# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# Add ops to restore all the variables.
restorer = tf.train.Saver()

# Add ops to restore some variables.
restorer = tf.train.Saver([v1, v2])

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  restorer.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Do some work with the model
  ...
~~~

### 2.部分模型微调

### 3.在模型名字不同的情况下进行微调

### 4.不同任务上微调

## 评估模型

### 1.评估参数

### 2.评估循环


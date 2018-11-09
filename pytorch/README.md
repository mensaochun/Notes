## 和numpy的对比

1. pytorch的tensor和numpy共享内存。改变一个就会改变另外一个。b=a.numpy()，a和b共享内存。b=torch.from_numpy(a)。互相转换的方法。

2. 可以随意定义tensor在cpu上还是在cuda上。

3. _方法代表是in-place的。


## tensor

1. 基本操作和numpy很像
2. 可以将numpy的数据转为touch的数据类型
3. 将数据移动到cuda

## Tensor和Variable的区别

http://www.pytorchtutorial.com/2-2-variable/

对Variable进行操作会构建计算图，以便在方向传播中使用到。

Tensor则不会。

## Autograd

### 1.variable

variable主要包含data，grad和grad_fn这三个东西。

如果variable是自己创建的，那么这时候就没有fn这个函数。

如果variable是一个scale，那么在反向传播的时候没有必要指定元素。

如果variable不是一个scale则要指定梯度来计算梯度。这个是什么意思？

## Sequential

这个实际上就是将多个module搞在了一起，把它当成Module就可以了。

通过add_module添加module。

## ModuleList

ModuleList对象可以append所有Module模块。在forword的时候直接根据索引调用。

~~~python
# 建立ModuleList列表
model=ModuleList()
# 将需要的模块添加到这个列表中
model.append(some_module)
# 一键将这个列表中的操作搬到cuda上去，如果用list的话是没有这个功能的。
model.cuda()
~~~

问题：为什么不直接用list来做这个事情呢？

答：但是直接使用列表不能将网络中相应的部分放入到cuda中去。 

参考：https://blog.csdn.net/daniaokuye/article/details/78827436

## 定义网络

1. 先定义好各种前向函数。
2. 注意要清理梯度。


## 网络可视化

pytorch没有工具直接支持网络的可视化，但是有第三方的方法来支持。目前有两种方法。

### PyTorchViz

参考github上的说明：https://github.com/szagoruyko/pytorchviz

以及两篇博客：

1.https://blog.csdn.net/gyguo95/article/details/78821617

2.https://blog.csdn.net/yanshuai_tek/article/details/79262981

### tensorboard

参考这篇博客：https://blog.csdn.net/sunqiande88/article/details/80155925

如果出现错误，参考以下解决方法：

1.AttributeError: module 'torch.jit' has no attribute 'get_trace_graph'解决：

https://github.com/lanpa/tensorboard-pytorch/issues/122

2.注意输入要加 Variable


# PaddlePaddle笔记

## tensorflow与paddle对比

graph和program是一个非常像的东西。

首先了解tensorflow中graph和session的关系：

https://blog.csdn.net/xg123321123/article/details/78017997?utm_source=blogxgwz4

然后了解paddle中exe和program之间的关系：

program会分为初始化program和前向和反向传播program。

最后对比paddle中program和tensorflow中graph：

https://blog.csdn.net/PaddlePaddle/article/details/80812082
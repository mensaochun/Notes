# 题目

```
stone: 使用C++，实现一个double类型的双向链表。
要求：
1. 双向链表需要实现：
    1. 构造函数，空，以及用double指针和元素个数。
    2. append功能，将一个double数据，插入在链表末端。
    3. get_forword_item(i，N),get_backward_item(i,N).从第i个数据， 向前或者向后取n个数据，并且将N个数据返回。
    4.  insert(a, i)将数据a，插入在第i个元素位置上
    5. show（），打印链表所有数据。
2. 对于链表类，使用doxygen生产接口的注释，并生成html的文档
3. 使用gtest对链表类做单元测试。
4. 在main.cpp实现主函数，实例化2个双向链表，完成以下操作。
    1. 链表1，append数据（1，2，3，4，5），get_forward_item(3,7),show()
    2. 链表2，实例化（6，7，8，9），get_backward(0,3),insert(1,3),show()
5. 用CMake，对整个工程进行管理，将双向链表编译成为一个动态链接库，enable测试，且将主函数编译成可执行文件。
```


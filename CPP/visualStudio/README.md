# C++笔记

## =default关键字

下面这6个函数都可以跟着`=default`关键字.

```c++
class A{
public:
    A() = default; // Default constructor
    A(const A&) = default; // Copy constructor
    A(A&&) = default; // Move constructor(since C++11)
    A& operator=(const A&) = default; // Copy assignment operator
    A& operator=(A&&) = default; // Move assignment operator (since C++11)
    ~A() = default; // Destructor
};
```
`=default`关键字的作用：显式的要求编译器生成函数的一个默认版本。

参考：https://www.jianshu.com/p/f964b929f2bc

## 宏定义

说白了，就是用一个字符串来表示一个标识符。

具体参考这篇文章：http://blog.chinaunix.net/uid-21372424-id-119797.html



## 结构体中的成员类型不能是该结构体本身



## 回调函数

TODO



## 声明，定义，初始化，赋值

**引用性声明**不分配存储空间，如`extern int x`; 只是告诉编译器x是整形，已经在其它地方定义了。
**定义**是在内存中确定变量的位置、大小。
**初始化**是定义变量时候赋给变量的值（从无到有）。
**赋值**是以后用到该变量，赋给该变量新的值。

~~~c++
int i;//定义
extern int i; //声明
int i=9;//初始化
i= 7;// 赋值
~~~



##  #ifndef #define #endif

在一个大的软件工程里面，可能会有多个文件同时包含一个头文件，当这些文件编译链接成一个可执行文件时，就会出现大量重定义的错误。在头文件中实用`#ifndef #define #endif`能避免头文件的重定义。
方法：例如要编写头文件test.h
在头文件开头写上两行：

~~~c++
# ifndef _TEST_H
# define _TEST_H//一般是文件名的大写
...
# endif
~~~

头文件结尾写上一行：`#endif`这样一个工程文件里同时包含两个test.h时，就不会出现重定义的错误了。
**分析**：当第一次包含test.h时，由于没有定义TEST_H，条件为真，这样就会包含（执行）`#ifndef _TEST_H`和`#endif`之间的代码，当第二次包含`test.h`时前面一次已经定义了TEST_H，条件为假，`#ifndef _TEST_H`和`#endif`之间的代码也就不会再次被包含，这样就避免了重定义了。主要用于防止**重复定义宏**和**重复包含头文件**。**也可以防止重复定义变量？**



## 将代码引入到工程中

参考教你快速将大量代码文件加入到VS项目中：http://www.cjjjs.com/paper/xmkf/201641716212844.aspx

## 快捷键

1.自动提示：ctrl + shift +空格

2.代码补全：ctrl + j

## 报错信息

生成解决方案的时候出现以下错误

~~~c++
1>main.obj : error LNK2019: 无法解析的外部符号 _av_register_all，该符号在函数 _main 中被引用
1>E:\code\src\QtGuiApplication1\Win32\Debug\\QtGuiApplication1.exe : fatal error LNK1120: 1 个无法解析的外部命令
~~~

说明只引用了头文件，但是相应的库没有加载进去。因此要指定相应的lib文件。

## 添加库文件注意事项

如果是动态库，有两种情况，一种是lib文件只是指定了运行的时候dll文件的位置。另外一种是直接在运行的时候指定dll。

如果是静态库，那么lib文件中则包括了所有代码内容。

## 在代码中添加库文件中的注意事项

只能在c文件中加入，不要再h文件加入，否则多次导入

## 本地调试器

什么是本地调试器？

## 视频播放器FFmpeg相关内容

你这个代码中用到了FFmpeg库，那么编译的时候需要该库相应的头文件，链接的时候需要静态lib文件，而运行的时候需要相应的dll文件，这和直接用播放器能打开视频没什么关系。

建议你下载个FFmpeg编译好的库，需要哪些文件就复制到你可执行文件的输出目录中。

## 动态链接库和静态链接库


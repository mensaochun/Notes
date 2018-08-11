# linux下的动态库和静态库

[TOC]

# on linux

## 1.动态库的生成

![1](./pics/1.png)

![2](./pics/2.png)

![3](./pics/3.png)

![4](./pics/4.png)

![5](./pics/5.png)



![6](./pics/6.png)

![7](./pics/7.png)

代码示例

~~~C++
//example.cpp
#include <stdio.h>
#include "example.h"
int example(int a, int b)
{
	printf("example library: a=%d, b=%d \n", a, b);
	return 0;
}
//example.h
int example(int a, int b);
~~~



## 2.动态库的使用

![8](./pics/8.png)

![9](./pics/9.png)



![10](./pics/10.png)



注意-L.中的点表示当前目录。

**注意两点**：

1. 在链接时候，一定要指定需要链接的动态库，像这样：`-lexample`，否则不知道你需要链接什么库，会报错。但是如果动态库已经放在（`/lib, /usr/lib, /usr/local/lib`或者`LD_LIBRAYR_PATH`环境变量）里，则不用指定动态库的具体位置。



2. 在程序运行的时候，若动态库已经放在动态库已经放在（/lib, /usr/lib,/usr/local/lib或者LD_LIBRAYR_PATH环境变量）里，则不用指定动态库的位置。


![11](./pics/11.png)

![12](./pics/12.png)

![13](./pics/13.png)

![14](./pics/14.png)

## 3.在makefile中生成动态库

![14](./pics/15.png)

![14](./pics/16.png)



![14](./pics/17.png)

~~~makefile
######### 标准Makefile Lv1.1 / 生成动态库 ########
EXE=libexample.so
SUBDIR=src 
CXX_SOURCES =$(foreach dir,$(SUBDIR), $(wildcard $(dir)/*.cpp))
CXX_OBJECTS=$(patsubst  %.cpp, %.o, $(CXX_SOURCES))
DEP_FILES  =$(patsubst  %.o,  %.d, $(CXX_OBJECTS))

$(EXE): $(CXX_OBJECTS)
	g++ -shared $(CXX_OBJECTS) -o $(EXE)	
%.o: %.cpp
	g++  -c -fPIC -MMD $<  -o  $@
-include $(DEP_FILES)
clean: 
	rm  -rf  $(CXX_OBJECTS)  $(DEP_FILES)  $(EXE)
test:
	echo $(CXX_OBJECTS)
~~~



![14](./pics/18.png)

![14](./pics/19.png)





##4.库的标准目录结构

![14](./pics/20.png)

![14](./pics/21.png)

![14](./pics/22.png)

![14](./pics/23.png)

![14](./pics/24.png)

~~~makefile
 
######### 标准Makefile Lv1.2 / 使用动态库 ########
EXE=helloworld
SUBDIR=src object

#CXXFLAGS:编译选项, LDFLAGS:链接选项
CXXFLAGS += -I/home/mytest/example/include/
LDFLAGS += -L/home/mytest/example/lib -lexample

CXX_SOURCES =$(foreach dir,$(SUBDIR), $(wildcard $(dir)/*.cpp))
CXX_OBJECTS=$(patsubst  %.cpp, %.o, $(CXX_SOURCES))
DEP_FILES  =$(patsubst  %.o,  %.d, $(CXX_OBJECTS))

$(EXE): $(CXX_OBJECTS)
	g++  $(CXX_OBJECTS) -o $(EXE) $(LDFLAGS)	
%.o: %.cpp
	g++  -c  $(CXXFLAGS) -MMD $<  -o  $@
-include $(DEP_FILES)
clean: 
	rm  -rf  $(CXX_OBJECTS)  $(DEP_FILES)  $(EXE)
test:
	echo $(CXX_OBJECTS)
~~~



![14](./pics/25.png)

## 5.静态库的创建和使用

![14](./pics/26.png)

![14](./pics/27.png)

![14](./pics/28.png)

![14](./pics/29.png)

![14](./pics/30.png)

![14](./pics/31.png)



![14](./pics/32.png)

## 6.静态库和动态库混用

![14](./pics/33.png)

![14](./pics/34.png)

![14](./pics/35.png)

![14](./pics/36.png)

![14](./pics/37.png)

![14](./pics/38.png)

## 7.c函数和c++函数

c++调用c的库需要添加external c……详情请见课件。

## 8.动态库的手工加载

也就是动态库在代码中来加载，此部分略过。



# on windows

## 1.动态库的基本概念

![39](E:\c++\cppNotes\libraryNote\pics\39.png)

![40](E:\c++\cppNotes\libraryNote\pics\40.png)





![41](E:\c++\cppNotes\libraryNote\pics\41.png)

![42](E:\c++\cppNotes\libraryNote\pics\42.png)

建立一个控制台程序，但是在以下地方设置为DLL

![44](E:\c++\cppNotes\libraryNote\pics\44.png)

完成之后，会自动生成一个如下的项目工程

![45](E:\c++\cppNotes\libraryNote\pics\45.png)

对这个项目要进行一些设置

![46](E:\c++\cppNotes\libraryNote\pics\46.png)

首先是取消预编译头文件

![49](E:\c++\cppNotes\libraryNote\pics\49.png)

然后是运行时库选择为多线程调试

![img](file:///E:/c++/cppNotes/libraryNote/pics/47.png)

修改输出的名字，注意，有一个问题，即使修改了dll文件的路径和名称，但是lib文件的路径和名称并没有改变，lib文件的生成位置仍然在debug之下。这个不知道为什么和阿发老师的不一样。 

![48](E:\c++\cppNotes\libraryNote\pics\48.png)

在项目中新建一个代码文件

~~~C++
//mydll.cpp
// 导出Add
__declspec(dllexport) int Add(int a, int b)
{
	return a + b;
}
~~~

接下来，进行编译，就可以得到mydll.dll和mydll.lib文件。

![50](E:\c++\cppNotes\libraryNote\pics\50.png)



写一个程序，然后在应用程序中使用DLL

~~~C++
#pragma comment(lib,"mylib.lib")//这一句是用来导入lib库的，一定要写，相当于导入了头文件。否则会出现以下的错误。
/*
1>testlib.obj : error LNK2019: 无法解析的外部符号 "__declspec(dllimport) int __cdecl add(int,int)" (__imp_?add@@YAHHH@Z)，该符号在函数 _main 中被引用
1>E:\c++\testlib\Debug\testlib.exe : fatal error LNK1120: 1 个无法解析的外部命令
*/
__declspec(dllimport) int add(int a, int b);

int main()
{
	int result = add(10, 11);
	printf("result: %d \n", result);
	getchar();
	return 0;
    return 0;
}
~~~



![51](E:\c++\cppNotes\libraryNote\pics\51.png)

注意：DLL包括dll和lib文件。这些都要放在以下指定目录中，程序编译和运行的时候才可以找得到。另外，需要特别注意的是：如果单独点击可执行程序的时候，需要dll文件和它在一个目录之下。

![52](E:\c++\cppNotes\libraryNote\pics\52.png)
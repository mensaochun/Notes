# cmake教程

## 一、简单cmake

### 1.准备和构建

建立一个 /cmake 目录 ，在cmake/目录下建立t1文件夹，在t1下建立main.cpp和CMakeLists.txt文件

main.cpp:

~~~cpp
//main.cpp
#include <iostream>
using namespace std;
int main()
{
	cout<<"Hello World from t1 Main\n"<<endl;
	return 0;
}
~~~

CMakeLists.txt:

~~~cmake
PROJECT (HELLO) #设置工程的名字
SET(SRC_LIST main.cpp) #设置变量
MESSAGE(STATUS "This is BINARY dir " ${HELLO_BINARY_DIR})#显示消息
MESSAGE(STATUS "This is SOURCE dir "${HELLO_SOURCE_DIR})#显示消息
ADD_EXECUTABLE(hello ${SRC_LIST})#源文件生成可执行文件
~~~

开始构建

~~~shell
cmake . # .表示当前目录，..表示上一级目录
~~~

/cmake/t1文件夹下生成的文件如下

![](./pics/1.png)

运行命令

~~~shell
./hello
~~~

得到输出结果

~~~shell
Hello World from t1 Main
~~~

### 2.简单的解释

`PROJECT`定义工程名称，这个指令隐式的定义了两个 cmake 变量:<projectname>_BINARY_DIR 以及
 <projectname>_SOURCE_DIR。同时系统还预定义了PROJECT_BINARY_DIR 和PROJECT_SOURCE_DIR 变量，他们的值分别跟 HELLO_BINARY_DIR 与HELLO_SOURCE_DIR 一致。为了统一起见，建议以后直接使用 PROJECT_BINARY_DIR，PROJECT_SOURCE_DIR，即使修改了工程名称，也不会影响这两个变量。 

`SET` 指令可以用来显式的定义变量即可 。如果有多个源文件，也可以定义成：SET(SRC_LIST main.c t1.c t2.c) 

`MESSAGE`这个指令用于向终端输出用户定义的信息，包含了三种类型:

- SEND_ERROR，产生错误，生成过程被跳过。
- STATUS，输出前缀--的信息。
- FATAL_ERROR，立即终止所有 cmake 过程. 

`ADD_EXECUTABLE`定义了这个工程会生成一个文件名为 hello 的可执行文件，相关的源文件是 SRC_LIST 中定义的源文件列表， 本例中你也可以直接写成 ADD_EXECUTABLE(hello main.cpp)。 

**NOTE：**

- 在本例我们使用了\${}来引用变量，这是 cmake 的变量应用方式，但是，有一些例外，比如在 IF 控制语句，变量是直接使用变量名引用，而不需要\${}。如果使用了\${} 去应用变量，其实 IF 会去判断名为${}所代表的值的变量，那当然是不存在的了。


- SET(SRC_LIST main.c)也可以写成SET(SRC_LIST “main.c”)是没有区别的 

将本例改写成一个最简化的 CMakeLists.txt：

```cmake
PROJECT(HELLO)
ADD_EXECUTABLE(hello main.c) 
```

### 3.内部构建和外部构建

内部编译上面已经演示过了，它生成了一些无法自动删除的中间文件，所以，引出了我们对外部编译的探讨，外部编译的过程如下：

- 首先，请清除 t1 目录中除 main.c CmakeLists.txt 之外的所有中间文件，最关键的是CMakeCache.txt。
- 在 t1 目录中建立 build(有的时候也命名为release) 目录，当然你也可以在任何地方建立 build 目录，不一定必须在工程目录中。
- 进入 build 目录，运行 cmake ..(注意,..代表父目录，因为父目录存在我们需要的 CMakeLists.txt，
- 如果你在其他地方建立了 build 目录，需要运行 cmake <工程的全路径>)，查看一下 build 目录，就
  会发现了生成了编译需要的 Makefile 以及其他的中间文件。
- 运行 make 构建工程，就会在当前目录(build 目录)中获得目标文件 hello。

上述过程就是所谓的 out-of-source 外部编译，一个最大的好处是，对于原有的工程没有任何影响，
所有动作全部发生在编译目录。通过这一点，也足以说服我们全部采用外部编译方式构建工程。
这里需要特别注意的是：通过外部编译进行工程构建，HELLO_SOURCE_DIR 仍然指代工程路径，即：/cmake/t1而 HELLO_BINARY_DIR 则指代编译路径，即：/cmake/t1/build 

##　二、项目cmake

本小节的任务是让前面的 Hello World 更像一个工程，我们需要作的是：

- 为工程添加一个子目录 src，用来放置工程源代码；
- 添加一个子目录 doc，用来放置这个工程的文档 hello.txt；
- 在工程目录添加文本文件 COPYRIGHT, README；
- 在工程目录添加一个 runhello.sh 脚本，用来调用 hello 二进制；
- 将构建后的目标文件放入构建目录的 bin 子目录；
- 最终安装这些文件：将 hello 二进制与 runhello.sh 安装至/usr/bin，将 doc 目录的内容以及 

具体做法：

**1.准备工作：**
在/cmake/目录下建立 t2 目录。
将 t1 工程的 main.cpp 和 CMakeLists.txt 拷贝到 t2 目录中。
**2.添加子目录 src：**

```shell
mkdir src
mv main.cpp src
```

现在的工程看起来是这个样子：一个子目录 src，一个 CMakeLists.txt。上一节我们提到，需要为任何子目录建立一个 CMakeLists.txt，进入子目录 src，编写 CMakeLists.txt 如下：

```cmake
ADD_EXECUTABLE(hello main.c)
```

将 t2 工程的 CMakeLists.txt 修改为：

```cmake
PROJECT(HELLO)
ADD_SUBDIRECTORY(src bin)
```

然后建立 build 目录，进入 build 目录进行外部编译。

```shell
cmake ..
make
```

构建完成后，你会发现生成的目标文件 hello 位于 build/bin 目录中。
**3.语法解释：**
ADD_SUBDIRECTORY 指令
ADD_SUBDIRECTORY(source_dir \[binary_dir][EXCLUDE_FROM_ALL])
这个指令用于向当前工程添加存放源文件的子目录，并可以指定**中间二进制**和**目标二进制**存放的位置。

EXCLUDE_FROM_ALL 参数的含义是将这个目录从编译过程中排除，比如，工程的 example，可能就需要工程构建完成后，再进入 example 目录单独进行构建(当然，你也可以通过定义依赖来解决此类问题)。上面的例子定义了将 src 子目录加入工程，并指定编译输出(包含编译中间结果)路径为 bin 目录。如果不进行 bin 目录的指定，那么编译结果(包括中间结果)都将存放在 build/src 目录(这个目录跟原有的 src 目录对应)，指定 bin 目录后，相当于在编译时将 src 重命名为 bin，所有的中间结果和目标二进制都将存放在 bin 目录。

**4.换个地方保存目标二进制**
不论是 SUBDIRS 还是 ADD_SUBDIRECTORY 指令(不论是否指定编译输出目录)，我们都可以通过
SET 指令重新定义 EXECUTABLE_OUTPUT_PATH 和 LIBRARY_OUTPUT_PATH 变量来指定最
终的目标二进制的位置(指最终生成的 hello 或者最终的共享库，不包含编译生成的中间文件)
`SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)`
`SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)`
在第一节我们提到了\<projectname>_BINARY_DIR 和 PROJECT_BINARY_DIR 变量，他们指的编译发生的当前录，如果是内部编译，就相当于 PROJECT_SOURCE_DIR 也就是工程代码所在目录，如果是外部编译，指的是外部编译所在目录，也就是本例中的 build 目录。所以，上面两个指令分别定义了：可执行二进制的输出路径为 build/bin 和库的输出路径为 build/lib.本节我们没有提到共享库和静态库的构建，所以，你可以不考虑第二条指令。问题是，我应该把这两条指令写在工程的 CMakeLists.txt 还是 src 目录下的 CMakeLists.txt，把握一个简单的原则，在哪里 ADD_EXECUTABLE 或 ADD_LIBRARY，如果需要改变目标存放路径，就在哪里加入上述的定义。在这个例子里，当然就是指 src 下的 CMakeLists.txt 了。

## 三、如何安装

安装就是把某些文件复制到特定的目录之下。

## 四、静态库和动态库的构建


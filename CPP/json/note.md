# JSON

## windows下安装jsoncpp

参考

VisualStudio2015 编译和导入使用JsonCpp库：https://www.jianshu.com/p/4bd571e02226

but failed!

## ubuntu下的安装

### 准备： 

1. 安装 scons 

   在终端下命令行安装

   ~~~shell
   sudo apt-get install scons 
   ~~~

2. 安装 jsoncpp 

   先去官网下载jsoncpp的源码包：http://sourceforge.net/projects/jsoncpp/files/ 
   接下来一次执行命令： 

   ~~~shell
   # 解压
   tar -zxf jsoncpp-src-0.5.0.tar.gz 
   cd jsoncpp-src-0.5.0 
   # scons编译jsoncpp的源码
   scons platform=linux-gcc 
   # 将动态库加入到系统搜索路径中，这样在程序执行的时候就可以直接到这个路径上找
   sudo cp libs/linux-gcc-4.1.2/libjson_linux-gcc-4.1.2_libmt.so /usr/lib 
   # 将头文件拷贝到系统的搜索路径中，这样就可以直接include
   sudo cp -r include/json/ /usr/include 
   ~~~

   ​
   其中，/usr/lib 和/usr/lib可以自行选择。

   ​

3. 运行一个例子：test.cpp

   ~~~json
   #include<iostream>
   #include<json/json.h>
   using namespace std;
   using namespace Json;
   int main(){
       Value root;
       FastWriter fast;
       root["DataTime"]=("2018.1.22");
       cout<<fast.write(root)<<endl;
       return 0;
   }
   ~~~

   在命令行中输入

   ~~~shell
   g++ test.cpp -o test path/to/libjson_linux-gcc-4.8_libmt.a
   ~~~

   程序的运行结果：

   ~~~shel
   {"DataTime":"2018.1.22"}
   ~~~

   则证明运行成功！

4. 以上是配置静态库的运行。若是在clion中，可以直接配置动态库，配置如下：

   ~~~cmake
   cmake_minimum_required(VERSION 3.9)
   project(Test)
   set(CMAKE_CXX_STANDARD 11)
   add_executable(Test main.cpp)
   target_link_libraries(Test libjson_linux-gcc-4.8_libmt.so)
   ~~~

   ​

   ​
>>>>>>> d889b7de6797ee45b7fcf5eb1d6353fcaf81da66

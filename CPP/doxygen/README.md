# Doxygen tutorial

参考文档

1. linux doxygen 使用说明：https://wenku.baidu.com/view/7b26e3d384254b35eefd342f.html
2. Ubuntu16.04安装与使用Doxygen：http://blog.csdn.net/q1302182594/article/details/51517880



## linux下安装Doxygen

1. 安装Doxygen

   ~~~shell
   sudo apt-get install doxygen  
   ~~~


2. 安装Graphviz

   Doxygen使用Graphviz这个工具来绘图：

   ~~~shell
   sudo apt-get install graphviz  
   ~~~

   ​

## 生成配置文件

进入要生成文档的目录，比如test/，执行以下命令生成配置文件

~~~shell
# -s表示生成简单版的配置文件，如果不加-s选项，则生成的配置文件会存在大量的注释
doxygen -s -g
~~~

然后在test文件夹下，会生成一个叫做Doxyfile的文件，当然也可以自己指定这个名字。命令为

~~~shell
doxygen -s -g myconfigname
~~~

现在可以开始配置这个文件了

# 项目名称，将作为于所生成的程序文档首页标题

```shell
# 项目名称，将作为于所生成的程序文档首页标题 
PROJECT_NAME         = Test  
# 文档版本号，可对应于项目版本号，譬如 svn、cvs 所生成的项目版本号
PROJECT_NUMBER       = 1.0.0 
# 程序文档输出目录
OUTPUT_DIRECTORY     =  doc/ 
# 程序文档语言环境
OUTPUT_LANGUAGE    = Chinese  
# 如果是制作 C 程序文档，该选项必须设为 YES，否则默认生成 C++ 文档格式
OPTIMIZE_OUTPUT_FOR_C  = YES  
# 对于使用 typedef 定义的结构体、枚举、联合等数据类型，只按照 typedef 定义的类型名进行文档化
TYPEDEF_HIDES_STRUCT   = YES  
# 在 C++ 程序文档中，该值可以设置为 NO，而在 C 程序文档中，由于 C 语言没有所谓的域/名字空间这样的概念，所以此处设置为YES 
HIDE_SCOPE_NAMES        = YES  
# 让 doxygen 静悄悄地为你生成文档，只有出现警告或错误时，才在终端输出提示信息
QUIET   = YES  
# 只对头文件中的文档化信息生成程序文档 
FILE_PATTERNS          = *.h  
# 递归遍历当前目录的子目录，寻找被文档化的程序源文件
RECURSIVE              = YES 
# 示例程序目录 
EXAMPLE_PATH           = example/  
# 示例程序的头文档 (.h 文件) 与实现文档 (.c 文件) 都作为程序文档化对象 
EXAMPLE_PATTERNS       = *.c *.h  
# 递归遍历示例程序目录的子目录，寻找被文档化的程序源文件 
EXAMPLE_RECURSIVE      = YES  
# 允许程序文档中显示本文档化的函数相互调用关系 
REFERENCED_BY_RELATION = YES REFERENCES_RELATION    = YES 
REFERENCES_LINK_SOURCE = YES 
# 不生成 latex 格式的程序文档 
GENERATE_LATEX         = NO  
# 在程序文档中允许以图例形式显示函数调用关系，前提是你已经安装了 graphviz 软件包 
HAVE_DOT               = YES 
CALL_GRAPH            = YES 
CALLER_GRAPH        = YES  
#让doxygen从配置文件所在的文件夹开始，递归地搜索所有的子目录及源文件 
RECURSIVE = YES    
#在最后生成的文档中，把所有的源代码包含在其中 
SOURCE BROWSER = YES   
#这会在HTML文档中，添加一个侧边栏，并以树状结构显示包、类、接口等的关系 
GENERATE TREEVIEW ＝ ALL
```

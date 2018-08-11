# cxx
## 声明，定义，初始化，复制

​	声明是告诉编译器“这个函数或者变量可以在哪找到，它的模样像什么”。而定义则是告诉编译器，“在这里建立变量或函数”，并且为它们分配内存空间。

**函数声明与定义：**

　　函数的声明如：int Add(int, int);函数声明就是给函数取名并指定函数的参数类型，返回值类型。值得注意的是，在C语言中，有一点跟C++不同，对于带空参数表的函数如：int func()；在C中代表可以带任意参数（任意类型，任意数量），而在C++中代表不带任何参数。

　　函数的定义如：int Add(int a, int b){} 函数定义看起来跟函数声明很像，但是它有函数体，如果函数体中使用了参数，就必须为参数命名，这里大括号代替了分号的作用。

**变量声明与定义：**

　　变量的声明如：extern int i; 在变量定义前加extern关键字表示声明一个变量但不定义它，这对函数同样有效，如：extern int Add(int a, int b);因为没有函数体，编译器必会把它视作声明而不是定义，extern关键字对于函数来说是多余的，可选的。

　　变量的定义如：int i;如果在此之前没有对i的声明，那么这里既是对它的声明也是对它的定义，编译器会为其分配对应的内存。

**注意，如果在头文件中声明了一个变量，extern int a;那么说明，会在另外的文件中来定义这个变量。**



变量初始化：int a;就是同时进行了变量声明和定义。int a=1;变量声明，定义，初始化一起。a=3;这个是赋值。

## 添加头文件编译

~~~c++
#ifndef XXX
#define XXX
...
#end if
~~~

# 

## multiple definition of

​	早上编译一段代码，出现了如下的错误：

![img](http://img.blog.csdn.net/20161211095611387?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbWFudGlzXzE5ODQ=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

​	multiple definition of `xxxxx`错误出现了，MD是不是谁写代码没在.h文件加条件编译啊？仔细查看了代码发现确实加了条件编译。

~~~c++
#ifndef TEST_H

#define TEST_H

...

#endif
~~~



​	不是这个问题。这个问题哪里引起的呢？重复定义的问题，在哪里出现的呢？要解决这个问题先来看看变量的定义和声明的区别。

​	**声明**是向编译器介绍名字－－标识符，它告诉编译器“这个函数或变量在某处可找到，它的模样象什么”。而**定义**是说：“在这里建立变量”或“在这里建立函数”，它为名字分配存储空间。无论定义的是函数还是变量，编译器都要为它们在定义点分配存储空间。对于变量，编译器确定变量的大小，然后在内存中开辟空间来保存其数据，对于函数，编译器会生成代码，这些代码最终也要占用一定的内存。总之就是：把建立空间的声明成为“定义”，把不需要建立存储空间的成为“声明”。

​	查看代码，确实我在一个.h文件中定义了一个变量，而这个.h文件被多个文件包含，单独编译都没有问题，但是到链接的时候就出现问题了。

~~~c++
#ifndef _TEST_H_
#define _TEST_H_

......

struct pdesc const cameractrl_params[] = {
	{PT_STRI_, 0,  1, OFFSET(cameractrl, homecmd), "homecmd", 32, 0, NULL, NULL},
    {PT_STRI_, 0,  1, OFFSET(cameractrl, zoomctrl), "zoomctrl", 32, 0, NULL, NULL},
    {PT_STRI_, 0,  1, OFFSET(cameractrl, focusctrl), "focusctrl", 32, 0, NULL, NULL},
    {PT_STRI_, 0,  1, OFFSET(cameractrl, aperturectrl), "aperturectrl", 32, 0, NULL, NULL},
    {PT_NULL_, 0,  0, 0, "", 0, 0, NULL, NULL} /* PT_NULL means tail of struct pdesc array */
};

......

#endif
~~~

 	一般在.h文件中定义一个变量声明时，在其他文件中只要包含了这个.h文件，编译的时候就会独立被编译器解释，然后每个.C文件会生成独立的标识符和符号表，所以上述代码在单独编译的时候并不会报错，语法是合法的。但是，最后在编译器链接的时候，就会将工程中所有的符号整合在一起，由于文件中有重复的变量，于是就会出现重复定义的错误,系统就是提示你“multiple definition of `xxx`”。

​	进一步解释，我们可以这样想象编译每一个C源文件时，相当于一条有管道包围的纵向水流，二者互不干扰。当链接时两条原本相互独立的水管横向流了，所有就出现了重复的元素。所以当进行链接时就会出现重复定义的标示符。重复定义的标示符在这里只是变量，函数不会。**因为函数确实只在.c中定义了一次，多次声明是没有问题的，而变量确实出现了两次定义。**两次重复的变量定义链接器就不知道该已那个地址作为变量的内存，所以报错。

​	怎么解决这个问题呢?

​	其实只需要将全局变量定义从.h文件中挪到.c文件里，然后在.h文件中用extern做外部声明即可。即在.c文件中声明变量，然后在头文件.h所有的变量声明前加上extern，注意在.h文件中就不要对变量进行初始化赋值了。然后其他需要使用全局变量的.c文件中包含.h文件即可。编译器会为.c生成目标文件，然后链接时，如果该.c文件使用了全局变量，链接器就会链接到此.c文件。其他文件需要使用此全局变量也是同样的方式，目的其实只有一个，就是使变量在内存中唯一化。

 

 例子，上面代码如此修改就对了：

在test.c中定义

~~~c++
//test.c

......

**

struct pdesc const cameractrl_params[] = {	{PT_STRI, 0,  1, OFFSET(cameractrl, homecmd), "homecmd", 32, 0, NULL, NULL},    {PT_STRI, 0,  1, OFFSET(cameractrl, zoomctrl), "zoomctrl", 32, 0, NULL, NULL},    {PT_STRI, 0,  1, OFFSET(cameractrl, focusctrl), "focusctrl", 32, 0, NULL, NULL},    {PT_STRI, 0,  1, OFFSET(cameractrl, aperturectrl), "aperturectrl", 32, 0, NULL, NULL},    {PT_NULL_, 0,  0, 0, "", 0, 0, NULL, NULL} /* PT_NULL means tail of struct pdesc array */};

**

......

~~~



在test.h中定义

~~~c++
//test.h

#ifndef TEST_H

#define TEST_H

**

......

**

extern struct pdesc const cameractrl_params[];

**

......

**

#endif

~~~

​	这样，multiple definition of `xxxx`就搞明白了。

## 类模板中使用友元函数

friend void display(Test<T> &t);

↑ 这不是函数模板

template <class T>

void display(Test<T> &t)

{

​    cout << t.x << endl;

}

↑ 这是函数模板

本质上不是同一类东西，自然不可能后者是前者的定义。一个可行的解决方法（也是你的编译器所指出的方法）是对头文件做两处改动。

~~~c++
//---test.h
 
#ifndef test_h_
#define test_h_
#include <iostream>
using namespace std;
 
// 改动一：增加函数模板的声明——而这又需要先声明类模板
template <class T> class Test;
template <class T>
void display(Test<T> &t);

template <class T>
class Test
{
private:
    T x;
public:
    Test (T x_): x(x_) {}
    friend void display<>(Test<T> &t);
// 改动二：在函数名后面加上<>，指明它是之前声明的函数模板 的实例
};
 
template <class T>
void display(Test<T> &t)
{
    cout << t.x << endl;
}
#endif // test_h_
~~~

或者按照自己的方法

~~~c++
//
// Created by yourenchun on 2018/3/17.
//

#ifndef FC_VEC_TEST_TEMPLETE_H
#define FC_VEC_TEST_TEMPLETE_H

#include <iostream>
using namespace std;

template <class T>
class Test
{
private:
    T x;
public:
    Test (T x_): x(x_) {};
    template <class T2>
    friend void display(Test<T2> &t);//在这里进行函数模板的声明。注意typename根类的typename不一样。
};

template <class T2>
void display(Test<T2> &t)
{
    cout << t.x << endl;
}

#endif //FC_VEC_TEST_TEMPLETE_H

~~~

参考：https://bbs.csdn.net/topics/391867303

## 类模板头文件和源文件分离

使用显式声明实现类模板的接口与实现的文件分离

假设上面那个类的接口与实现分别放在了 .h 和 .cpp 文件中。然后在 .cpp 文件中显式的声明要使用的模板类实例，比如：

```c++
template class TestTemplate<int>;
```

然后，使用 `TestTemplate<int>` 也可以通过编译链接，但是只能使用已经显式声明的模板类实例。比如如果还要使用 `TestTemplate<float>`，就要这样：

``` c++
//不能使用typename？
template class TestTemplate<int>;
template class TestTemplate<float>;
```

就是说只能只用已经显式声明过的模板类实例。

如果是函数，要怎么显示实例化?

参考这篇文章：http://qixinbo.info/2017/07/09/cplusplus-template/，但是没有成功，FUCK！



## 常指针

`const char *p;` 常量指针，指向一块区域，这块区域不可写，只能读。
`char * const p;` 指针常量，指向一块区域，这块区域可读可写，但是指针的值初始后就不能改，类似于一般常量。

## C++中template的.h文件和.cpp文件的问题

在C++中，用到类模板时，如果类似一般的类声明定义一样，把类声明放在.h文件中，而具体的函数定义放在.cpp文件中的话，会发现编译器会报错。

原因在于，类模版并不是真正的类，它只是告诉编译器一种生成类的方法，编译器在遇到类模版的实例化时，就会按照模版生成相应的类。

在这里就是编译器遇到main函数中的`test<int> abc`;时就会去生成一个int类型的test类。

而每一个cpp文件是独立编译的，那么如果将类模版的成员函数单独放在一个cpp文件中，编译器便无法确定要根据什么类型来产生相应的类，也就造成了错误。

一般的解决方法就是将类模版中的成员函数定义也写入.h文件中。

参考：[C++中template的.h文件和.cpp文件的问题](http://www.cnblogs.com/caiminfeng/p/4835855.html)



## 随机数

[如何在C++中产生随机数](http://www.cnblogs.com/S031602240/p/6391960.html)

C++中没有自带的random函数，要实现随机数的生成就需要使用rand()和srand()。不过，由于rand()的内部实现是用线性同余法做的，所以生成的并不是真正的随机数，而是在一定范围内可看为随机的伪随机数。

- Rand
- Srand
- 通式

**Rand**

​	单纯的rand()会返回一个0至RAND_MAX之间的随机数值，而RAND_MAX的值与int位数有关，最小是32767。不过rand()是一次性的，因为系统默认的随机数种子为1，只要随机数种子不变，其生成的随机数序列就不会改变。

其实，对于rand()的范围，我们是可以进行人为设定的，只需要在宏定义中定义一个random(int x)函数，就可以生成范围为0至x的随机数值。当然，也可以定义为random(a,b)，使其生成范围为a至b的随机数值。具体定义方法在通式部分。

**Srand**

​	`srand()`可用来设置rand()产生随机数时的随机数种子。通过设置不同的种子，我们可以获取不同的随机数序列。可以利用`srand((unsigned int)(time(NULL))`的方法，利用系统时钟，产生不同的随机数种子。不过要调用`time()`，需要加入头文件`< ctime >`。

示例如下：

```c++
#include<iostream>
#include<cstdlib>
#include<ctime>
using namespace std;
int main()
{
    //如果没有这个随机种子，每次可执行文件运行出来的东西都是一样的。即使重新编译之后，运行出来的结果也是一样的。
    srand((unsigned)time(NULL));
    for(int i=0;i<10;i++)
    cout<<rand()<<' ';
    return 0;
}
```

**通式**

产生一定范围随机数的通用表示公式是：

- 取得(0,x)的随机整数：rand()%x；
- 取得(a,b)的随机整数：rand()%(b-a)；
- 取得[a,b)的随机整数：rand()%(b-a)+a；
- 取得[a,b]的随机整数：rand()%(b-a+1)+a；
- 取得(a,b]的随机整数：rand()%(b-a)+a+1；
- 取得0-1之间的浮点数：rand()/double(RAND_MAX)。

示例如下：

```c++
#include<iostream>
#include<cstdlib>
#include<ctime>
#define random(a,b) (rand()%(b-a+1)+a)
using namespace std;
int main()
{
    srand((unsigned)time(NULL));//注意，随机种子只能设一次，不能每次调用函数都设一次，否则每次的输出的结果都一样的。
    for(int i=0;i<10;i++)
    cout<<random(1,100)<<' ';
    return 0;
}
```

srand详解

​	rand函数在产生随机数前，需要系统提供的生成**伪随机数序列**的种子，rand根据这个种子的值产生一系列随机数。如果系统提供的种子没有变化，每次调用rand函数生成的伪随机数序列都是一样的。srand(unsigned seed)通过参数seed改变系统提供的种子值，从而可以使得每次调用rand函数生成的伪随机数序列不同，从而实现真正意义上的“随机”。通常可以利用系统时间来改变系统的种子值，即srand(time(NULL))，可以为rand函数提供不同的种子值，进而产生不同的随机数序列

​	特别注意：`srand()`在这里是吃一个时间的种子，如果时间一样，那么随机数序列也是一样的。因此，如果想让每次生成的随机数都是一样的，那么可以这么用：

~~~c++
srand(0);
~~~





## 数组大小不能是变量

在c++中时不支持变量作为数组长度参数的，如 `int n=10;byte bs[n]`;   这样写会提示编译错误”表达式必须含有常量值“。

虽然用变量声明数组大小会报编译错误，但是可以通过指针来动态申请空间实现动数组长度的变量赋值，写法如下：

~~~c++
int length = 10;
int * varArray;
varArray = new int[length];
~~~

这样varArray就可以当做数组来用了，这个数组的长度可以在程序运行时由计算得来。如果是普通的数组如int is[10] 编译时必须能确定数组长度，不然会报编译错误，这样灵活性受限比较大。我想这个就是new的存在原因之一吧，在栈中分配的内存，大小都是编译时就确定好的，如果想在运行时来动态计算使用内存的大小的话，就要用new这样的动态分配函数，来达到更高的灵活性。

可以自己声明一个结构体，来代表这个指针实现的数组，这样可读性会高点，用起来也方便点。

注意：c++ 用new分配空间以后，不用的时候要记得delete释放内存，不然会有内存泄露问题。

## 二维数组的首地址理解

## 字符串相关问题

~~~c++
void read_data(float X[][3],int n_rows,int n_cols,std::string &file_path){

    //X is 2D array.
    std::fstream f;
    f.open(file_path.data());
    // ensure f is opened rightly.
    if (f) {
        for (int i = 0; i < n_rows; i++) {
            for (int j = 0; j < n_cols; j++) {
                f<<X[i][j];
            }
        }
    }else{
        std::cout<<"error"<<std::endl;
    }
    f.close();
}

~~~



~~~c
//    float A[20][3], B[20][3];
//    std::string file_path_A = "/home/pi/stone/c++/cuda/test/data/A.txt";
//    std::string file_path_B = "/home/pi/stone/c++/cuda/test/data/B.txt";
//    read_data(A, 20, 3, file_path_A);
//    read_data(B, 20, 3, file_path_B);
//    for (int i = 0; i < 20; i++) {
//        for (int j = 0; j < 3; j++) {
//            std::cout << A[i][j] << " ";
//        }
//        std::cout << std::endl;
//    }
//    read_data(B, 20, 3, file_path_B);
~~~



## 二维数组动态分配内存

### demo1

C++中一维数组的动态分配十分常用，但C++初学者可能很少想过要使用动态分配的二维数组，或者自认为二维数组就是这样分配的（至少我自己开始的时候就这样认为）：`int m=2, n=3; int** array2D=new int[m][n];`。这完全是我们写多了像`int n=4; int* array=new int[n]`;

~~~c++
#include <iostream>
using std::cout;
using std::endl;
int main() {
    int i, j;
    int m = 2, n = 3;
    //分配行指针数组
    int **array2D = new int *[m];
    //为每一行分配空间
    for (i = 0; i < m; ++i) {
        array2D[i] = new int[n];
    }
    //可以稍微测试一下
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            array2D[i][j] = i + j;
        }
    }
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            cout << array2D[i][j] << "/t";
        }
        cout << endl;
    }
    //删除每一行分配的空间
    for (i = 0; i < m; ++i) {
        delete[] array2D[i];
    }
    //删除行指针数组
    delete[] array2D;
    return EXIT_SUCCESS;
}
~~~
### demo2

[动态创建二维数组](http://www.cnblogs.com/zzy19961112/p/6803326.html)

## 用数组作函数参数

### 解释1

数组是一系列数据的集合，无法通过参数将它们一次性传递到函数内部，如果希望在函数内部操作数组，必须传递数组指针。下面的例子定义了一个函数 max()，用来查找数组中值最大的元素：

```c++
include<stdio.h>
int max(int *intArr, int len) {
    int i, maxValue = intArr[0];  //假设第0个元素是最大值
    for (i = 1; i < len; i++) {
        if (maxValue < intArr[i]) {
            maxValue = intArr[i];
        }
    }
    return maxValue;
}

int main() {

    int nums[6], i;
    int len = sizeof(nums) / sizeof(int);
    //读取用户输入的数据并赋值给数组元素
    for (i = 0; i < len; i++) {
        scanf("%d", nums + i);
    }
    printf("Max value is %d!\n", max(nums, len));
    return 0;
}
```

运行结果：

~~~bash
12 55 30 8 93 27
Max value is 93!
~~~

**参数 intArr 仅仅是一个数组指针，在函数内部无法通过这个指针获得数组长度，必须将数组长度作为函数参数传递到函数内部。**数组 nums 的每个元素都是整数，scanf() 在读取用户输入的整数时，要求给出存储它的内存的地址，nums+i就是第 i 个数组元素的地址。

用数组做函数参数时，参数也能够以“真正”的数组形式给出。例如对于上面的 max() 函数，它的参数可以写成下面的形式：
```c++
int max(int intArr[6], int len) {
    int i, maxValue = intArr[0];  //假设第0个元素是最大值
    for (i = 1; i < len; i++) {
        if (maxValue < intArr[i]) {
            maxValue = intArr[i];
        }
    }
    return maxValue;
}
```
int intArr[6]好像定义了一个拥有 6 个元素的数组，调用 max() 时可以将数组的所有元素“一股脑”传递进来。

读者也可以省略数组长度，把形参简写为下面的形式：

~~~c++
int max(int intArr[], int len) {
    int i, maxValue = intArr[0];  //假设第0个元素是最大值
    for (i = 1; i < len; i++) {
        if (maxValue < intArr[i]) {
            maxValue = intArr[i];
        }
    }
    return maxValue;
}
~~~

int intArr[]虽然定义了一个数组，但没有指定数组长度，好像可以接受任意长度的数组。

实际上这两种形式的数组定义都是假象，不管是int intArr[6]还是int intArr[]都不会创建一个数组出来，编译器也不会为它们分配内存，实际的数组是不存在的，它们最终还是会转换为int *intArr这样的指针。这就意味着，两种形式都不能将数组的所有元素“一股脑”传递进来，大家还得规规矩矩使用数组指针。

int intArr[6]这种形式只能说明函数期望用户传递的数组有 6 个元素，并不意味着数组只能有 6 个元素，真正传递的数组可以有少于或多于 6 个的元素。

需要强调的是，不管使用哪种方式传递数组，都不能在函数内部求得数组长度，因为 intArr 仅仅是一个指针，而不是真正的数组，所以必须要额外增加一个参数来传递数组长度。

### 解释2

[详谈C++中数组作为函数参数](http://blog.csdn.net/oNever_say_love/article/details/49422517)

## 二维数组作为函数参数

注意：如果数组在栈中分配内存， 那么只能是正方形的，因为内存是连续的。

如果在堆中分配的内存，则可以不用是正方形的。

[C++二维数组做函数参数](http://www.cnblogs.com/L-Lotus-F/p/4377998.html)

参考：如何将二维数组作为函数的参数传递http://blog.csdn.net/xuleicsu/article/details/919801

~~~c++
#include <iostream>
void test(int **a,int row,int col){
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            std::cout<<a[i][j]<<"";
        }
        std::cout<<std::endl;
    }

}
int main(){
    int a[2][3]={{1,2,2},{2,2,2}};
    test((int **)a,2,3);//需要强制类型转换
}
~~~



## 指针数组和数组指针

## 动态分配数组内存

当按照以下方式给数组动态分配了内存之后

~~~c++
double *A=new double[100]
~~~

然后把里面的元素都打印出来，发现其中一个元素出现了`-nan`，这个很怪异的情况现在还没弄清楚是为什么。

不过提醒自己，最后在数组进行动态分配内存之后要先初始化，避免这种情况的发生。

## 数组动态内存删除

1.对于基本类型，delete和delete[]对销毁数组内存来说效果一样。

2.对于自定义类型，比如类，只能用delete[]来销毁内存。

问题，自己跑了一个例子：

~~~c++
# include<iosteram>
int main() {
    int *a = new int[10];
    std::cout << "Create array..." << std::endl;
    for (int i = 0; i < 10; i++) {
        a[i] = i;
        std::cout << a[i] << std::endl;
    }
    std::cout << "Delete array..." << std::endl;
    delete[]a;
    for (int i = 0; i < 10; i++) {
        std::cout << a[i] << std::endl;
    }
}
~~~

输出的结果竟然是

~~~bash
Create array...
0
1
2
3
4
5
6
7
8
9
Delete array...
0
0
2
3
4
5
6
7
8
9
~~~



参考文献：https://www.cnblogs.com/chucks123/p/7764449.html

## map使用

C++ Map常见用法说明：http://blog.csdn.net/shuzfan/article/details/53115922



## 模板类中使用友元函数

[在类模板中使用友元函数的问题](http://blog.csdn.net/qq_34232889/article/details/74922411)

## 函数指针作为模板

通过这样的方式来指定模板：`void(*f)(T*v)`

而不是这样的方式：`T`

参考文章：[函数指针模板参数](http://blog.csdn.net/microsoftwin32/article/details/37054849)



## 函数指针

1. 如何声明函数指针

   ~~~c++
   void (*pf)(int,int);
   ~~~

2. 函数指针作为参数

   ~~~c++
   int add(void (*pf)(int,int));
   ~~~

3. 通过typedef来进行简化

   ~~~c++
   typedef void (*pf)(int,int);
   pf ptr//定义了一个函数指针！
   ~~~

   ​
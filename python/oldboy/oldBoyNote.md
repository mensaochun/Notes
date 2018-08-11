## python课程笔记

[TOC]

### python变量作用域

python中没有块作用域的说法，也就是说if或者for语句里面的变量并不是局部的,注意这一点和java和c++是不一样的。请看连个例子：

~~~python
a = 1
if a == 1:
    b = 1
elif a == 2:
    b = 2
elif a == 3:
    b = 3
print b
#out:1

for i in range(10):
    b=1
print b
#out:1
~~~

关于变量作用域的知识请参考博客：

http://python.jobbole.com/86465/

### python中变量赋值

python中也有引用的概念，和c++中引用的概念类似。与指针的区别在哪里呢？在 python 中赋值语句总是建立对象的引用值，而不是复制对象。因此，python 变量更像是指针，而不是数据存储区域，这点和大多数语言类似吧，比如 C++、java 等。

 具体参看这篇博文，讲得非常详细：http://www.jb51.net/article/102796.htm

或者参考这篇：http://blog.csdn.net/qq_32907349/article/details/52190796

~~~python
#赋值操作只是建立了引用，因此a,b同时指向了同一个内存地址
In [1]: a=1
In [2]: b=a
In [3]: id(a)
Out[3]: 37774984L
In [4]: id(b)
Out[4]: 37774984L
  
#现在改变a的值，这个时候会建立一个新的对象，a的地址发生变化
In [17]: a=1
In [18]: b=a
In [19]: id(a)
Out[19]: 37774984L
In [20]: id(b)
Out[20]: 37774984L
In [21]: a=2
In [22]: b
Out[22]: 1z

In [24]: id(a)
Out[24]: 37774960L

In [25]: id(b)
Out[25]: 37774984L
  
#但是对于可变对象来说，却不是这样的。可变对象包括字典，列表，类等等。

#列表的例子
In [26]: a=[1,2,3]
In [27]: b=a
In [28]: a[0]=4
In [29]: b
Out[29]: [4, 2, 3]
In [30]: id(a)
Out[30]: 70641672L
In [31]: id(b)
Out[31]: 70641672L
 
#类的例子
In [35]: class A(object):
	         pass
In [36]: a=A()
In [37]: b=a
In [38]: a.x=1
In [40]: b.x
Out[40]: 1
In [41]: a.x=2
In [42]: b.x
Out[42]: 2

  
#函数传递的时候，也遵循这样的道理。当参数是可变对象的时候，采用的是引用传递，当参数是不可变对象的时候，采用的是值传递。
~~~



###python的深拷贝和浅拷贝

浅拷贝：shallow copy，含义是他是对原来引用的对象进行了复制，但是不再引用同一对象地址。

深拷贝：deep copy , 对原来引用的对象和对象里面嵌套的可变元素都进行拷贝，注意只是对嵌套的可变元素进行拷贝！

浅拷贝和深拷贝的区别详见这篇文章，讲得非常好：http://www.jb51.net/article/100170.htm

~~~python
#[:]操作是浅复制，可以完全复制一份，且内存地址发生了变化。但是对象内部的元素的引用还是一样的，这一点要非常的注意！！！
In [53]: a=[1,2,3]
In [54]: a
Out[54]: [1, 2, 3]
In [55]: b=a[:]
In [56]: b&     d y
Out[56]: [1, 2, 3]
In [57]: id(a)
Out[57]: 73632712L
In [58]: id(b)
Out[58]: 73431368L
#看看地址就知道怎么回事了，元素的地址还是一样的。如果列表里面嵌套了列表，就可能会出问题了！
In [59]: print("id of elements of a",[id(i) for i in a])
('id of elements of a', [37774984L, 37774960L, 37774936L])

In [60]: print("id of elements of b",[id(i) for i in b])
('id of elements of b', [37774984L, 37774960L, 37774936L])


#deep copy,需要把import copy
~~~



注意可变类型：

~~~python
可变类型：list , set , dict，类等等
不可变类型：int, str , float , tuple
~~~





### 类的初始化发生了什么？

初始化的时候，首先建立了一个Dog实例对象，然后将这个实例对象作为参数传入了`__init__()`函数中，在此函数中即可以对该对象进行属性赋值等操作。

~~~python
class Dog(object):
    def __init__(self,name,age):
        self.name=name
        self.age=age
    def bulk():
        print "wang!"
        
Dog dog(name="chun",18)
~~~



### 类变量（静态变量）和类方法

类变量是属于类的，不属于实例。但是在创建实例的时候，会将类变量复制一份给实例，也就相当于增加了一个实例属性。实例可以随意地读取和改变这个属性，但是不影响类变量的值（除非这个类变量是一个可变类型的数据）。类变量可以通过Dog.dog_num来直接访问和修改，或者通过类方法来进行访问和修改。

**问题**：类方法仍然是传递self,而不是cls?

类变量的好处：1.可以定义常量来使用。2.不同实例之间进行共享变量，如计数总共有多少只狗。

~~~python
# coding=utf-8
class Dog(object):
    dog_num = 0  # 这是一个类变量

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def bulk(self):
        print "wang!"

    @classmethod
    def count(cls):
        cls.dog_num += 1

    @staticmethod
    def Do():
        print "I am a static method."
        
dog1=Dog("chun", 18)
dog1.count()
dog1.Do()
print Dog.dog_num
print
dog2=Dog("stone", 19)
dog2.count()
print Dog.dog_num
~~~



### 静态方法

静态方法就是包装在类中的一个方法，和类没有半毛钱关系，它只是作为一个工具来使用而已。因此不需要传入self参数，只需要用装饰器@staticmethod来修饰方法就可以了。



### 属性方法

pass

通过装饰器来实现，@property

### 总体原则

类变量只通过类方法或者类名来调用和改写。

静态方法只通过类名来调用。



### `__dict__`属性

类和实例都有这个属性，`__dict__`，这个属性是一个字典，key是属性，value是属性值。

~~~python

class Animal(object):
    def __init__(self,age,name):
        self.age=age
        self.name=name

    def run(self):
        print 'I can run.'
class Dog(Animal):
    def __init__(self,age,name,legs):
        super(Dog,self).__init__(age,name)
        self.legs=legs
    def bulk(self):
        print "Wang!"
if __name__ == '__main__':
    dog=Dog(age=1,name='chun',legs=4)
    print Dog.__dict__
    """
    {'bulk': <function bulk at 0x00000000029EDC18>, '__module__': '__main__', 
    '__doc__': None, '__init__': <function __init__ at 0x00000000029EDBA8>}
    """
    print dog.__dict__
    """
    {'age': 1, 'legs': 4, 'name': 'chun'}
    """
~~~



### `__setattr__()`和`__getattr__()`

通过这个可以增加或者查看实例的属性。

### super的使用

由于子类中有`__init__()`方法，会覆盖父类中的方法，因此，需要通过方法来调用父类的方法来进行父类属性的初始化。

有两种初始化方法：

一种是通过父类的类名直接调用`__init__()`，但是这种方法的明显缺点就是当父类的名字改变，所有子类都要去修改相应的类名，这样太麻烦了。

另外一种是通过super关键字来调用，这种方法父类一定要继承object，也就是说父类必须是新式类。

~~~python
class Animal(object):
    def __init__(self,age,name):
        self.age=age
        self.name=name
    def run(self):
        print 'I can run.'
class Dog(Animal):
    def __init__(self,age,name,legs):
        super(Dog,self).__init__(age,name)
        self.legs=legs
    def bulk(self):
        print "Wang!"
if __name__ == '__main__':
    dog=Dog(age=1,name='chun',legs=4)
~~~



### 继承

注意，在父类可以在成员函数中使用未初始化的成员属性，只要在子类中对这个属性有赋值就可以了。参考下面我自己写的一个例子：

~~~python
# coding=utf-8
class Animal(object):
    def __init__(self, name):
        self.name = name

    def get_food(self):
        print(self.food)


class Dog(Animal):
    def __init__(self, name, food):
        super(Dog, self).__init__(name)
        self.food = food

dog=Dog("chun","meat")
dog.get_food()

#out：meat
~~~



### 多继承

pass

多继承的顺序是有关系的。

### 多态

多态的作用很简单，就是在函数参数中可以传递不同子类的实例，但是都可以调用同一个父类的方法。

### python的垃圾回收机制

### python的内存模型

### 析构函数

`__delete__()`这个函数是析构函数。

在对象销毁的时候会自动调用。在python中实行垃圾回收机制，会定期清理。问题：怎么判断什么时候可以清理？

如果没有销毁对象，则在程序结束的时候，对象会销毁，这时候会调用析构函数。

析构函数一般做一些关闭文件，清理垃圾(?)的任务。



### 装饰器

@property的使用！

详情参看老男孩的博客：http://www.cnblogs.com/alex3714/articles/5765046.html





首先看几个应用实例：

1.在京东购物的时候，当你没有登陆的时候，你点击很多商品购买的时候都会让你登录，但是一旦登录之后，再购买其他商品就不需要登录了。这个时候就要用到装饰器。

2.当要计算一个函数运行的时间的时候，可以用上装饰器，就不需要在每调用一个函数的时候都去做计算运行时间的操作。

3.当很多函数都会抛出相似的error的时候，也可以写一个装饰器来捕捉这些error

总之一句话，装饰器说白了就是用一个函数来包装另一个函数的，以进行某些操作，然后返回这个函数。注意，装饰器可以有参数。这个参数可以进行一些操作。



作业：写一个京东登录的例子。

写一个log装饰器，带参数。功能，1.进行登录检测。2.判断要执行哪个函数。

写三个函数，home（），进行登录。finance（），有关金融。buy（），有关购物。

思考一下，以下装饰器的缺点在哪里？为什么不能运行？

~~~python
# coding=utf-8
import time

def print_sth():
    print "hello"
def get_sum(a, b):
    return a + b

def get_time(f, *args, **kwargs):
    begin = time.time()
    f(*args, **kwargs)
    end = time.time()
    print end - begin
if __name__ == '__main__':
    print get_time(get_sum,1,2)#这一行不能运行，请问是不是可变参数出了问题？
    get_time(print_sth)
~~~



### 闭包

pass



### 生成器

列表生成式的缺点：如果是生成一个列表的内容很大，很可能会出现out of memory的问题。

~~~python
L = [x * x for x in range(100000000)]
#out:out of memory
~~~

生成器的优点：列表元素可以按照某种算法推算出来。代码很简单：

~~~python
L = (x * x for x in range(10))
for i in range(10):
    print next(L)
#out: 0,1,4,9....
~~~

生成器有一个问题，就是列表生成式的函数比较复杂的时候，不好在列表生成式中写出来。这时候相关功能可以在函数中独立地实现出来。举个例子：

~~~python

~~~



可以运用到的地方：用生成器生成一个可以不断生成batch数据的功能函数。有待实现。

~~~python

#这是一个生成器，用来生成batch数据，实际上，还需要进行改进
def next_batch(data,batch_size):
    begin = 0
    end = begin + batch_size
    while end<len(data):
        yield data[begin:end]
        begin = end
        end += batch_size
for i in next_batch(range(1000),2):
    print i
~~~



### 迭代器



### 字符编码

字符编码一直没有怎么搞懂，需要以后用到的时候再好好研究。不过有几篇博客可以好好参考一下。



编码与屏幕显示的编码有关系。



首先是老男孩的：

http://www.cnblogs.com/alex3714/articles/5717620.html

python字符编码详解：

http://python.jobbole.com/82107/

py编码终极版

http://www.cnblogs.com/yuanchenqi/articles/5956943.html

  




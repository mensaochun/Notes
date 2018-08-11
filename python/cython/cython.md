# cython

## setup

**Ref**: https://stackoverflow.com/questions/1471994/what-is-setup-py

It helps to install a python package foo on your machine (can also be in virtualenv) so that you can import the package foo from other projects and also from [I]Python prompts. It does the similar job of pip, easy_install etc.,

**Using setup.py**

Let's start with some definitions:

Package - A folder/directory that contains __init__.py file.
Module - A valid python file with .py extension.
Distribution - How one package relates to others.

Let's say you want to install a package named foo. Then you do,

~~~shell
$ git clone https://github.com/user/foo  
$ cd foo
$ python setup.py install
~~~

Instead, if you don't want to actually install it but still would like to use it. Then do,

~~~shell
$ python setup.py develop  
~~~

This command will create symlinks to the source directory within site-packages instead of copying things. Because of this, it is quite fast (particularly for large packages).

**Creating setup.py**

If you have your package tree like,

foo
├── foo
│   ├── data_struct.py
│   ├── __init__.py
│   └── internals.py
├── README
├── requirements.txt
└── setup.py
Then, you do the following in your setup.py script so that it can be installed on some machine:

~~~python
from setuptools import setup
setup(
   name='foo',
   version='1.0',
   description='A useful module',
   author='Man Foo',
   author_email='foomail@foo.com',
   packages=['foo'],  #same as name
   install_requires=['bar', 'greek'], #external packages as dependencies
)
~~~


Instead, if your package tree is more complex like the one below:

foo
├── foo
│   ├── data_struct.py
│   ├── __init__.py
│   └── internals.py
├── README
├── requirements.txt
├── scripts
│   ├── cool
│   └── skype
└── setup.py
Then, your setup.py in this case would be like:

~~~python
from setuptools import setup
setup(
   name='foo',
   version='1.0',
   description='A useful module',
   author='Man Foo',
   author_email='foomail@foo.com',
   packages=['foo'],  #same as name
   install_requires=['bar', 'greek'], #external packages as dependencies
   scripts=[
            'scripts/cool',
            'scripts/skype',
           ]
)
~~~


Add more stuff to (setup.py) & make it decent:

~~~python
from setuptools import setup
with open("README", 'r') as f:
    long_description = f.read()
setup(
   name='foo',
   version='1.0',
   description='A useful module',
   license="MIT",
   long_description=long_description,
   author='Man Foo',
   author_email='foomail@foo.com',
   url="http://www.foopackage.com/",
   packages=['foo'],  #same as name
   install_requires=['bar', 'greek'], #external packages as dependencies
   scripts=[
            'scripts/cool',
            'scripts/skype',
           ]
)
~~~

The long_description is used in pypi.org as the README description of your package.

And finally, you're now ready to upload your package to PyPi.org so that others can install your package using pip install yourpackage.

First step is to claim your package name & space in pypi using:

~~~shell
$ python setup.py register
~~~

Once your package name is registered, nobody can claim or use it. After successful registration, you have to upload your package there (to the cloud) by,

~~~shell
$ python setup.py upload
~~~

Optionally, you can also sign your package with GPG by,

~~~shell
$ python setup.py --sign upload
~~~

Bonus: See a sample setup.py from a real project here: torchvision-setup.py



## cython

Refs:

1. cython初探:http://darren1231.pixnet.net/blog/post/336855350-cython-%E5%88%9D%E6%8E%A2
2. Cython基础系列博客：http://blog.csdn.net/i2cbus/article/details/18181637
3. Cython’s Documentation: https://cython.readthedocs.io/en/latest/index.html


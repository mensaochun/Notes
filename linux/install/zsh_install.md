# zsh教程

### 1.安装zsh

~~~shell
sudo apt-get install zsh
~~~

### 2.设置默认shell

如果想设置zsh为默认shell，执行

~~~shell
chsh -s /bin/zsh #chsh:change shell
~~~

切换回bash为默认shell，执行

~~~shell
chsh -s /bin/bash
~~~

### 3.安装oh-my-zsh

原版的zsh需要很麻烦的配置，而oh-my-zsh是第三方的一个配置方案，安装它就ok了

执行命令

~~~shell
sh -c "$(wget https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh -O -)"
~~~

### 4.配置主题

主题非常多，选择一个合适的主题很重要。目前测试了一个非常简洁舒爽的主题，gnzh，通过修改~/.zshrc来进行配置

~~~vim
ZSH_THEME="gnzh" 
~~~

### 5. 进入和退出zsh模式

如果当前shell是bash，那么可以通过

~~~shell
zsh
~~~

命令来进入zsh模式，如果想退出，执行

~~~shell
exit
~~~

就可以退出zsh模式，进入bash模式



Refs：

Linux终极shell-Z Shell--用强大的zsh & oh-my-zsh把Bash换掉：http://blog.csdn.net/gatieme/article/details/52741221
# softwares install 

## typora

see here: https://www.typora.io/#linux

### 出错处理

参考1.https://blog.csdn.net/weixin_29053197/article/details/80369723

安装后直接打开的话是会闪退的，提示 
version GLIBCXX_3.4.21 not defined in file libstdc++.so.6 with link time reference (ubuntu14.04) 
这时需要这样做 

~~~shell
sudo apt-get install npm 
sudo npm install spellchecker 
sudo cp node_modules/spellchecker/build/Release/spellchecker.node /usr/share/typora/resources/app/node_modules/spellchecker/build/Release/
~~~

如果在安装spellchecker时，提示npm ERR! Error: CERT_UNTRUSTED 
那么执行下面这条语句

~~~shell
npm config set strict-ssl false 
npm config set registry="http://registry.npmjs.org/"
~~~

此时typora应该就可以正常打开了

出错参考2：https://github.com/typora/typora-issues/issues/504

如果还是出错，请按照：

```
sudo apt-get update && sudo apt-get upgrade
sudo apt-get install build-essential
sudo apt-get install aptitude
sudo apt-get install libstdc++6
```


## youdao

see here: http://blog.csdn.net/qianggezhishen/article/details/49208689

## fuxin

see here: http://blog.topspeedsnail.com/archives/8655

## sougou input 

https://blog.csdn.net/ljheee/article/details/52966456

yuan https://blog.csdn.net/lm409/article/details/53939990

https://www.cnblogs.com/lixiaolun/p/5495911.html



install:

Ubuntu 16.04 LTS安装sogou输入法详解: http://blog.csdn.net/qq_21792169/article/details/53152700

problem:

Ubuntu安装搜狗输入法找不到的问题:https://jingyan.baidu.com/article/54b6b9c0eedd252d583b4714.html

## zsh教程

- 安装zsh

    sudo apt-get install zsh
- 设置默认shell

如果想设置zsh为默认shell，执行

    chsh -s /bin/zsh #chsh:change shell

切换回bash为默认shell，执行

    chsh -s /bin/bash
- 安装oh-my-zsh

原版的zsh需要很麻烦的配置，而oh-my-zsh是第三方的一个配置方案，安装它就ok了

执行命令

    sh -c "$(wget https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh -O -)"
- 配置主题

主题非常多，选择一个合适的主题很重要。目前测试了一个非常简洁舒爽的主题，gnzh，通过修改~/.zshrc来进行配置

    ZSH_THEME="gnzh" 
- 进入和退出zsh模式

如果当前shell是bash，那么可以通过

    zsh

命令来进入zsh模式，如果想退出，执行

    exit

就可以退出zsh模式，进入bash模式

**Refs：**

Linux终极shell-Z Shell--用强大的zsh & oh-my-zsh把Bash换掉：http://blog.csdn.net/gatieme/article/details/52741221
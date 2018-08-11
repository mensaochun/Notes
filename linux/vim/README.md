## install

一键安装vim插件的命令。

~~~bash
wget -qO- https://raw.github.com/ma6174/vim/master/setup.sh | sh -x
~~~

## unstall

~~~bash
#!/bin/sh
rm -f ~/.vimrc
rm -rf ~/.vim
mv -f ~/.vimrc_old ~/.vimrc
mv -f ~/.vim_old ~/.vim
~~~




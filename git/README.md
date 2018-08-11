## git基本使用

## 一、git在windows上的安装

msysgit是Windows版的Git，从[https://git-for-windows.github.io](https://git-for-windows.github.io/)下载，网速慢的到国内镜像下载：https://pan.baidu.com/s/1kU5OCOB#list/path=%252Fpub%252Fgi ，然后按默认选项安装即可。

安装完成后，在开始菜单里找到“Git”->“Git Bash”，蹦出一个类似命令行窗口的东西，就说明Git安装成功！

安装完成后，还需要最后一步设置，在命令行输入：

```
$ git config --global user.name "Your Name"
$ git config --global user.email "email@example.com"
```

因为Git是分布式版本控制系统，所以，每个机器都必须自报家门：你的名字和Email地址。你也许会担心，如果有人故意冒充别人怎么办？这个不必担心，首先我们相信大家都是善良无知的群众，其次，真的有冒充的也是有办法可查的。

注意`git config`命令的`--global`参数，用了这个参数，表示你这台机器上所有的Git仓库都会使用这个配置，当然也可以对某个仓库指定不同的用户名和Email地址。

## 二、github的使用

### 1.创建版本库

初始化一个Git仓库，使用`git init`命令。

添加文件到Git仓库，分两步：

- 第一步，使用命令`git add <file>`，注意，可反复多次使用，添加多个文件；
- 第二步，使用命令`git commit`，完成。

### 2.工作区和暂存区

电脑里能看到的目录就是工作区。

工作区有一个隐藏目录`.git`，这个不算工作区，而是Git的版本库。Git的版本库里存了很多东西，其中最重要的就是称为stage（或者叫index）的暂存区，还有Git为我们自动创建的第一个分支`master`，以及指向`master`的一个指针叫`HEAD`。

![0](E:\git\pics\0.jpg)

前面讲了我们把文件往Git版本库里添加的时候，是分两步执行的：

第一步是用`git add`把文件添加进去，实际上就是把文件修改添加到暂存区；

第二步是用`git commit`提交更改，实际上就是把暂存区的所有内容提交到当前分支。

因为我们创建Git版本库时，Git自动为我们创建了唯一一个`master`分支，所以，现在，`git commit`就是往`master`分支上提交更改。

### 添加远程库：

1.创建远程仓库

`create repository`

2.本地关联到远程仓库

~~~shell
git remote add origin git@github.com:michaelliao/learngit.git
~~~

### 从远程库中pull到本地：

~~~shell
git pull origin master#origin 是远程库的名字，master是本地分支
~~~

### 添加gitignore

~~~shell
# 创建文件.gitignore
touch .gitignore
# 忽略后缀为.o和.a的文件
*.o
*.a
# 忽略名称为main.cpp文件
main.cpp
# 匹配模式最后跟"/"说明要忽略的是目录 
dir/
~~~





http://blog.csdn.net/cscmaker/article/details/8553980

### github删除文件夹

`git rm -rf dir`命令删除的是本地仓库，如果push之后，远程仓库应该也没有了。问题：会删除本地文件吗？

`git rm -r --cached some-directory`则是删除缓冲，即可以删除github上的文件夹，之后再`commit`，`push`就可以了！

### git的冲突如何解决

**1. 冲突是如何产生的**

我们都知道，Git的实现途径是1棵树。比如有一个节点树(point1), 

- 我们基于point1进行开发，开发出了结点point2； 
- 我们基于point1进行开发，开发出了结点point3； 

如果我们在point2和point3内操作了同一类元素，那么势必会导致冲突的存在。 
主要的思想如下图1所示:

代码: poin1.class

```java
public class Point{
	int size;
public void add(){
	size+=1;
}
}
```

人物甲 更新了版本2 
代码: poin2.class

```java
public class Point{    
    int size;
public void add(){	
    size+=2;
}
}
```

人物乙 更新了版本3 
代码: poin3.class

```java
public class Point{
	int size;
public void add(){
	size+=3;
}
}
```

场景如下，甲乙都是根据point.java 文件进行了开发。甲开发出了版本2，并且提交了代码；乙开发出了版本3，也需要提交了代码，此时将会报错存在冲突。

为什么呢？因为甲开发完了版本，提交了版本之后，此时远端的代码已经是版本2点代码了，而乙是基于版本1进行的开发出了版本3。所以，乙想要提交代码，势必要将自己的代码更新为版本2的代码，然后再进行提交，如果存在冲突则解决冲突后提交。

**2. 冲突是如何解决的**

上面已经详细的说明了冲突时如何产生的，那么又该如何解决冲突呢?

解决冲突通常使用如下的步骤即可:

情况1 无冲突 

- 先拉取远端的代码，更新本地代码。然后提交自己的更新代码即可。

情况2 有冲突 

- 拉取远端代码。存在冲突，会报错。 
- 此时我们需要将本地代码暂存起来 **stash**； 
- 更新本地代码，将本地代码版本更新和远端的代码一致即可； 
- 将暂存的代码合并到更新后的代码后，有冲突解决冲突(需要手动进行解决冲突)； 
- 提交解决冲突后的代码。

具体的git命令，略。

问题

- 我们没有add到仓库的代码也能检测到冲突？如果是以后就用stash这一招。
- 怎么手动解决冲突？
- merge命令怎么使用？



### git使用beyond compare

### github建立分支

在本地新建一个分支： git branch newBranch 
切换到你的新分支: git checkout newBranch 
将新分支发布在github上： git push origin newBranch 
在本地删除一个分支： git branch -d newBranch 
在github远程端删除一个分支： git push origin :newBranch (分支名前的冒号代表删除) 
/git push origin –delete newBranch 
注意删除远程分支后，如果有对应的本地分支，本地分支并不会同步删除！


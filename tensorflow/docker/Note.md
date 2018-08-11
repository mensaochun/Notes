## docker

1. 下载tensorflow镜像，并且创建自己的容器。

   ~~~shell
   # -it在命令行中运行，否则会在后台运行，不会在shell中显示。bash是指定了在bash运行，也可以在notebook上运行。
   nvidia-docker run --name yourc_tf -it tensorflow/tensorflow:latest-gpu bash
   # 注意nvidia-docker是gpu版本的。
   ~~~
   如果是添加端口映射的话，

   ~~~shell
   nvidia-docker run -p 2222:22 --name yourc_tf -it tensorflow/tensorflow:latest-gpu bash
   ~~~

   如果要挂载文件

   ~~~shell
   nvidia-docker run -p 2222:22 --name yourc_tf2 -it -v /home/yourc/stone:/root/stone tensorflow/tensorflow:latest-gpu bash
   ~~~

   ​

2. 查看镜像和容器

   ~~~shell
   docker ps -a(包括没有启动的容器)
   docker images
   ~~~

3. 启动容器，关闭容器，删除容器

   ~~~shell
    # 是否需要用nvidia-docker?
    docker start -i yourc_tf(加上-i才能在shell上运行)
    docker stop 
    docker rm
   ~~~

4. 建立软链接

   ~~~shell
   ln -s source dist        # 建立软连接
   ~~~

5. 在docker中安装软件。

   在使用docker容器时，有时候里边没有安装vim，敲vim命令时提示说：vim: command not found，这个时候就需要安装vim，可是当你敲apt-get install vim命令时，提示：

   ```shell
   Reading package lists... Done
   Building dependency tree       
   Reading state information... Done
   E: Unable to locate package vim
   ```
    这时候需要敲：apt-get update，这个命令的作用是：同步 /etc/apt/sources.list 和 /etc/apt/sources.list.d 中列出的源的索引，这样才能获取到最新的软件包。等更新完毕以后再敲命令：apt-get install vim命令即可。

6. 将主机的目录挂载到容器内，实现数据共享。

   `Docker`可以支持把一个宿主机上的目录挂载到镜像里。
   命令如下:

   ~~~shell
   docker run -it -v /home/yourc:/root ubuntu64 /bin/bash
   ~~~

   通过-v参数，冒号前为宿主机目录，必须为绝对路径，冒号后为镜像内挂载的路径。

   在自己的电脑上是这样的：

   ~~~shell
   docker run -it -v /home/yourc:/root tensorflow/tensorflow:latest-gpu bash
   ~~~

   ​

7. 拷贝数据s

   但是对这三种方法我都不太喜欢，无意间看到另位一种方法供大家参考：

   从主机复制到容器`sudo docker cp host_path containerID:container_path`

   从容器复制到主机`sudo docker cp containerID:container_path host_path`

   容器ID的查询方法想必大家都清楚:`docker ps -a`


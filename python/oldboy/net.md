# 计算机网络

###计算机网络

计算机网络说白一点，就是用网络来实现资源共享和数据传递的。资源共享的意思是，我从这里就可以访问你那里的资源，比如网页；数据传递的意思是，我从这里就可以传递信息到你那里，比如QQ。

![1](.\pics\1.png)

###OSI模型

OSI模型只要了解有这个东西就可以了。

![img](./pics/2.png?lastModify=1511018919)

### 网络通信要素

IP地址：指定是哪台机子；

端口：指定是某个进程；

传输协议：指定通讯的规则，像人类的交流语言一样。

![3](.\pics\3.png)





UDP：是不可靠的，不管有没有连接，都把数据扔过去。这样的好处是，速度快，效率高。坏处是不安全。

TCP：必须要是连接，而且是实行三次握手协议。

![4](.\pics\4.png)

注意：TCP学会了，UDP自然就会了。![5](.\pics\5.png)

### 网络通讯步骤

#### 客户端



建立一个socket连接

~~~Python
# 导入socket库:
import socket
# 创建一个socket:
# family用于指定是ipv4还是ipv6，一般是用Ipv4.
# type:用于指定面向流的TCP协议，还有其他选项，暂时不用管。
s = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
# 建立连接:
s.connect(('www.sina.com.cn', 80))
~~~

发送的文本格式必须符合HTTP标准，如果格式没问题，接下来就可以接收新浪服务器返回的数据了：

~~~python
# 接收数据:
buffer = []
while True:
    # 每次最多接收1k字节:
    d = s.recv(1024)
    if d:
        buffer.append(d)
    else:
        break
# 把list中的元素按照空格分开
data = b''.join(buffer)
~~~



#### 服务端

1.首先，创建一个基于IPv4和TCP协议的Socket：

~~~python
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
~~~

2.然后，我们要绑定监听的地址和端口。服务器可能有多块网卡，可以绑定到某一块网卡的IP地址上，也可以用`0.0.0.0`绑定到所有的网络地址，还可以用`127.0.0.1`绑定到本机地址。`127.0.0.1`是一个特殊的IP地址，表示本机地址，如果绑定到这个地址，客户端必须同时在本机运行才能连接，也就是说，外部的计算机无法连接进来。

~~~python
# 监听端口:
# bind里面接收的是一个元组，因此可以这么写：address=('127.0.0.1', 9999)
s.bind(('127.0.0.1', 9999))#注意1024之下的端口都被系统占用了，不要使用。
~~~

3.紧接着，调用`listen()`方法开始监听端口，传入的参数指定等待连接的最大数量，也就是说最多等待几个客户端来连接：

~~~python
s.listen(5)#这里指定了最多5个客户端来连接，注意并不是代表着可以5个客户端同时连接，只是说最多可以有这么多人一起来等待。
print('Waiting for connection...')
~~~

4.接下来，服务器程序通过一个永久循环来接受来自客户端的连接，`accept()`会等待并返回一个客户端的连接:

~~~python
while True:
    # Wait for an incoming connection.  
    # Return a new socket representing the connection, and the address of the client.  
    # For IP sockets, the address info is a pair (hostaddr, port).
    sock, addr = s.accept()
    # 创建新线程来处理TCP连接:
    t = threading.Thread(target=tcplink, args=(sock, addr))
    t.start()
~~~

5.每个连接都必须创建新线程（或进程）来处理，否则，单线程在处理连接的过程中，无法接受其他客户端的连接：

~~~python
def tcplink(sock, addr):
    print('Accept new connection from %s:%s...' % addr)
    sock.send(b'Welcome!')
    while True:
        # Receive up to buffersize bytes from the socket.
        # When no data is available, block until at least one byte is available or until the 		# remote end is closed.  
        # When the remote end is closed and all data is read, return the empty string.
        data = sock.recv(1024)
        time.sleep(1)
        if not data or data.decode('utf-8') == 'exit':
            break
        # Send a data string to the socket. 
        # Return the number of bytes sent; 
        sock.send(('Hello, %s!' % data.decode('utf-8')).encode('utf-8'))
    sock.close()
    print('Connection from %s:%s closed.' % addr)
~~~



一些注意点

~~~python
s.send(bytes("heh",'utf-8'))
input(">>>")
~~~




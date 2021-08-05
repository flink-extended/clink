![](https://github.com/Qihoo360/libfg/blob/master/docs/logo.png)
# Clink编译
## Linux（Centos）
### Docker编译
+ 运行third_party/docker 下的build.sh,生成Docker镜像
```
    cd third_party
    sh build.sh
```
+ 在容器内编译代码
```
    docker run -it -v ${your_code_path}:/home/clink clink-base:1.0.0 bash
    cd /home/clink
    mkdir build
    cmake3 ..
    make -j && make install
```  

### 物理机编译
+ 运行third_party centos_deps.sh, 安装编译环境依赖
```
    sh centos_deps.sh
    sh docker/bootstrap.sh
```
+ 物理机编译代码
```
    mkdir build
    cmake3 ..
    make -j && make install
```

## MacOS
### 编译环境
运行 third_party 下macos_deps.sh ,安装编译环境依赖 
```
    cd third_party
    sh macos_deps.sh
```

### 编译
```
    mkdir build
    cd build
    make -j && make install
```


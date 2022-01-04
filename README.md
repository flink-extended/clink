<img src="./docs/clink-logo.png" height="80px">

> Clink is a linker between Flink/Alink stream processing and C/C++ online serving focus on ML feature engineering.

**<u>*NOTE: Currently we are making a major re-design to Clink (e.g. use TFRT as the underlying runtime for parallelization). Before the adjustments are merged to `master`, please refer to [dev](https://github.com/flink-extended/clink/tree/dev) branch for our latest progress.*</u>**

 # 编译
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


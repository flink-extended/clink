<p align="center">
  <img src="./docs/clink_logo.png" height="80px"/>
</p>

# Clink

Clink is a library that provides infrastructure to do the following:
- Defines C++ functions that can be parallelized by TFRT threadpool.
- Executes a graph (in the MLIR format) of these C++ functions in parallel.
- Makes C++ functions executable as Java functions using
  [JNA](https://github.com/java-native-access/jna).

Furthermore, Clink provides an off-the-shelf library of reusable Feature
Processing functions that can be executed as Java and C++ functions.

Clink is useful in the scenario where users want to do online feature
processing with low latency (in sub-millisecond) in C++, apply the same logic
to do offline feature processing in Java, and implement this logic only once
(in C++).


## Getting Started

### Prerequisites

Clink uses [TFRT](https://github.com/tensorflow/runtime) as the underlying
execution engine and therefore follows TFRT's Operation System and installation
requirements.

Here are the prerequisites to build and install Clink:
- Ubuntu 16.04
- Bazel 4.0.0
- Clang 11.1.0
- libstdc++8 or greater
- openjdk-8

Please checkout the [TFRT](https://github.com/tensorflow/runtime) README for
more detailed instructions to install, configure and verify Bazel, Clang and
libstdc++8.

### Build docker image with the libraries required by Clink

#### Build docker image based on Ubuntu 16.04

```
docker build -t ubuntu:16.04_clink -f docker/Dockerfile_ubuntu_1604 .
```

#### Build docker image based on CentOS 7.7

Note that the CentOS Dockerfile needs to compile Clang 11 from source code.
This could take a few hours depending on the machine's CPU capacity.

```
docker build -t centos:centos7.7.1908_clink -f docker/Dockerfile_centos_77 .
```

### Setup Clink repo before building Clink

```
git submodule update --init --recursive
```

### Execute Clink C++ functions in parallel in C++

```
bazel run //:executor -- `pwd`/mlir_test/executor/basic.mlir --work_queue_type=mstd --host_allocator_type=malloc
```

### Execute Clink C++ functions in Java

```
bazel run //:example
```

### Format C++ code using clang-format

Clink uses [ClangFormat](https://clang.llvm.org/docs/ClangFormat.html) to format C++ code.

```
find . \( -name "*.cc" -or -name "*.h" \) -not -path "./tfrt/*" -exec clang-format -i -style=llvm {} \;
```



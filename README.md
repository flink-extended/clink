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

### Build & Test all Targets

Users can run the following command to build all targets and to run all tests.

```
bazel test $(bazel query //...)
```

### Execute Clink C++ functions in parallel in C++

```
bazel run //:executor -- `pwd`/mlir_test/executor/basic.mlir --work_queue_type=mstd --host_allocator_type=malloc
```

### Format codes using clang-format and diffplug-spotless

Clink uses [ClangFormat](https://clang.llvm.org/docs/ClangFormat.html) to format C++ code.

```
find . \( -name "*.cc" -or -name "*.h" \) -not -path "./tfrt/*" -exec clang-format -i -style=llvm {} \;
```

And it uses [diffplug/spotless](https://github.com/diffplug/spotless) to format java code.

```
mvn -f java-lib spotless:apply
```

### View & Edit Java Code with IDE

Clink provides maven configuration that allows users to view or edit java code with IDEs like IntelliJ IDEA. Before IDEs can correctly compile java project, users need to run the following commands after setting up Clink repo and build Clink.

```
bazel build //:clink_java_proto
cp bazel-bin/libclink_proto-speed.jar java-lib/lib/
```

Then users can open `java-lib` directory with their IDEs.

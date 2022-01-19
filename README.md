<p align="center">
  <img src="./docs/clink_logo.png" height="80px"/>
</p>

# Clink

Clink is a library that provides infrastructure to do the following:
- Defines C++ functions that can be parallelized by TFRT thread pool.
- Executes a graph (in the MLIR format) of these C++ functions in parallel.
- Makes C++ functions executable as Java functions using
  [JNA](https://github.com/java-native-access/jna).

Furthermore, Clink provides an off-the-shelf library of reusable Feature
Processing functions that can be executed as Java and C++ functions.

Clink is useful in the scenario where users want to do online feature processing
with low latency (in sub-millisecond) in C++, apply the same logic to do offline
feature processing in Java, and implement this logic only once (in C++).

## Getting Started

### Prerequisites

Clink uses [TFRT](https://github.com/tensorflow/runtime) as the underlying
execution engine and therefore follows TFRT's Operation System and installation
requirements.

Currently supported operating systems are as follows:

- Ubuntu 16.04
- CentOS 7.7.1908

Here are the prerequisites to build and install Clink:
- Bazel 4.0.0
- Clang 11.1.0
- libstdc++8 or greater
- openjdk-8

Clink provides dockerfiles and pre-built docker images that satisfy the
installation requirements listed above. You can use one of the following
commands to build the docker image, according to the operating system you expect
to use.

```bash
$ docker build -t ubuntu:16.04_clink -f docker/Dockerfile_ubuntu_1604 .
```

```bash
$ docker build -t centos:centos7.7.1908_clink -f docker/Dockerfile_centos_77 .
```

Or you can use one of the following commands to pull the pre-built Docker image
from Docker Hub.

```bash
$ docker pull docker.io/flinkextended/clink:ubuntu16.04
```

```bash
$ docker pull docker.io/flinkextended/clink:centos7.7.1908
```

If you plan to set up the Clink environment without the docker images provided
above, please check the [TFRT](https://github.com/tensorflow/runtime) README for
more detailed instructions to install, configure and verify Bazel, Clang, and
libstdc++8.

### Building Clink from Source

After setting up the environment according to the instructions above and pulling
Clink repository, please use the following command to update submodules like
TFRT.

```bash
$ git submodule update --init --recursive
```

Then, users can run the following command to build all targets and to run all
tests.

```bash
$ bazel test $(bazel query //...)
```

### Executing Examples

Users can execute Clink C++ function example in parallel in C++ using one of the
following commands.

```bash
$ bazel run //:executor -- `pwd`/mlir_test/executor/basic.mlir --work_queue_type=mstd --host_allocator_type=malloc
```

<!-- TODO: Add detailed example illustrating the usage of Clink Runner API. -->

## Developer Guidelines

### Code Formatting

Changes to Clink C++ code should conform to [Google C++ Style
Guide](https://google.github.io/styleguide/cppguide.html).

Clink uses [ClangFormat](https://clang.llvm.org/docs/ClangFormat.html) to check
C++ code, [diffplug/spotless](https://github.com/diffplug/spotless) to check
java code, and [Buildifier](https://github.com/bazelbuild/buildtools) to check
bazel code.

Please run the following command to format codes before uploading PRs for
review.

```bash
$ ./tools/format-code.sh
```

### View & Edit Java Code with IDE

Clink provides maven configuration that allows users to view or edit java code
with IDEs like IntelliJ IDEA. Before IDEs can correctly compile java project,
users need to run the following commands after setting up Clink repo and build
Clink.

```bash
$ bazel build //:clink_java_proto
$ cp bazel-bin/libclink_proto-speed.jar java-lib/lib/
```

Then users can open `java-lib` directory with their IDEs.
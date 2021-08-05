#!/bin/bash
# clink mac 编译环境搭建
NPROCS=`nproc`
NPROCS=4

brew install cmake gflags protobuf glog rapidjson libarchive

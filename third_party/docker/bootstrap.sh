#!/bin/bash

NPROCS=`nproc`
NPROCS=4

rapidjson_repository='https://github.com/Tencent/rapidjson/archive/refs/tags/v1.1.0.tar.gz'
mkdir -p /opt/rapidjson && pushd /opt/rapidjson && wget $rapidjson_repository -O rapidjson-1.1.0.tar.gz
if [ -e rapidjson-1.1.0.tar.gz ]; then
  tar xzvf rapidjson-1.1.0.tar.gz && pushd rapidjson-1.1.0 && \
  mkdir build && pushd build && cmake3 .. && make -j${NPROCS} && make install && popd && popd && \
  popd && rm -rf /opt/rapidjson
  echo 'Build rapidjson successfully'
else
  echo 'Fail to download rapidjson'
  exit 1
fi

gflags_repository='https://github.com/gflags/gflags/archive/v2.2.2.tar.gz'
mkdir -p /opt/gflags && pushd /opt/gflags && wget $gflags_repository -O gflags-v2.2.2.tar.gz
if [ -e gflags-v2.2.2.tar.gz ]; then
  tar xzvf gflags-v2.2.2.tar.gz && pushd gflags-2.2.2 && \
  mkdir build && pushd build && cmake3 -DCMAKE_CXX_FLAGS=-fPIC -DBUILD_SHARED_LIBS=ON .. && \
  make -j${NPROCS} && make install && popd && popd && \
  popd && rm -rf /opt/gflags
  echo 'Build gflags successfully'
else
  echo 'Fail to download gflags'
  exit 1
fi

glog_repository='https://github.com/google/glog/archive/v0.4.0.tar.gz'
mkdir -p /opt/glog && pushd /opt/glog && wget $glog_repository -O glog-v0.4.0.tar.gz
if [ -e glog-v0.4.0.tar.gz ]; then
  tar xzvf glog-v0.4.0.tar.gz && pushd glog-0.4.0 && \
  mkdir build && pushd build && cmake3 -DCMAKE_CXX_FLAGS=-fPIC -DBUILD_SHARED_LIBS=ON .. && make -j${NPROCS} && make install && popd && popd && \
  popd && rm -rf /opt/glog
  echo 'Build glog successfully'
else
  echo 'Fail to download glog'
  exit 1
fi

protobuf_repository='https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protobuf-cpp-3.15.6.tar.gz'
mkdir -p /opt/protobuf && pushd /opt/protobuf && wget $protobuf_repository -O protobuf-cpp-3.15.6.tar.gz
if [ -e protobuf-cpp-3.15.6.tar.gz ]; then
  tar xzvf protobuf-cpp-3.15.6.tar.gz && pushd protobuf-3.15.6 && \
  ./configure CFLAGS="-fPIC" CXXFLAGS="-fPIC" && make -j${NPROCS} && make install && popd && \
  popd && rm -rf /opt/protobuf
  echo 'Build glog successfully'
else
  echo 'Fail to download protobuf'
  exit 1
fi

brpc_repository='https://github.com/apache/incubator-brpc/archive/refs/tags/0.9.7.tar.gz'
mkdir -p /opt/brpc && pushd /opt/brpc &&  wget $brpc_repository -O incubator-brpc-0.9.7.tar.gz
if [ -e incubator-brpc-0.9.7.tar.gz ] ; then
  tar xzvf incubator-brpc-0.9.7.tar.gz && pushd incubator-brpc-0.9.7 && \
  mkdir -p build && pushd build && cmake3 -DWITH_GLOG=ON .. && make -j${NPROCS} && make install && popd && popd && \
  popd && rm -rf /opt/brpc
  echo 'Build brpc successfully'
else
  echo 'Fail to download brpc and brpc.patch'
  exit 1
fi

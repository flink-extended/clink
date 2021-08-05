#!/bin/bash
# clink centos 编译环境搭建

yum -y install epel-release && sed -e 's|^mirrorlist=|#mirrorlist=|g' \
    -e 's|^#baseurl=http://mirror.centos.org/centos|baseurl=https://mirrors.ustc.edu.cn/centos|g' -i.bak \
    /etc/yum.repos.d/CentOS-Base.repo && \
    sed -i 's/enabled=1/enabled=0/g' /etc/yum/pluginconf.d/fastestmirror.conf && yum clean all && yum makecache && \
    yum -y install epel-release && yum -y install gcc gcc-c++ make vim wget cmake3 \
    libarchive-devel zlib-devel curl-devel gperftools-devel \
    openssl-devel leveldb-devel gtest-devel&& \
    yum clean all
    
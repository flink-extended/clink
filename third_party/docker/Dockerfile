FROM centos:7
RUN yum -y install epel-release && sed -e 's|^mirrorlist=|#mirrorlist=|g' \
    -e 's|^#baseurl=http://mirror.centos.org/centos|baseurl=https://mirrors.ustc.edu.cn/centos|g' -i.bak \
    /etc/yum.repos.d/CentOS-Base.repo && \
    sed -i 's/enabled=1/enabled=0/g' /etc/yum/pluginconf.d/fastestmirror.conf && yum clean all && yum makecache && \
    yum -y install epel-release && yum -y install gcc gcc-c++ make vim wget cmake3 \
    libarchive-devel zlib-devel curl-devel gperftools-devel \
    openssl-devel leveldb-devel gtest-devel&& \
    yum clean all && \
    localedef -c -f UTF-8 -i zh_CN zh_CN.utf8

ENV LC_ALL=zh_CN.utf8 \
    TZ="Asia/Shanghai" \
    CLINK_WORKSPACE=/opt/clink \
    LD_LIBRARY_PATH=$CLINK_WORKSPACE/lib:/usr/local/lib:/usr/local/lib64:/lib64:/usr/lib64:$LD_LIBRARY_PATH

COPY bootstrap.sh /bootstrap.sh

RUN chmod +x /bootstrap.sh && /bootstrap.sh
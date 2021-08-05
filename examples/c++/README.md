# c++ example使用说明

## 首先编译生成example依赖的动态库文件
`cd ${clink_project_dir}`
`mkidr build`
`cmake .. && make -j && make install`

## 编译example
gcc版本：4.8.5
proto版本: 3.15.6
## 编译
`cd ${clink_example_c++}`
`mkdir build & cd build`
`cmake ..`
`make -j 10`
## 运行
`./example`



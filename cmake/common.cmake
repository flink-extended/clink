if (DEFINED INCLUDE_DIR)
    return()
endif ()

list(APPEND LIB_DIR /usr/lib /usr/lib64 /usr/local/lib)


IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(CMAKE_CXX_FLAGS "-O3 -Wall -Wno-unused-function -Wno-unused-variable \
                         -Wno-switch -Wno-deprecated-declarations -Wno-format \
                         -Wno-unused-value -Wno-sign-compare -fPIC")

list(APPEND LIB_DIR /usr/local/lib64)

ENDIF (CMAKE_SYSTEM_NAME MATCHES "Linux")


# c++11
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT DEFINED THIRD_PARTY)
    set(THIRD_PARTY /opt/third_party)
endif ()


set(INCLUDE_DIR)
#set(LIB_DIR /opt/gcc-8.1.3/lib64)
set(PLUGIN_PROTO_PATH ${CMAKE_BINARY_DIR}/protos/plugin/protos)
set(SERVER_PROTO_PATH ${CMAKE_BINARY_DIR}/protos/server/protos)
if (PROTO2)
    SET(SRC_PROTO_PATH ${CMAKE_SOURCE_DIR}/protos/proto2)
else ()
    SET(SRC_PROTO_PATH ${CMAKE_SOURCE_DIR}/protos/proto3)

endif ()
file(GLOB PLUGIN_PROTO_FILES
        DIRECTORIES ${SRC_PROTO_PATH}/plugin/*.proto)
file(GLOB SERVER_PROTO_FILES
        DIRECTORIES ${SRC_PROTO_PATH}/server/*.proto)


if (NOT EXISTS ${PROTO_PATH})
    execute_process(COMMAND mkdir -p ${PLUGIN_PROTO_PATH})
    execute_process(COMMAND mkdir -p ${SERVER_PROTO_PATH})
    execute_process(COMMAND protoc -I ${SRC_PROTO_PATH}/plugin --cpp_out=${PLUGIN_PROTO_PATH} ${PLUGIN_PROTO_FILES})
    execute_process(COMMAND protoc -I ${SRC_PROTO_PATH}/server --cpp_out=${SERVER_PROTO_PATH} ${SERVER_PROTO_FILES})
endif ()
set(PROTO_PLUGIN_HDRS_DIR "${CMAKE_SOURCE_DIR}/core/protos")
file(GLOB PROTO_HDRS ${PLUGIN_PROTO_PATH}/*.pb.h)
file(COPY ${PROTO_HDRS} DESTINATION ${PROTO_PLUGIN_HDRS_DIR})
set(PROTO_SERVER_HDRS_DIR "${CMAKE_SOURCE_DIR}/service/protos")
file(GLOB PROTO_HDRS ${SERVER_PROTO_PATH}/*.pb.h)
file(COPY ${PROTO_HDRS} DESTINATION ${PROTO_SERVER_HDRS_DIR})

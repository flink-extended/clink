set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -D_GLIBCXX_USE_NANOSLEEP -fPIC")
foreach (folder config operands operators transformer utils processor protos common processor variables source ${PLUGIN_PROTO_PATH})
    #    list(APPEND INCLUDE_DIR ${folder})
    file(GLOB_RECURSE MODULE_SRC_FILES ${folder}/*.c ${folder}/*.cc ${folder}/*.cpp ${folder}/*.hpp)
    list(APPEND SERVER_SRC_FILES ${MODULE_SRC_FILES})
endforeach (folder)


include_directories(${INCLUDE_DIR} ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/ ..)
link_directories(${LIB_DIR})
add_library(${PROJECT_NAME} SHARED ${SERVER_SRC_FILES})
if (PLUGIN_ARCHIVE)
    target_link_libraries(${PROJECT_NAME} -fPIC -Wl,--whole-archive libglog.a libprotobuf.a libarchive.a libcityhash.a -Wl,--no-whole-archive -rdynamic -lstdc++ curl)
    #target_link_libraries(${PROJECT_NAME}_plugin -fPIC -Wl,-all_load libglog.a libprotobuf.a libarchive.a libcityhash.a -Wl,-noall_load -rdynamic -lstdc++ curl)
else ()
    target_link_libraries(${PROJECT_NAME} -lglog -lprotobuf -rdynamic -lstdc++ curl -larchive -ltcmalloc)
endif ()


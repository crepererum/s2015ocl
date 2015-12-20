# generic CMake setup
cmake_minimum_required (VERSION 3.0)
include (ExternalProject)

# project metadata
project (s2015ocl)

# compiler flags
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -g -pthread -Wall -Wextra -Wconversion -Wsign-conversion")
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# detect clang
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics -fdiagnostics-show-category=name")
endif ()

# bunch of external projects
externalproject_add (
    cmock
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/cmock"
    GIT_REPOSITORY "https://github.com/ThrowTheSwitch/CMock.git"
    GIT_TAG "8dc65e82bf4697a2fdc86f0d45ccad0749028d2d"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    TEST_COMMAND ""
)

externalproject_add (
    unity
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/unity"
    GIT_REPOSITORY "https://github.com/ThrowTheSwitch/Unity.git"
    GIT_TAG "v2.1.0"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    TEST_COMMAND ""
)

externalproject_add (
    clheaders
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/clheaders"
    GIT_REPOSITORY "https://github.com/KhronosGroup/OpenCL-Headers"
    GIT_TAG "a501078069186a4a950590382abb926b70114f89"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR> <INSTALL_DIR>/include/CL
    TEST_COMMAND ""
)

externalproject_add (
    clhpp
    DEPENDS cmock unity clheaders
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/clhpp"
    GIT_REPOSITORY "https://github.com/KhronosGroup/OpenCL-CLHPP.git"
    GIT_TAG "v2.0.9"
    PATCH_COMMAND sed -e "s/add_subdirectory(examples)//g" -e "s/add_subdirectory(tests)//g" -i "" CMakeLists.txt
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -DCMOCK_DIR=${CMAKE_CURRENT_BINARY_DIR}/cmock/src/cmock -DUNITY_DIR=${CMAKE_CURRENT_BINARY_DIR}/unity/src/unity -DOPENCL_INCLUDE_DIR=${CMAKE_CURRENT_BINARY_DIR}/clheaders/include
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target generate_cl2hpp
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <BINARY_DIR>/include/CL <INSTALL_DIR>/include/CL
    TEST_COMMAND ""
)

externalproject_add (
    spdlog
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/spdlog"
    GIT_REPOSITORY "https://github.com/gabime/spdlog.git"
    GIT_TAG "01f3d6f5f0de7c14ebb24ca579c17e44003a15d7"
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
)

externalproject_add (
    backward
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/backward"
    GIT_REPOSITORY "https://github.com/bombela/backward-cpp.git"
    GIT_TAG "v1.0"
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy <SOURCE_DIR>/backward.hpp <INSTALL_DIR>/include/backward.hpp
)

include_directories (
    SYSTEM
    "${CMAKE_CURRENT_BINARY_DIR}/backward/include"
    "${CMAKE_CURRENT_BINARY_DIR}/clheaders/include"
    "${CMAKE_CURRENT_BINARY_DIR}/clhpp/include"
    "${CMAKE_CURRENT_BINARY_DIR}/spdlog/include"
)

include_directories (
    "include"
)

# main executable
aux_source_directory ("src" SourceFiles)
add_executable (s2015ocl ${SourceFiles})
add_dependencies (s2015ocl backward clhpp spdlog)
target_link_libraries (s2015ocl dw OpenCL)

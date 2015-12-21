# generic CMake setup
cmake_minimum_required (VERSION 3.0)
include (ExternalProject)
set (CMAKE_EXPORT_COMPILE_COMMANDS 1)

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
    project_cmock
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/cmock"
    GIT_REPOSITORY "https://github.com/ThrowTheSwitch/CMock.git"
    GIT_TAG "8dc65e82bf4697a2fdc86f0d45ccad0749028d2d"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    TEST_COMMAND ""
)

externalproject_add (
    project_unity
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/unity"
    GIT_REPOSITORY "https://github.com/ThrowTheSwitch/Unity.git"
    GIT_TAG "v2.1.0"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    TEST_COMMAND ""
)

externalproject_add (
    project_clheaders
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/clheaders"
    GIT_REPOSITORY "https://github.com/KhronosGroup/OpenCL-Headers"
    GIT_TAG "a501078069186a4a950590382abb926b70114f89"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR> <INSTALL_DIR>/include/CL
    TEST_COMMAND ""
)

externalproject_add (
    project_clhpp
    DEPENDS project_cmock project_unity project_clheaders
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
    project_spdlog
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/spdlog"
    GIT_REPOSITORY "https://github.com/gabime/spdlog.git"
    GIT_TAG "01f3d6f5f0de7c14ebb24ca579c17e44003a15d7"
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
)

externalproject_add (
    project_backward
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/backward"
    GIT_REPOSITORY "https://github.com/bombela/backward-cpp.git"
    GIT_TAG "v1.0"
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy <SOURCE_DIR>/backward.hpp <INSTALL_DIR>/include/backward.hpp
)

externalproject_add (
    project_nanogui
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/nanogui"
    GIT_REPOSITORY "https://github.com/wjakob/nanogui.git"
    GIT_TAG "a6480f5b09f0b334ab3dceb79601e6355a7df3ea"
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -DNANOGUI_BUILD_SHARED=OFF -DNANOGUI_BUILD_EXAMPLE=OFF -DNANOGUI_BUILD_PYTHON=OFF
    INSTALL_COMMAND
        ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include <INSTALL_DIR>/include
        && ${CMAKE_COMMAND} -E copy <SOURCE_DIR>/ext/nanovg/src/nanovg.h <INSTALL_DIR>/include/nanovg.h
        && ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/ext/eigen/Eigen <INSTALL_DIR>/include/Eigen
        && ${CMAKE_COMMAND} -E copy <BINARY_DIR>/libnanogui.a <INSTALL_DIR>/lib/libnanogui.a
    TEST_COMMAND ""
)

include_directories (
    SYSTEM
    "${CMAKE_CURRENT_BINARY_DIR}/backward/include"
    "${CMAKE_CURRENT_BINARY_DIR}/clheaders/include"
    "${CMAKE_CURRENT_BINARY_DIR}/clhpp/include"
    "${CMAKE_CURRENT_BINARY_DIR}/nanogui/include"
    "${CMAKE_CURRENT_BINARY_DIR}/spdlog/include"
)

link_directories (
    "${CMAKE_CURRENT_BINARY_DIR}/nanogui/lib"
)

include_directories (
    "include"
)

# main executable
aux_source_directory ("src" SourceFiles)
add_executable (s2015ocl ${SourceFiles})
add_dependencies (s2015ocl project_backward project_clhpp project_nanogui project_spdlog)
target_link_libraries (
    s2015ocl
    dl
    dw
    GL
    nanogui
    OpenCL
    rt
    X11
    Xcursor
    Xi
    Xinerama
    Xrandr
    Xxf86vm
)

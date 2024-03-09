cmake_minimum_required(VERSION 3.13)
project(AdvPT)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../")

include_directories(include)
include_directories(extract-data)
include_directories(lib)

file(GLOB_RECURSE SRCS src/*.cpp extract-data/utils.cpp)
file(GLOB_RECURSE INCLUDE include/*.h extract-data/utils.h extract-data/mnist_reader.h)

add_executable(AdvPT ${SRCS} ${INCLUDE})

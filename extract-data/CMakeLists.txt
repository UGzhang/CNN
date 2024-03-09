cmake_minimum_required(VERSION 3.13)
project(MnistData)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../../")
add_executable(MnistData main.cpp utils.h utils.cpp mnist_reader.h)
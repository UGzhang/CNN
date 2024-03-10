cmake_minimum_required(VERSION 3.13)
project(AdvPT)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../")

include_directories(include)
include_directories (third_party/eigen)

file(GLOB_RECURSE SRCS src/*.cpp )
file(GLOB_RECURSE INCLUDE include/*.h )

add_executable(AdvPT ${SRCS} ${INCLUDE})

cmake_minimum_required(VERSION 3.9)
project(OpenCL VERSION 6.6.6 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

add_executable(main src/main.cpp)

find_package(OpenCL REQUIRED)
target_link_libraries(main OpenCL::OpenCL)

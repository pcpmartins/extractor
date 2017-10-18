cmake_minimum_required(VERSION 2.8)
file (GLOB targets *.cpp)
add_library(src ${targets})

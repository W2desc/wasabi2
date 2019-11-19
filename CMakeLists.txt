cmake_minimum_required (VERSION 2.6)

project(wasabi2 CXX)

list(APPEND CMAKE_CXX_FLAGS "-fmax-errors=10 -pthread -std=c++11 ${CMAKE_CXX_FLAGS} -g -ggdb -Wall")
#add_definitions(-Wall -g -ggdb -O3 )
add_definitions(-g -ggdb -Wall -O3 -pthread -std=c++11 )


include(cmake/Dependencies.cmake) # opencv

set(wasabi2_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/)
set(wasabi2_SRC_DIR ${PROJECT_SOURCE_DIR}/src/)

add_subdirectory(src)
add_subdirectory(examples)

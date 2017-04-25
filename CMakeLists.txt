cmake_minimum_required(VERSION 3.6)
project(CleverResize)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
set(SOURCE_FILES main.cpp)
add_executable(CleverResize ${SOURCE_FILES})
target_link_libraries( CleverResize ${OpenCV_LIBS} )

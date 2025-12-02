cmake_minimum_required(VERSION 3.30)
project(AI_MLP)

set(CMAKE_CXX_STANDARD 20)

add_executable(AI_MLP main.cpp)


set(OpenCV_DIR "C:/OpenCV/build")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(AI_MLP ${OpenCV_LIBS})


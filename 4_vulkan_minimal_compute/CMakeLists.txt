cmake_minimum_required(VERSION 3.10)
project(vulkan_app)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(Vulkan REQUIRED)

include_directories(${Vulkan_INCLUDE_DIR})

include_directories(${Vulkan_INCLUDE_DIRS})

add_executable(vulkan_app main.cpp lodepng.cpp)

target_link_libraries(vulkan_app ${Vulkan_LIBRARY})
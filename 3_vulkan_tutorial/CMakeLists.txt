cmake_minimum_required(VERSION 3.10)
project(vulkan_app)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Set the build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)

include_directories(${Vulkan_INCLUDE_DIRS} ${GLFW_INCLUDE_DIRS})

add_executable(vulkan_app main.cpp)

target_link_libraries(vulkan_app ${Vulkan_LIBRARIES} glfw)
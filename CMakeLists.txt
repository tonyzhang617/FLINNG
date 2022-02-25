cmake_minimum_required (VERSION 3.10)
project(FLINNG)
set (CMAKE_CXX_STANDARD 11)

add_library(flinng SHARED ${PROJECT_SOURCE_DIR}/src/lib_flinng.cpp)
target_include_directories(flinng PUBLIC ${PROJECT_SOURCE_DIR}/include)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(flinng PUBLIC OpenMP::OpenMP_CXX)
endif()
cmake_minimum_required (VERSION 3.10)
project(FLINNG)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-O3 -ffast-math -Wall -Werror")

add_library(flinng SHARED ${PROJECT_SOURCE_DIR}/src/lib_flinng.cpp ${PROJECT_SOURCE_DIR}/src/LshFunctions.cpp)
target_include_directories(flinng PUBLIC ${PROJECT_SOURCE_DIR}/include)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(flinng PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(test ${PROJECT_SOURCE_DIR}/test/test_dense.cpp)
target_link_libraries(test flinng)

install(TARGETS flinng DESTINATION lib)
install(FILES ${PROJECT_SOURCE_DIR}/include/lib_flinng.h ${PROJECT_SOURCE_DIR}/include/io.h ${PROJECT_SOURCE_DIR}/include/Flinng.h ${PROJECT_SOURCE_DIR}/include/LshFunctions.h DESTINATION include)
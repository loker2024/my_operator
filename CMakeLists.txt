cmake_minimum_required(VERSION 3.24)

# "native" compiles for the CUDA-capable GPU on the configuration machine.
# Override it at configure time if a fixed target is needed, for example:
#   -DCMAKE_CUDA_ARCHITECTURES=89
# This must be set before project() so CMake does not fall back to nvcc's
# default architecture during CUDA compiler detection.
set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA GPU architecture(s)")

project(my_operator LANGUAGES CXX CUDA C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(Reduce Reduce/src/reduce.cu)
set_target_properties(Reduce PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(Sgemm Sgemm/src/main.cpp)

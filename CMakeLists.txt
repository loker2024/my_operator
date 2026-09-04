cmake_minimum_required(VERSION 3.24)

# CUDA 可选启用：先以 CXX 初始化工程，找到 Toolkit 后再开启 CUDA 语言，
# 这样在没有 GPU/CUDA 的环境下也能完成文档级配置。
project(my_operator LANGUAGES CXX)

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION} (${CUDAToolkit_BIN_DIR})")
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # 默认适配开发机：RTX 4060 Laptop（Ada，sm_89）。
    # 其他 GPU 可通过 -DCMAKE_CUDA_ARCHITECTURES=native 或 preset 覆盖。
    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures" FORCE)
  endif()
else()
  message(WARNING "CUDA Toolkit 未找到：跳过 CUDA 相关配置，仅保留工程骨架。")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 共享宿主工具（头文件库）
add_subdirectory(common)

# 算子目标：目录自带 CMakeLists.txt 并在此统一注册。
# 各算子 CMakeLists 内部判断 src/main.cu 是否存在：
#   不存在 → configure 时自动跳过；存在 → 创建与目录同名的可执行目标。
add_subdirectory(operators/softmax)
add_subdirectory(operators/gemm)
add_subdirectory(operators/attention)
add_subdirectory(operators/reduce)

cmake_minimum_required(VERSION 3.24)

# CUDA 可选启用：先以 CXX 初始化工程，找到 Toolkit 后再开启 CUDA 语言，
# 这样在没有 GPU/CUDA 的环境下也能完成文档级配置。
project(my_operator LANGUAGES CXX)

# toolkit 根目录兜底：若 PATH 里的 nvcc 是软链（如 /usr/local/bin/nvcc ->
# /usr/local/cuda-12.8/bin/nvcc），CMake 会按软链所在目录把根误判为 /usr/local，
# 进而找不到 include/cuda_runtime.h。这里先把 nvcc 解析到真实路径，仅当其上级目录
# 确实是 CUDA 安装根（含 include/cuda_runtime.h）时才作为提示；解析不到就保持原行为。
if(NOT CUDAToolkit_ROOT)
  find_program(MY_OPERATOR_NVCC_IN_PATH NAMES nvcc)
  if(MY_OPERATOR_NVCC_IN_PATH)
    get_filename_component(_nvcc_real "${MY_OPERATOR_NVCC_IN_PATH}" REALPATH)
    get_filename_component(_cuda_root "${_nvcc_real}" DIRECTORY)
    get_filename_component(_cuda_root "${_cuda_root}" DIRECTORY)
    if(EXISTS "${_cuda_root}/include/cuda_runtime.h")
      set(CUDAToolkit_ROOT "${_cuda_root}")
      # 编译器也指向真实路径：否则 enable_language(CUDA) 会再次按软链所在目录
      # 推导库根，报 "Couldn't find CUDA library root"。用户显式指定时以用户为准。
      if(NOT CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_COMPILER "${_nvcc_real}" CACHE FILEPATH "CUDA compiler")
      endif()
    endif()
  endif()
endif()

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

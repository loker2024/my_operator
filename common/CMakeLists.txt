# 共享宿主工具：仅包含头文件，供各算子 target 复用。
add_library(operator_common INTERFACE)
target_include_directories(operator_common INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
if(CUDAToolkit_FOUND)
  target_link_libraries(operator_common INTERFACE CUDA::cudart)
endif()

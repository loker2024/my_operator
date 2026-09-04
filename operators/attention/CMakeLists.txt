# Attention 算子目标（已由顶层 CMakeLists 注册）。
# 规则：src/main.cu 尚未创建时 configure 自动跳过；创建后自动构建。
# src/ 下新增任意 .cu 文件（CONFIGURE_DEPENDS 跟踪）都会自动编入该目标。
if(NOT CUDAToolkit_FOUND)
  message(STATUS "operators/attention: 跳过（未找到 CUDA）")
  return()
endif()

if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu)
  message(STATUS "operators/attention: src/main.cu 未创建，跳过（参考 README.md）")
  return()
endif()

file(GLOB ATTENTION_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)
add_executable(attention ${ATTENTION_SOURCES})
target_link_libraries(attention PRIVATE operator_common)

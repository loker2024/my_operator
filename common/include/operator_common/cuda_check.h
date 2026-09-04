#pragma once
// 公共 CUDA 工具：错误检查与设备信息。
// 引用方式：#include "operator_common/cuda_check.h"
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                              \
  do {                                                                \
    cudaError_t err = (expr);                                         \
    if (err != cudaSuccess) {                                         \
      std::fprintf(stderr, "CUDA error %s (%d) at %s:%d in %s: %s\n", \
                  cudaGetErrorName(err), static_cast<int>(err),       \
                  __FILE__, __LINE__, __func__,                       \
                  cudaGetErrorString(err));                           \
      std::exit(EXIT_FAILURE);                                        \
    }                                                                 \
  } while (0)

// 打印第一个可用设备的信息，帮助确认架构与选中的 sm 版本。
inline void PrintDeviceInfo() {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  std::printf("Device: %s\n", prop.name);
  std::printf("  compute capability : %d.%d\n", prop.major, prop.minor);
  std::printf("  SMs                : %d\n", prop.multiProcessorCount);
  std::printf("  shared mem / block  : %zu bytes\n", prop.sharedMemPerBlock);
  std::printf("  global mem         : %.1f GB\n",
              prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
}

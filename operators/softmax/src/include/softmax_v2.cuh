#pragma once
// softmax_v2.cuh —— 两级 warp shuffle 归约版本。
// 启动：grid=M，blockDim.x 为 32 的倍数且 <=1024，动态共享内存为 0。

#include <cuda_runtime.h>
__global__ void softmax_v2(const float* input, float* output, const int M, const int N);

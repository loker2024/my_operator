#pragma once
// softmax_v1.cuh —— 动态共享内存树形归约版本。
// 启动：grid=M，blockDim.x 为 2 的幂；动态共享内存为 blockDim.x*sizeof(float)。

#include <cuda_runtime.h>
__global__ void softmax_v1(const float* input, float* output, const int M, const int N);

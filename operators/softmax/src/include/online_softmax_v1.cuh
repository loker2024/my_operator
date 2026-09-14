#pragma once
// online_softmax_v1.cuh —— 每行一个 block 的在线归约版本。
// 启动：grid=M（M=0 时至少 1），blockDim.x 为 32 的倍数且 <=1024，动态共享内存为 0。

#include <cuda_runtime.h>

__global__ void online_softmax_v1(const float* input, float* output, const int M, const int N);

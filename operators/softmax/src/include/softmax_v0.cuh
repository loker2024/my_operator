#pragma once
// softmax_v0.cuh —— 每线程处理一行的三遍标量 Softmax 基线。
// 启动：grid=ceil(M/blockDim.x)，动态共享内存为 0，N 可为任意非负数。

#include <cuda_runtime.h>
__global__ void softmax_v0(const float* input, float* output, const int M, const int N);

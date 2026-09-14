#pragma once
// online_softmax_v0.cuh —— 每线程独占一行的 online softmax 基线。
// 启动：grid=ceil(M/blockDim.x)，无动态共享内存；N 可为任意非负数，越界线程空转。

#include <cuda_runtime.h>

__global__ void online_softmax_v0(const float* input, float* output, const int M, const int N);

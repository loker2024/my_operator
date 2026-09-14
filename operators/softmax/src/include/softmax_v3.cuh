#pragma once
// softmax_v3.cuh —— v2 的 float4 向量化版本；N%4==0 走向量路径，否则标量回退。
// 启动约束同 v2，动态共享内存为 0。

#include <cuda_runtime.h>
__global__ void softmax_v3(const float* input, float* output, const int M, const int N);

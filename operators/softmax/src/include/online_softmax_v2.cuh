#pragma once
// online_softmax_v2.cuh —— v1 的 float4 向量化版本。
// 启动约束同 v1；N%4==0 时走 16B 对齐 float4 路径，否则整行回退标量路径。

#include <cuda_runtime.h>

__global__ void online_softmax_v2(const float* input, float* output, const int M, const int N);

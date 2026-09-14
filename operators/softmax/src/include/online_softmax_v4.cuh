#pragma once
// online_softmax_v4.cuh —— grid-stride 多行的 online softmax 版本。
// 启动：grid 由调用方限制为 min(M, SM数×系数)，blockDim.x 为 32 的倍数且 <=1024，
// 无动态共享内存；N%4==0 走 float4，否则走标量路径。

#include <cuda_runtime.h>

__global__ void online_softmax_v4(const float* input, float* output, const int M, const int N);

#pragma once
// online_softmax_v3.cuh —— 真寄存器分片缓存版本。
// 启动约束同 v1；ceil(N/blockDim.x)<=16 时缓存并免第二次全局读，超过时回退重读路径。

#include <cuda_runtime.h>

__global__ void online_softmax_v3(const float* input, float* output, const int M, const int N);

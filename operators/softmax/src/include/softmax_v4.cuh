#pragma once
// softmax_v4.cuh —— 一遍全局读、整行 x 动态共享内存缓存版本。
// 启动约束同 v2；动态共享内存为 N*sizeof(float)，须不超过每 block 上限。

#include <cuda_runtime.h>
__global__ void softmax_v4(const float* input, float* output, const int M, const int N);

#pragma once
// softmax_v5.cuh —— 两遍全局读、整行 exp 动态共享内存缓存版本。
// 启动约束与动态共享内存需求同 v4；N%4==0 时三遍 float4，否则标量回退。

#include <cuda_runtime.h>
__global__ void softmax_v5(const float* input, float* output, const int M, const int N);

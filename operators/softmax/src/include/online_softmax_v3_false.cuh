#pragma once
// online_softmax_v3_false.cuh —— 运行期下标 local-memory 缓存对照版本。
// 启动约束同 v1；调用方必须保证 ceil(N/blockDim.x)<=16，本版本不提供回退路径。

#include <cuda_runtime.h>

__global__ void online_softmax_v3_false(const float* input, float* output, const int M,
                                        const int N);

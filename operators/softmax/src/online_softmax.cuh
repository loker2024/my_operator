#pragma once
// ============================================================================
// online_softmax.cuh —— online softmax 接口声明（行主序 fp32 矩阵，逐行归一化）
//   独立实现单元：实现见 online_softmax.cu，测试驱动见 test.cuh/test.cu，注册与
//   场景见 main.cu；推导与实测见 operators/softmax/README.md。
//
//   与 softmax.cuh（v0~v5）的差异：v0~v5 是「先求行最大 m、再求 Σexp」的两趟归约，
//   本单元改单趟 online 归约 —— 用二元组 (m, d) 在同一趟遍历里增量维护行最大与
//   分母：m' = max(m, x)、d' = d * exp(m - m') + exp(x - m')（d 先按新旧基准之差
//   缩放回新基准，再加新元素贡献），把求 m 与求 Σexp 合并为一趟全局读。
// ============================================================================

#include <cuda_runtime.h>  // __global__、cudaError_t 等 CUDA 基本定义

// online_softmax_v0 每线程处理一行：
//   * row = blockIdx.x * blockDim.x + threadIdx.x；行内单趟 online 归约求 (m, d)，
//     再第二趟读行重算 exp(x - m) / d 写回；
//   * grid = ceil(M / blockDim.x)（M == 0 时也须 >= 1，由 row >= M 越界空转）；
//   * 无共享内存 / 同步（smem_bytes = 0）；行宽 N 任意，N == 0 时空行不读不写；
//   * warp 内各线程处理不同行 → 访存不合并，行内 fp32 串行累加误差随行宽增长。
__global__ void online_softmax_v0(const float* input, float* output, const int M, const int N);

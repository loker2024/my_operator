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
//
//   单元内两个版本的差别只在行内/块内的协作方式：v0 每线程独占一行（无协作），
//   v1 每行一个 block、块内两级 warp shuffle 合并各线程的 (m, d)（读合并，是
//   后续「分块 + 在线合并」的 FlashAttention 形态的雏形）。
// ============================================================================

#include <cuda_runtime.h>  // __global__、cudaError_t 等 CUDA 基本定义

// online_softmax_v0 每线程处理一行：
//   * row = blockIdx.x * blockDim.x + threadIdx.x；行内单趟 online 归约求 (m, d)，
//     再第二趟读行重算 exp(x - m) / d 写回；
//   * grid = ceil(M / blockDim.x)（M == 0 时也须 >= 1，由 row >= M 越界空转）；
//   * 无共享内存 / 同步（smem_bytes = 0）；行宽 N 任意，N == 0 时空行不读不写；
//   * warp 内各线程处理不同行 → 访存不合并，行内 fp32 串行累加误差随行宽增长。
__global__ void online_softmax_v0(const float* input, float* output, const int M, const int N);

// online_softmax_v1 每行一个 block，行内在线归约 + 块内两级 warp shuffle 合并：
//   * row = blockIdx.x，grid = M（M == 0 时也须 >= 1，由 row >= M 越界空转）；
//   * 每线程以 stride = blockDim.x 单趟在线归约自己的列子集得局部 (m, d)，再由
//     两级 warp shuffle 合并出整行 (m, d) —— 每 warp 先 shfl 归约为 1 个二元组，
//     warp 0 再合并 num_warps 个 warp 值并广播回全体（归约语义同 softmax.cuh
//     v2）；warp 内同轮访问相邻列 → 全局读合并；
//   * blockDim.x 须为 32 的倍数且 <= 1024（默认 256）：shuffle 需整 warp 收敛，
//     且各 warp 的二元组要装进中转用的 warp_m[32] / warp_d[32]；中转用静态
//     __shared__，无动态共享内存（smem_bytes = 0）；
//   * 未分到元素的线程 / warp（N 不整除或小于 blockDim.x）以 (m = -inf, d = 0)
//     作归约单位元参与合并 —— 合并时须跳过该侧的缩放，否则 (-inf) - (-inf) 的
//     NaN 会经 expf 污染整行分母；N == 0 的空行不读不写。
__global__ void online_softmax_v1(const float* input, float* output, const int M, const int N);

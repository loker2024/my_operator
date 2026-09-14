#pragma once
// ============================================================================
// sgemm_v0.cuh —— SGEMM v0 朴素基线的公开接口。
// 计算行主序 C(M×N) = A(M×K) × B(K×N)。启动：block=(16,16)，
// grid=(ceil(M/16), ceil(N/16))，动态共享内存为 0；M/N 边界线程空转。
// 注意：K 可为 0（对应输出写 0），累加使用 fp32，长 K 的舍入误差由测试按 1e-3
// 容差校验。
// ============================================================================

#include <cuda_runtime.h>

__global__ void sgemm_v0(const float* A, const float* B, float* C, int M, int N, int K);

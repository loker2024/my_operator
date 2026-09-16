#pragma once
// ============================================================================
// sgemm_v0.cuh —— SGEMM v0 朴素内核的公开接口：行主序 C(M×N) = A(M×K) × B(K×N)，
// 每线程一个输出元素；实现在 sgemm_v0.cu。
// ============================================================================

#include <cuda_runtime.h>

__global__ void sgemm_v0(const float* A, const float* B, float* C, int M, int N, int K);

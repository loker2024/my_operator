#pragma once
// ============================================================================
// sgemm_reference.cuh —— GEMM 测试共用的内核地址类型与行主序 CPU 参考接口
// （实现见 sgemm_reference.cu）。
// ============================================================================

// 内核以裸函数地址交给 cudaLaunchKernel 启动。
using GemmKernel = const void*;

void sgemm_cpu(const float* A, const float* B, float* C, int M, int N, int K);

#pragma once
// ============================================================================
// sgemm_reference.cuh —— GEMM 测试共用的内核函数类型与 CPU 参考接口。
// CPU 参考独立以 double 累加，输入和输出均是行主序；实现见 sgemm_reference.cu。
// ============================================================================

using GemmKernel = void (*)(const float* A, const float* B, float* C, int M, int N, int K);

void sgemm_cpu(const float* A, const float* B, float* C, int M, int N, int K);

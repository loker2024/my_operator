#pragma once
// ============================================================================
// sgemm_reference.cuh —— GEMM 测试共用的 CUDA 内核地址类型与 CPU 参考接口。
// 内核地址交给 cudaLaunchKernel 统一启动，允许测试不同模板/参数签名的教学版本；CPU 参考
// 独立以 double 累加，输入和输出均是行主序；实现见 sgemm_reference.cu。
// ============================================================================

using GemmKernel = const void*;

void sgemm_cpu(const float* A, const float* B, float* C, int M, int N, int K);

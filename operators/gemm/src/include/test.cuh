#pragma once
// ============================================================================
// test.cuh —— GEMM 可复用测试驱动接口（实现见 test.cu）。
// 对任意符合 GemmKernel 签名的内核执行确定性输入、CPU double 参考、CUDA event
// 计时、逐元素相对误差比较与英文报告。调用方负责按版本契约传入 grid/block/smem。
// 正确性容差为 1e-3；false 使用 1 次预热和 100 次计时，true 使用 100 次预热及
// 21 组 × 1000 次计时。非法尺寸或启动配置返回 false。
// ============================================================================

#include <cstddef>

#include "sgemm_reference.cuh"

bool test_gemm_kernel(GemmKernel kernel, const char* kernel_name, int M, int N, int K, int grid_x,
                      int grid_y, int block_x, int block_y, std::size_t smem_bytes = 0,
                      bool strict_benchmark = false);

// 以 cuBLAS 的严格 FP32 数学模式执行行主序 SGEMM 对照。参数 M/N/K 定义
// C(M×N)=A(M×K)×B(K×N)，返回值表示 CPU 参考比较是否通过；无 CUDA 启动配置。
bool test_cublas_sgemm(const char* test_name, int M, int N, int K, bool strict_benchmark = false);

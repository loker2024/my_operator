#pragma once
// ============================================================================
// test.cuh —— GEMM 可复用测试驱动接口（实现见 test.cu）：正确性校验与计时报告。
// ============================================================================

#include <cstddef>

#include "sgemm_reference.cuh"

// strict_benchmark = false 时 1 次预热 + 100 次计时，true 时 100 次预热 + 21 组 × 1000 次。
// 形状或启动配置非法、正确性不通过时返回 false。
bool test_gemm_kernel(GemmKernel kernel, const char* kernel_name, int M, int N, int K, int grid_x,
                      int grid_y, int block_x, int block_y, std::size_t smem_bytes = 0,
                      bool strict_benchmark = false);

// cuBLAS 对照（CUBLAS_PEDANTIC_MATH），返回 CPU 参考比较是否通过。
bool test_cublas_sgemm(const char* test_name, int M, int N, int K, bool strict_benchmark = false);

// 扫描模式（bench）：只统计耗时，不生成输入、不做 CPU 参考、不回拷结果。返回单次调用耗时的
// 中位数（ms）；参数非法、尺寸为 0 或内核启动失败时返回负数，由调用方跳过该采样点。
// iters_out 非空时回传每组迭代数（按 budget_ms 自适应）。

// 内核路径：kernel 为 __global__ 函数地址，grid/block/smem 契约与 test_gemm_kernel 相同。
double bench_gemm_kernel(const void* kernel, int M, int N, int K, int grid_x, int grid_y,
                         int block_x, int block_y, std::size_t smem_bytes, int warmup_iterations,
                         double budget_ms, int* iters_out = nullptr);

// cuBLAS 对照路径：固定 CUBLAS_PEDANTIC_MATH，返回契约同 bench_gemm_kernel。
double bench_cublas_sgemm(int M, int N, int K, int warmup_iterations, double budget_ms,
                          int* iters_out = nullptr);

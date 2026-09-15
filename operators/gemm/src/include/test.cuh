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

// ---------------------------------------------------------------------------
// 扫描模式（bench）：只分配设备缓冲并做 CUDA event 计时，不生成输入、不做 CPU 参考、
// 不回拷结果 —— 跨尺寸性能曲线只关心耗时，逐点校验会让大尺寸（如 4096³）慢到不可用。
// 返回单次调用耗时的中位数（ms）；参数非法、尺寸为 0 或内核启动失败时返回负数，
// 由调用方跳过该采样点。iters_out 非空时回传实际每组迭代数。
// 迭代数按预算自适应：先用 1 次调用估计单次耗时，再取每组迭代数 = 预算内可容纳的
// 调用数，限制在 [1, 100]，连续采 3 组取中位数 —— 固定 100 次迭代会让慢内核（如
// 朴素 v0 在 4096³ 下约 1 s/次）单点就耗掉数分钟。
// ---------------------------------------------------------------------------

// 内核路径：kernel 为 __global__ 函数地址，grid/block/smem 契约与 test_gemm_kernel 相同。
double bench_gemm_kernel(const void* kernel, int M, int N, int K, int grid_x, int grid_y,
                         int block_x, int block_y, std::size_t smem_bytes, int warmup_iterations,
                         double budget_ms, int* iters_out = nullptr);

// cuBLAS 对照路径：固定 CUBLAS_PEDANTIC_MATH，参数含义与 test_cublas_sgemm 相同。
double bench_cublas_sgemm(int M, int N, int K, int warmup_iterations, double budget_ms,
                          int* iters_out = nullptr);

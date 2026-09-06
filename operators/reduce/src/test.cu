// ============================================================================
// test.cu —— 一维归约算子的可复用测试驱动：实现
//
// 与 test.cuh 配对：本文件包含其中声明的 test_reduce_kernel 的实现。
// 执行入口不在本文件：由 src/main.cu 调用本函数完成测试。
//
// 实现要点（详见 test.cuh 的声明注释）：
//   1) 以函数指针接收任意符合 ReduceKernel 签名的归约内核；
//   2) 用 cudaLaunchKernel 启动，从而一份测试代码驱动所有版本；
//   3) 正确性：主机 double 累加参考 vs 各 block 部分和二次汇总，容差 1e-3；
//   4) 性能：GpuTimer(event) 分组采样取中位数，报告有效带宽。
//
// 被测算子实现见 reduce.cu（reduce_v0 等），接口见 reduce.cuh。
// ============================================================================

#include <algorithm>  // std::sort：对计时样本排序求中位数
#include <cmath>      // std::fabs
#include <cstdio>     // printf
#include <vector>     // std::vector：测试数据与计时样本容器

#include "operator_common/cuda_check.h"  // CUDA_CHECK 错误检查宏
#include "operator_common/GpuTimer.h"    // 基于 CUDA event 的 GPU 计时

#include "test.cuh"  // test_reduce_kernel 声明（间接引入 reduce.cuh）

// ============================================================================
// test_reduce_kernel —— 可复用归约测试驱动
// ============================================================================
bool test_reduce_kernel(ReduceKernel kernel, const char* kernel_name, int n,
                        int grid, int block, bool strict_benchmark) {
  // 迭代口径（见 test.cuh / docs §3.2）：开发阶段 vs 严格阶段。
  const int warmup_iterations = strict_benchmark ? 1000 : 1;
  const int iterations = strict_benchmark ? 10000 : 100;
  const int sample_count = strict_benchmark ? 21 : 1;

  // --- 1. 生成可复现的测试数据，并计算 CPU 参考值 --------------------------
  // 输入取 (i % 1000)：确定性生成、可复现（等价固定种子伪随机），
  // 同时保证和为较大的正数，避免求和因正负抵消而退化、放大相对误差。
  std::vector<float> h_input(n);
  double h_ref = 0.0;  // 高精度参考：double 累加，误差可忽略
  for (int i = 0; i < n; ++i) {
    h_input[i] = static_cast<float>(i % 1000);
    h_ref += h_input[i];
  }

  // --- 2. 分配设备内存并拷入输入 ------------------------------------------
  // d_output 长度 = grid：每个 block 恰好写 1 个部分和。
  float *d_input = nullptr, *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, grid * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));

  // --- 3. 准备启动配置（统一签名 + 统一动态共享内存） ----------------------
  const dim3 grid_dim(grid);
  const dim3 block_dim(block);
  const size_t smem_bytes = block * sizeof(float);  // v0 的动态共享内存需求
  // cudaLaunchKernel 要求 args 中每一项是对应“内核形参地址”的指针，
  // 因此需要与形参 const 修饰严格匹配的中间变量：
  //   d_input_arg 是 const float*，恰好对应内核形参 const float* input。
  const float* d_input_arg = d_input;
  float* d_output_arg = d_output;
  int n_arg = n;
  void* args[] = {&d_input_arg, &d_output_arg, &n_arg};

  // 用 cudaLaunchKernel 而非 <<< >>>：这样才能把“运行期内核函数指针”
  // 作为参数传给本函数，从而一份测试代码驱动所有版本。
  auto launch = [&]() {
    CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), grid_dim,
                                block_dim, args, smem_bytes));
  };

  // --- 4. 预热：消除冷启动 / 驱动初始化 / 模块加载影响 ----------------------
  for (int i = 0; i < warmup_iterations; ++i) {
    launch();
  }
  CUDA_CHECK(cudaDeviceSynchronize());  // 确认预热内核真正执行完毕

  // --- 5. 性能采样：每组连续 iterations 次，记录“单次内核平均耗时” ----------
  // 每组独立计时，最终取中位数（避免离群值带偏），并给出 P5/P95 波动区间。
  std::vector<float> samples_ms(sample_count);
  for (int s = 0; s < sample_count; ++s) {
    GpuTimer timer;  // 内部为 CUDA event，StopMs 前隐式同步设备
    timer.Start();
    for (int i = 0; i < iterations; ++i) {
      launch();
    }
    samples_ms[s] = timer.StopMs() / static_cast<float>(iterations);
  }
  std::vector<float> sorted_ms = samples_ms;
  std::sort(sorted_ms.begin(), sorted_ms.end());
  const float median_ms = sorted_ms[sample_count / 2];      // 主指标：中位数
  const float p5_ms = sorted_ms[(sample_count - 1) * 5 / 100];
  const float p95_ms = sorted_ms[(sample_count - 1) * 95 / 100];

  // 有效带宽 =（读入 n 个 float + 写出 grid 个部分和）/ 中位耗时
  // （GB/s 口径，1 GB = 1e9 bytes；ms * 1e6 = ms·1e-3s·1e9，单位换算一次完成）
  const double bytes_per_run =
      (static_cast<double>(n) + static_cast<double>(grid)) * sizeof(float);
  const double bandwidth_gbps = bytes_per_run / (median_ms * 1e6);

  // --- 6. 拷回部分和，主机侧做最后一次归约，并与 CPU 参考对比 ---------------
  std::vector<float> h_partial(grid);
  CUDA_CHECK(cudaMemcpy(h_partial.data(), d_output, grid * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // 各 block 的部分和都是 fp32，汇总时用 double 累加，避免二次误差。
  double gpu_sum = 0.0;
  for (int i = 0; i < grid; ++i) {
    gpu_sum += h_partial[i];
  }

  // CPU 朴素基线：独立于测试数据的另一份参考（reduce_cpu 本身也一并校验）。
  const float cpu_sum = reduce_cpu(h_input.data(), n);

  // 判据（docs/benchmark-methodology.md）：相对误差 <= 1e-3，且无 NaN/Inf。
  // 用 !(rel <= tol) 判定：NaN/Inf/超差都会被记为失败。
  const double abs_err = std::fabs(gpu_sum - h_ref);
  const double rel_err = abs_err / (std::fabs(h_ref) + 1e-30);
  const bool pass = rel_err <= 1e-3;

  // --- 7. 输出报告 ----------------------------------------------------------
  std::printf("[%s] n=%d, grid=%d, block=%d\n", kernel_name, n, grid, block);
  std::printf("    CPU 参考值    = %.4f (reduce_cpu 基线 %.4f)\n", h_ref, cpu_sum);
  std::printf("    GPU 归约结果  = %.4f\n", gpu_sum);
  std::printf("    相对误差      = %.3e (容差 1e-3)  ->  %s\n", rel_err,
              pass ? "PASS" : "FAIL");
  if (strict_benchmark) {
    std::printf("    基准(严格)    = %d 次预热, %d 组 x %d 次\n", warmup_iterations,
                sample_count, iterations);
    std::printf("    耗时(中位数)  = %.4f ms (P5 %.4f, P95 %.4f)\n", median_ms, p5_ms,
                p95_ms);
  } else {
    std::printf("    耗时          = %.4f ms/kernel (%d 次)\n", median_ms, iterations);
  }
  std::printf("    有效带宽      = %.2f GB/s\n", bandwidth_gbps);

  // --- 8. 释放资源 -----------------------------------------------------------
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  return pass;
}

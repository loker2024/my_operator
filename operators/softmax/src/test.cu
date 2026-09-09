// test.cu —— test_softmax_kernel 的实现（接口见 test.cuh），执行入口见 main.cu。

#include <algorithm>  // std::sort
#include <cmath>      // std::fabs / std::isnan / std::isinf
#include <cstddef>    // std::size_t
#include <cstdint>    // std::int64_t
#include <cstdio>     // printf
#include <vector>     // std::vector

#include "operator_common/cuda_check.h"  // CUDA_CHECK 错误检查宏
#include "operator_common/GpuTimer.h"    // 基于 CUDA event 的 GPU 计时

#include "test.cuh"

bool test_softmax_kernel(SoftmaxKernel kernel, const char* kernel_name, int rows,
                         int cols, int grid, int block, std::size_t smem_bytes,
                         bool strict_benchmark) {
  // 迭代口径：开发（默认）vs 严格两档，与 reduce 测试文件一致
  //（见 docs/benchmark-methodology.md）。
  const int warmup_iterations = strict_benchmark ? 1000 : 1;
  const int iterations = strict_benchmark ? 10000 : 100;
  const int sample_count = strict_benchmark ? 21 : 1;

  // 契约防御：非法参数判 FAIL 而非崩溃（rows/cols < 0 无意义，cudaMalloc(0)
  // 未定义，grid/block < 1 无法启动）。
  if (rows < 0 || cols < 0 || grid < 1 || block < 1) {
    std::printf("[%s] 非法参数: rows=%d, cols=%d, grid=%d, block=%d "
                "(契约: rows>=0, cols>=0, grid>=1, block>=1)  ->  FAIL\n",
                kernel_name, rows, cols, grid, block);
    return false;
  }

  // 元素总数。启动网格按被测内核的行映射由调用方给出（见 softmax.cuh）：v0
  // 线程铺满行号、v1/v2 每行一个 block；空矩阵 rows == 0 时 grid 也须 >= 1，内核
  // 由 row >= rows 越界判定空转。
  const std::int64_t count = static_cast<std::int64_t>(rows) * cols;

  // 输入取确定性伪随机（行/列相关），范围 [-10, 10)：可复现，且不会让 exp 溢出
  // / 下溢；max-shift 后 exp 参数 <= 0 恒成立，行和 >= 1（含 exp(0) 项），避免
  // 极小分母放大相对误差。CPU 参考 softmax_cpu 用 double 计算，避免参考值自身的
  // 舍入误差吃掉 1e-5 的容差。
  std::vector<float> h_input(static_cast<std::size_t>(count));
  for (std::int64_t i = 0; i < count; ++i) {
    const int r = static_cast<int>(i / cols);  // cols == 0 时 count == 0，不进入
    const int c = static_cast<int>(i % cols);
    const int v = ((r * 31 + c * 17) % 2001) - 1000;  // [-1000, 1000] -> [-10, 10]
    h_input[static_cast<std::size_t>(i)] = static_cast<float>(v) / 100.0f;
  }
  std::vector<float> h_ref(static_cast<std::size_t>(count));
  softmax_cpu(h_input.data(), h_ref.data(), rows, cols);

  // 空矩阵/空行时 cudaMalloc(0) 未定义，统一按 max(count, 1) 申请；内核该路径全走
  // 越界/空行分支、不读写数据，输出缓冲区保持原样（无可比元素，天然通过）。
  float *d_input = nullptr, *d_output = nullptr;
  const size_t in_bytes = static_cast<std::size_t>(
                              (count > 0 ? count : 1)) *
                          sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_input, in_bytes));
  CUDA_CHECK(cudaMalloc(&d_output, in_bytes));
  if (count > 0) {  // count == 0 时无可拷数据
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), in_bytes, cudaMemcpyHostToDevice));
  }

  // 只有 cudaLaunchKernel 才能以“运行期内核函数指针”启动，从而一份驱动复用所有
  // 版本；args 中需放与形参 const 修饰严格匹配的指针。grid/block/smem_bytes 按
  // 被测内核的映射给出（v0/v2/v3 无动态共享内存 smem_bytes = 0 —— v2/v3 仅用
  // 内部静态 __shared__ 中转；v1 为 blockDim.x 个 float）。
  const dim3 grid_dim(grid);
  const dim3 block_dim(block);
  const float* d_input_arg = d_input;
  float* d_output_arg = d_output;
  int rows_arg = rows;
  int cols_arg = cols;
  void* args[] = {&d_input_arg, &d_output_arg, &rows_arg, &cols_arg};
  auto launch = [&]() {
    CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), grid_dim,
                                block_dim, args, smem_bytes));
  };

  // 预热：消除冷启动 / 驱动初始化 / 模块加载影响；空形状不做计时（无有意义数据量）。
  for (int i = 0; i < warmup_iterations; ++i) {
    launch();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> samples_ms(sample_count);
  if (count > 0) {
    // 采样：每组连续 iterations 次取平均，最终取中位数，避免离群值带偏。
    for (int s = 0; s < sample_count; ++s) {
      GpuTimer timer;  // 内部为 CUDA event，StopMs 前隐式同步
      timer.Start();
      for (int i = 0; i < iterations; ++i) {
        launch();
      }
      samples_ms[s] = timer.StopMs() / static_cast<float>(iterations);
    }
  }
  std::vector<float> sorted_ms = samples_ms;
  std::sort(sorted_ms.begin(), sorted_ms.end());
  const float median_ms = sorted_ms[sample_count / 2];  // 主指标：中位数
  const float p5_ms = sorted_ms[(sample_count - 1) * 5 / 100];
  const float p95_ms = sorted_ms[(sample_count - 1) * 95 / 100];

  // 有效带宽 =（输入读 rows*cols 个 float + 输出写 rows*cols 个 float）/ 中位耗时
  //（逻辑数据量，每元素计 1 读 1 写；各版本实际都三遍各读行一次、访问模式
  // 各异，低效会直接反映为更低的“有效带宽”，见 README）。
  const double bytes_per_run = 2.0 * static_cast<double>(count) * sizeof(float);
  const double bandwidth_gbps = bytes_per_run / (median_ms * 1e6);

  // 拷回 GPU 输出，逐元素比对。
  std::vector<float> h_gpu(static_cast<std::size_t>(count));
  if (count > 0) {
    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_output, in_bytes, cudaMemcpyDeviceToHost));
  }

  // 判据：相对误差 <= 1e-5（逐元素），跳过 |ref| 过小的元素（< 1e-30，相对比较
  // 无意义，见 docs/benchmark-methodology.md）；任一 NaN/Inf 立即记为不合规。
  constexpr double kRelTol = 1e-5;
  constexpr double kSkipAbs = 1e-30;
  double max_rel_err = 0.0;
  std::int64_t worst_idx = -1;
  std::int64_t bad_count = 0;     // 超差或 NaN/Inf 的元素数
  std::int64_t skipped = 0;       // |ref| 过小被跳过的元素数
  for (std::int64_t i = 0; i < count; ++i) {
    const double ref = h_ref[static_cast<std::size_t>(i)];
    const double got = h_gpu[static_cast<std::size_t>(i)];
    if (std::isnan(got) || std::isinf(got)) {
      ++bad_count;  // GPU 出现 NaN/Inf 是硬错误
      continue;
    }
    const double denom = std::fabs(ref);
    if (denom < kSkipAbs) {
      ++skipped;
      continue;
    }
    const double rel_err = std::fabs(got - ref) / denom;
    if (rel_err > max_rel_err) {
      max_rel_err = rel_err;
      worst_idx = i;
    }
    if (rel_err > kRelTol) {
      ++bad_count;
    }
  }
  const bool pass = (bad_count == 0);

  // 报告。
  std::printf("[%s] rows=%d, cols=%d, grid=%d, block=%d\n", kernel_name, rows,
              cols, grid, block);
  if (count > 0) {
    std::printf("    样本 out[0][0]  = GPU %.4f / CPU 参考 %.4f\n", h_gpu[0],
                h_ref[0]);
    std::printf("    相对误差(max)  = %.3e (容差 1e-5)  ->  %s\n", max_rel_err,
                pass ? "PASS" : "FAIL");
    if (skipped > 0) {
      std::printf("    跳过 |ref|<1e-30 的元素 %lld 个\n",
                  static_cast<long long>(skipped));
    }
    if (!pass) {
      std::printf("    不合规元素数    = %lld 个 (超差或 NaN/Inf)\n",
                  static_cast<long long>(bad_count));
      if (worst_idx >= 0) {
        // 打印最大误差位置及其前后元素，便于定位。
        const std::int64_t from = worst_idx > 2 ? worst_idx - 2 : 0;
        const std::int64_t to = worst_idx + 3 < count ? worst_idx + 3 : count;
        for (std::int64_t j = from; j < to; ++j) {
          std::printf("      [%lld] GPU %.6e / CPU %.6e\n",
                      static_cast<long long>(j),
                      h_gpu[static_cast<std::size_t>(j)],
                      h_ref[static_cast<std::size_t>(j)]);
        }
      }
    }
    if (strict_benchmark) {
      std::printf("    基准(严格)      = %d 次预热, %d 组 x %d 次\n",
                  warmup_iterations, sample_count, iterations);
      std::printf("    耗时(中位数)    = %.4f ms (P5 %.4f, P95 %.4f)\n", median_ms,
                  p5_ms, p95_ms);
    } else {
      std::printf("    耗时            = %.4f ms/kernel (%d 次)\n", median_ms,
                  iterations);
    }
    std::printf("    有效带宽        = %.2f GB/s (逻辑数据量: 读1次+写1次)\n",
                bandwidth_gbps);
  } else {
    std::printf("    空矩阵/空行: 无输出元素可比，内核空转  ->  %s\n",
                pass ? "PASS" : "FAIL");
  }

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  return pass;
}

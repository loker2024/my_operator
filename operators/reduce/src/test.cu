// test.cu —— test_reduce_kernel 的实现（接口见 test.cuh），执行入口见 main.cu。

#include <algorithm>  // std::sort
#include <cmath>      // std::fabs
#include <cstdio>     // printf
#include <vector>     // std::vector

#include "operator_common/GpuTimer.h"    // 基于 CUDA event 的 GPU 计时
#include "operator_common/cuda_check.h"  // CUDA_CHECK 错误检查宏
#include "test.cuh"

bool test_reduce_kernel(ReduceKernel kernel, const char* kernel_name, int n, int grid, int block,
                        bool strict_benchmark) {
	// 迭代口径：开发（默认）vs 严格两档，见 docs/benchmark-methodology.md。严格档
	// 面向本机 RTX 4060 Laptop 调低采样量（100 预热 + 21 组 × 1000 次），21 组保留
	// P5/P95 分位分辨率。
	const int warmup_iterations = strict_benchmark ? 100 : 1;
	const int iterations = strict_benchmark ? 1000 : 100;
	const int sample_count = strict_benchmark ? 21 : 1;

	// 契约防御：非法参数判 FAIL 而非崩溃（n<0 无意义，cudaMalloc(d_output, 0) 未定义）。
	if (n < 0 || grid < 1) {
		std::printf("[%s] 非法参数: n=%d, grid=%d (契约: n>=0 且 grid>=1)  ->  FAIL\n", kernel_name,
		            n, grid);
		return false;
	}

	// 输入取 (i % 1000)：确定性可复现，且和为较大的正数，避免正负抵消放大相对误差。
	std::vector<float> h_input(n);
	double h_ref = 0.0;  // double 累加的高精度参考
	for (int i = 0; i < n; ++i) {
		h_input[i] = static_cast<float>(i % 1000);
		h_ref += h_input[i];
	}

	// d_output 长 grid：每 block 写 1 个部分和。n == 0 时 cudaMalloc(0) 未定义，
	// 输入按 max(n,1) 申请；内核此时全走“越界补 0”分支、不会读 d_input，结果恒为 0。
	float *d_input = nullptr, *d_output = nullptr;
	const size_t in_bytes = static_cast<size_t>(std::max(n, 1)) * sizeof(float);
	CUDA_CHECK(cudaMalloc(&d_input, in_bytes));
	CUDA_CHECK(cudaMalloc(&d_output, static_cast<size_t>(grid) * sizeof(float)));
	if (n > 0) {  // n == 0 时无可拷数据
		CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), static_cast<size_t>(n) * sizeof(float),
		                      cudaMemcpyHostToDevice));
	}

	// 只有 cudaLaunchKernel 才能以“运行期内核函数指针”启动，从而一份驱动复用所有
	// 版本；args 中需放与形参 const 修饰严格匹配的指针。
	const dim3 grid_dim(grid);
	const dim3 block_dim(block);
	const size_t smem_bytes = block * sizeof(float);
	const float* d_input_arg = d_input;
	float* d_output_arg = d_output;
	int n_arg = n;
	void* args[] = {&d_input_arg, &d_output_arg, &n_arg};
	auto launch = [&]() {
		CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), grid_dim, block_dim,
		                            args, smem_bytes));
	};

	// 预热：消除冷启动 / 驱动初始化 / 模块加载影响。
	for (int i = 0; i < warmup_iterations; ++i) {
		launch();
	}
	CUDA_CHECK(cudaDeviceSynchronize());

	// 采样：每组连续 iterations 次取平均，最终取中位数，避免离群值带偏。
	std::vector<float> samples_ms(sample_count);
	for (int s = 0; s < sample_count; ++s) {
		GpuTimer timer;  // 内部为 CUDA event，StopMs 前隐式同步
		timer.Start();
		for (int i = 0; i < iterations; ++i) {
			launch();
		}
		samples_ms[s] = timer.StopMs() / static_cast<float>(iterations);
	}
	std::vector<float> sorted_ms = samples_ms;
	std::sort(sorted_ms.begin(), sorted_ms.end());
	const float median_ms = sorted_ms[sample_count / 2];  // 主指标：中位数
	const float p5_ms = sorted_ms[(sample_count - 1) * 5 / 100];
	const float p95_ms = sorted_ms[(sample_count - 1) * 95 / 100];

	// 有效带宽 =（读入 n 个 float + 写出 grid 个部分和）/ 中位耗时（GB/s，1 GB = 1e9）。
	const double bytes_per_run =
	    (static_cast<double>(n) + static_cast<double>(grid)) * sizeof(float);
	const double bandwidth_gbps = bytes_per_run / (median_ms * 1e6);

	// 拷回部分和，主机侧 double 二次汇总（避免部分和再次舍入）；
	// reduce_cpu 作为独立基线一并校验。
	std::vector<float> h_partial(grid);
	CUDA_CHECK(
	    cudaMemcpy(h_partial.data(), d_output, grid * sizeof(float), cudaMemcpyDeviceToHost));
	double gpu_sum = 0.0;
	for (int i = 0; i < grid; ++i) {
		gpu_sum += h_partial[i];
	}
	const float cpu_sum = reduce_cpu(h_input.data(), n);

	// 判据：相对误差 <= 1e-3；NaN/Inf/超差都会记为失败。
	const double abs_err = std::fabs(gpu_sum - h_ref);
	const double rel_err = abs_err / (std::fabs(h_ref) + 1e-30);
	const bool pass = rel_err <= 1e-3;

	// 报告。
	std::printf("[%s] n=%d, grid=%d, block=%d\n", kernel_name, n, grid, block);
	std::printf("    CPU 参考值    = %.4f (reduce_cpu 基线 %.4f)\n", h_ref, cpu_sum);
	std::printf("    GPU 归约结果  = %.4f\n", gpu_sum);
	std::printf("    相对误差      = %.3e (容差 1e-3)  ->  %s\n", rel_err, pass ? "PASS" : "FAIL");
	if (strict_benchmark) {
		std::printf("    基准(严格)    = %d 次预热, %d 组 x %d 次\n", warmup_iterations,
		            sample_count, iterations);
		std::printf("    耗时(中位数)  = %.4f ms (P5 %.4f, P95 %.4f)\n", median_ms, p5_ms, p95_ms);
	} else {
		std::printf("    耗时          = %.4f ms/kernel (%d 次)\n", median_ms, iterations);
	}
	std::printf("    有效带宽      = %.2f GB/s\n", bandwidth_gbps);

	CUDA_CHECK(cudaFree(d_input));
	CUDA_CHECK(cudaFree(d_output));
	return pass;
}

// ============================================================================
// test.cu —— GEMM 测试驱动实现：确定性输入、CPU double 参考、CUDA event 采样、
// 逐元素校验与英文报告（接口见 include/test.cuh，入口与场景在 main.cu）。
// ============================================================================

#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "include/test.cuh"
#include "operator_common/CpuTimer.h"
#include "operator_common/GpuTimer.h"
#include "operator_common/cuda_check.h"

namespace {

constexpr double kRelTol = 1e-3;

struct TimingStats {
	float median_ms = 0.0f;
};

struct CompareStats {
	double max_rel_err = 0.0;
	std::int64_t worst_idx = -1;
	std::int64_t bad_count = 0;
	bool pass = true;
};

// 用行列索引与 salt 生成 1/1000…1 的可复现正数输入。
void FillDeterministicInput(std::vector<float>& values, int rows, int cols, int salt) {
	for (int row = 0; row < rows; ++row) {
		for (int col = 0; col < cols; ++col) {
			const int raw = (row * 37 + col * 17 + salt) % 1000;
			values[static_cast<std::size_t>(row) * cols + col] =
			    static_cast<float>(raw + 1) / 1000.0f;
		}
	}
}

// 采集每次计算调用耗时：连续 iterations 次平均后取样本中位数。
template <typename LaunchFn>
TimingStats MeasureKernel(const LaunchFn& launch, int warmup_iterations, int iterations,
                          int sample_count) {
	for (int i = 0; i < warmup_iterations; ++i) launch();
	CUDA_CHECK(cudaDeviceSynchronize());

	std::vector<float> samples(static_cast<std::size_t>(sample_count));
	for (int sample = 0; sample < sample_count; ++sample) {
		GpuTimer timer;
		timer.Start();
		for (int i = 0; i < iterations; ++i) launch();
		samples[static_cast<std::size_t>(sample)] = timer.StopMs() / static_cast<float>(iterations);
	}
	std::sort(samples.begin(), samples.end());
	return {samples[static_cast<std::size_t>(sample_count / 2)]};
}

// cuBLAS 调用失败时打印数值状态码并立即终止。
void CheckCublas(cublasStatus_t status, const char* operation) {
	if (status == CUBLAS_STATUS_SUCCESS) return;
	std::fprintf(stderr, "cuBLAS error %d in %s\n", static_cast<int>(status), operation);
	std::exit(EXIT_FAILURE);
}

// 对输出逐元素比较。NaN/Inf 或相对误差超过 1e-3 均失败，记录最大误差以便定位。
CompareStats CompareCellwise(const std::vector<float>& reference, const std::vector<float>& gpu) {
	CompareStats stats;
	for (std::size_t i = 0; i < reference.size(); ++i) {
		const double expected = reference[i];
		const double actual = gpu[i];
		if (std::isnan(actual) || std::isinf(actual)) {
			++stats.bad_count;
			continue;
		}
		const double rel_err = std::fabs(actual - expected) / std::max(std::fabs(expected), 1e-30);
		if (rel_err > stats.max_rel_err) {
			stats.max_rel_err = rel_err;
			stats.worst_idx = static_cast<std::int64_t>(i);
		}
		if (rel_err > kRelTol) ++stats.bad_count;
	}
	stats.pass = stats.bad_count == 0;
	return stats;
}

struct SweepTiming {
	double median_ms = -1.0;
	int iters = 0;
};

// 扫描模式用的设备缓冲：只负责分配与释放。
struct SweepBuffers {
	float* a = nullptr;
	float* b = nullptr;
	float* c = nullptr;

	~SweepBuffers() {
		if (a != nullptr) cudaFree(a);
		if (b != nullptr) cudaFree(b);
		if (c != nullptr) cudaFree(c);
	}
};

// 扫描模式的计时：预热 → 1 次调用估计单次耗时 → 每组迭代数 = clamp(预算内的调用数 /
// kSampleCount, 1, kMaxItersPerSample) → 采 kSampleCount 组取单次耗时中位数。
// launch 返回 false 表示启动失败，立即放弃该采样点；异步执行错误由 GpuTimer::StopMs 的
// 隐式同步检出（CUDA_CHECK 处理）。
template <typename LaunchFn>
SweepTiming MeasureKernelAdaptive(const LaunchFn& launch, int warmup_iterations, double budget_ms) {
	constexpr int kSampleCount = 3;
	constexpr int kMaxItersPerSample = 100;
	for (int i = 0; i < warmup_iterations; ++i) {
		if (!launch()) return {};
	}
	CUDA_CHECK(cudaDeviceSynchronize());

	GpuTimer probe;
	probe.Start();
	if (!launch()) return {};
	const double single_ms = std::max(static_cast<double>(probe.StopMs()), 1e-6);

	int iters = static_cast<int>(std::lround(budget_ms / (single_ms * kSampleCount)));
	iters = std::min(std::max(iters, 1), kMaxItersPerSample);

	std::vector<float> samples(static_cast<std::size_t>(kSampleCount));
	for (int sample = 0; sample < kSampleCount; ++sample) {
		GpuTimer timer;
		timer.Start();
		for (int i = 0; i < iters; ++i) {
			if (!launch()) return {};
		}
		samples[static_cast<std::size_t>(sample)] = timer.StopMs() / static_cast<float>(iters);
	}
	std::sort(samples.begin(), samples.end());
	return {samples[static_cast<std::size_t>(kSampleCount / 2)], iters};
}

}  // namespace

bool test_gemm_kernel(GemmKernel kernel, const char* kernel_name, int M, int N, int K, int grid_x,
                      int grid_y, int block_x, int block_y, std::size_t smem_bytes,
                      bool strict_benchmark) {
	if (kernel == nullptr || M < 0 || N < 0 || K < 0 || grid_x < 1 || grid_y < 1 || block_x < 1 ||
	    block_y < 1) {
		std::printf("Test Case: %s\n  Result: FAIL (invalid kernel, shape, or launch config)\n",
		            kernel_name == nullptr ? "N/A" : kernel_name);
		return false;
	}

	const int warmup_iterations = strict_benchmark ? 100 : 1;
	const int iterations = strict_benchmark ? 1000 : 100;
	const int sample_count = strict_benchmark ? 21 : 1;
	const std::size_t a_count = static_cast<std::size_t>(M) * K;
	const std::size_t b_count = static_cast<std::size_t>(K) * N;
	const std::size_t c_count = static_cast<std::size_t>(M) * N;
	// 空输出：直接判通过，跳过计时与比较。
	if (c_count == 0) {
		std::printf("Test Case: %s\n  Result: PASS (empty output; measurement skipped)\n",
		            kernel_name);
		return true;
	}

	std::vector<float> h_a(a_count);
	std::vector<float> h_b(b_count);
	std::vector<float> h_ref(c_count);
	FillDeterministicInput(h_a, M, K, 13);
	FillDeterministicInput(h_b, K, N, 97);
	CpuTimer cpu_timer;
	cpu_timer.Start();
	sgemm_cpu(h_a.data(), h_b.data(), h_ref.data(), M, N, K);
	const float cpu_ms = cpu_timer.StopMs();

	float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
	CUDA_CHECK(cudaMalloc(&d_a, std::max(a_count, std::size_t{1}) * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_b, std::max(b_count, std::size_t{1}) * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_c, c_count * sizeof(float)));
	if (a_count > 0)
		CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), a_count * sizeof(float), cudaMemcpyHostToDevice));
	if (b_count > 0)
		CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), b_count * sizeof(float), cudaMemcpyHostToDevice));

	const float* d_a_arg = d_a;
	const float* d_b_arg = d_b;
	float* d_c_arg = d_c;
	int m_arg = M;
	int n_arg = N;
	int k_arg = K;
	// 内核参数按 (A, B, C, M, N, K) 打包，通过 cudaLaunchKernel 按给定配置启动。
	void* args[] = {&d_a_arg, &d_b_arg, &d_c_arg, &m_arg, &n_arg, &k_arg};
	const auto launch = [&]() {
		CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(grid_x, grid_y),
		                            dim3(block_x, block_y), args, smem_bytes));
	};

	const TimingStats gpu = MeasureKernel(launch, warmup_iterations, iterations, sample_count);
	std::vector<float> h_gpu(c_count);
	CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_c, c_count * sizeof(float), cudaMemcpyDeviceToHost));
	const CompareStats compare = CompareCellwise(h_ref, h_gpu);
	const double tflops = 2.0 * static_cast<double>(M) * N * K / (gpu.median_ms * 1e9);

	std::printf("Test Case: %s\n", kernel_name);
	std::printf("  Shape: M=%d, N=%d, K=%d (row-major fp32)\n", M, N, K);
	std::printf("  Launch: grid=(%d, %d), block=(%d, %d), dynamic smem=%zu B\n", grid_x, grid_y,
	            block_x, block_y, smem_bytes);
	std::printf(
	    "  Correctness: max relative error %.3e (tolerance 1e-3), bad elements %lld -> %s\n",
	    compare.max_rel_err, static_cast<long long>(compare.bad_count),
	    compare.pass ? "PASS" : "FAIL");
	std::printf("  CPU reference: %.3f ms\n", cpu_ms);
	std::printf("  GPU time: median %.4f ms\n", gpu.median_ms);
	std::printf("  Throughput: %.3f TFLOPS (%d warmup, %d sample x %d iterations)\n", tflops,
	            warmup_iterations, sample_count, iterations);
	if (!compare.pass && compare.worst_idx >= 0) {
		const std::size_t idx = static_cast<std::size_t>(compare.worst_idx);
		std::printf("  Worst output: C[%zu] GPU %.6e / CPU %.6e\n", idx, h_gpu[idx], h_ref[idx]);
	}

	CUDA_CHECK(cudaFree(d_a));
	CUDA_CHECK(cudaFree(d_b));
	CUDA_CHECK(cudaFree(d_c));
	return compare.pass;
}

bool test_cublas_sgemm(const char* test_name, int M, int N, int K, bool strict_benchmark) {
	if (M < 0 || N < 0 || K < 0) {
		std::printf("Test Case: %s\n  Result: FAIL (invalid shape)\n",
		            test_name == nullptr ? "cublasSgemm" : test_name);
		return false;
	}

	const int warmup_iterations = strict_benchmark ? 100 : 1;
	const int iterations = strict_benchmark ? 1000 : 100;
	const int sample_count = strict_benchmark ? 21 : 1;
	const std::size_t a_count = static_cast<std::size_t>(M) * K;
	const std::size_t b_count = static_cast<std::size_t>(K) * N;
	const std::size_t c_count = static_cast<std::size_t>(M) * N;
	if (c_count == 0) {
		std::printf("Test Case: %s\n  Result: PASS (empty output; measurement skipped)\n",
		            test_name == nullptr ? "cublasSgemm" : test_name);
		return true;
	}

	std::vector<float> h_a(a_count);
	std::vector<float> h_b(b_count);
	std::vector<float> h_ref(c_count);
	FillDeterministicInput(h_a, M, K, 13);
	FillDeterministicInput(h_b, K, N, 97);
	CpuTimer cpu_timer;
	cpu_timer.Start();
	sgemm_cpu(h_a.data(), h_b.data(), h_ref.data(), M, N, K);
	const float cpu_ms = cpu_timer.StopMs();

	float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
	CUDA_CHECK(cudaMalloc(&d_a, std::max(a_count, std::size_t{1}) * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_b, std::max(b_count, std::size_t{1}) * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_c, c_count * sizeof(float)));
	if (a_count > 0)
		CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), a_count * sizeof(float), cudaMemcpyHostToDevice));
	if (b_count > 0)
		CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), b_count * sizeof(float), cudaMemcpyHostToDevice));

	cublasHandle_t handle = nullptr;
	CheckCublas(cublasCreate(&handle), "cublasCreate");
	CheckCublas(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode");
	CheckCublas(cublasSetStream(handle, nullptr), "cublasSetStream");
	const float alpha = 1.0f;
	const float beta = 0.0f;
	const auto launch = [&]() {
		// 行主序 C=A×B 的内存等价于列主序 C^T=B^T×A^T；交换 A/B 与 M/N 即可避免转置。
		CheckCublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b, N, d_a, K,
		                        &beta, d_c, N),
		            "cublasSgemm");
	};

	const TimingStats gpu = MeasureKernel(launch, warmup_iterations, iterations, sample_count);
	std::vector<float> h_gpu(c_count);
	CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_c, c_count * sizeof(float), cudaMemcpyDeviceToHost));
	const CompareStats compare = CompareCellwise(h_ref, h_gpu);
	const double tflops = 2.0 * static_cast<double>(M) * N * K / (gpu.median_ms * 1e9);

	std::printf("Test Case: %s\n", test_name == nullptr ? "cublasSgemm" : test_name);
	std::printf("  Shape: M=%d, N=%d, K=%d (row-major fp32)\n", M, N, K);
	std::printf("  Library: cuBLAS, math mode CUBLAS_PEDANTIC_MATH\n");
	std::printf(
	    "  Correctness: max relative error %.3e (tolerance 1e-3), bad elements %lld -> %s\n",
	    compare.max_rel_err, static_cast<long long>(compare.bad_count),
	    compare.pass ? "PASS" : "FAIL");
	std::printf("  CPU reference: %.3f ms\n", cpu_ms);
	std::printf("  GPU time: median %.4f ms\n", gpu.median_ms);
	std::printf("  Throughput: %.3f TFLOPS (%d warmup, %d sample x %d iterations)\n", tflops,
	            warmup_iterations, sample_count, iterations);
	if (!compare.pass && compare.worst_idx >= 0) {
		const std::size_t idx = static_cast<std::size_t>(compare.worst_idx);
		std::printf("  Worst output: C[%zu] GPU %.6e / CPU %.6e\n", idx, h_gpu[idx], h_ref[idx]);
	}

	CheckCublas(cublasDestroy(handle), "cublasDestroy");
	CUDA_CHECK(cudaFree(d_a));
	CUDA_CHECK(cudaFree(d_b));
	CUDA_CHECK(cudaFree(d_c));
	return compare.pass;
}

double bench_gemm_kernel(const void* kernel, int M, int N, int K, int grid_x, int grid_y,
                         int block_x, int block_y, std::size_t smem_bytes, int warmup_iterations,
                         double budget_ms, int* iters_out) {
	// 形状与启动配置契约同 test_gemm_kernel；M / N / K 必须为正。
	if (kernel == nullptr || M < 1 || N < 1 || K < 1 || grid_x < 1 || grid_y < 1 || block_x < 1 ||
	    block_y < 1 || budget_ms <= 0.0) {
		return -1.0;
	}

	SweepBuffers buffers;
	// 先按 size_t 提升再相乘，避免大尺寸下 int 溢出。
	const std::size_t a_bytes = static_cast<std::size_t>(M) * K * sizeof(float);
	const std::size_t b_bytes = static_cast<std::size_t>(K) * N * sizeof(float);
	CUDA_CHECK(cudaMalloc(&buffers.a, a_bytes));
	CUDA_CHECK(cudaMalloc(&buffers.b, b_bytes));
	CUDA_CHECK(cudaMalloc(&buffers.c, static_cast<std::size_t>(M) * N * sizeof(float)));
	CUDA_CHECK(cudaMemset(buffers.a, 0, a_bytes));
	CUDA_CHECK(cudaMemset(buffers.b, 0, b_bytes));

	const float* d_a_arg = buffers.a;
	const float* d_b_arg = buffers.b;
	float* d_c_arg = buffers.c;
	int m_arg = M;
	int n_arg = N;
	int k_arg = K;
	void* args[] = {&d_a_arg, &d_b_arg, &d_c_arg, &m_arg, &n_arg, &k_arg};
	// 启动失败只返回 false，交由调用方跳过该点；此处不用 CUDA_CHECK 终止整轮扫描。
	const auto launch = [&]() -> bool {
		return cudaLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(grid_x, grid_y),
		                        dim3(block_x, block_y), args, smem_bytes) == cudaSuccess;
	};

	const SweepTiming timing = MeasureKernelAdaptive(launch, warmup_iterations, budget_ms);
	if (iters_out != nullptr) *iters_out = timing.iters;
	return timing.median_ms;
}

double bench_cublas_sgemm(int M, int N, int K, int warmup_iterations, double budget_ms,
                          int* iters_out) {
	if (M < 1 || N < 1 || K < 1 || budget_ms <= 0.0) {
		return -1.0;
	}

	SweepBuffers buffers;
	const std::size_t a_bytes = static_cast<std::size_t>(M) * K * sizeof(float);
	const std::size_t b_bytes = static_cast<std::size_t>(K) * N * sizeof(float);
	CUDA_CHECK(cudaMalloc(&buffers.a, a_bytes));
	CUDA_CHECK(cudaMalloc(&buffers.b, b_bytes));
	CUDA_CHECK(cudaMalloc(&buffers.c, static_cast<std::size_t>(M) * N * sizeof(float)));
	CUDA_CHECK(cudaMemset(buffers.a, 0, a_bytes));
	CUDA_CHECK(cudaMemset(buffers.b, 0, b_bytes));

	cublasHandle_t handle = nullptr;
	CheckCublas(cublasCreate(&handle), "cublasCreate");
	CheckCublas(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode");
	CheckCublas(cublasSetStream(handle, nullptr), "cublasSetStream");
	const float alpha = 1.0f;
	const float beta = 0.0f;
	// 行主序 C=A×B 等价于列主序 C^T=B^T×A^T，与 test_cublas_sgemm 的映射保持一致。
	const auto launch = [&]() -> bool {
		return cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, buffers.b, N,
		                   buffers.a, K, &beta, buffers.c, N) == CUBLAS_STATUS_SUCCESS;
	};

	const SweepTiming timing = MeasureKernelAdaptive(launch, warmup_iterations, budget_ms);
	CheckCublas(cublasDestroy(handle), "cublasDestroy");
	if (iters_out != nullptr) *iters_out = timing.iters;
	return timing.median_ms;
}

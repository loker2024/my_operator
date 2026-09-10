// demo_utils.cu —— 演示 operator_common 工具头用法
// 1) CPU 朴素基线 + 与 GPU kernel 的耗时对比；
// 2) 主机端数组使用 pinned memory（cudaMallocHost / cudaFreeHost）。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "operator_common/CpuTimer.h"    // CpuTimer（CPU 计时）
#include "operator_common/GpuTimer.h"    // GpuTimer（CUDA event 计时）
#include "operator_common/cuda_check.h"  // CUDA_CHECK / PrintDeviceInfo

// 向量加法 kernel
__global__ void VecAddKernel(const float* a, const float* b, float* c, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		c[i] = a[i] + b[i];
	}
}

// CPU 基线：朴素单线程参考实现（同时兼作正确性基准，见 docs/benchmark-methodology.md）
void VecAddCpu(const float* a, const float* b, float* c, int n) {
	for (int i = 0; i < n; ++i) {
		c[i] = a[i] + b[i];
	}
}

// 多轮计时取中位数（口径与基准文档一致，避免离群值带偏）
template <typename T>
static T Median(std::vector<T> v) {
	std::sort(v.begin(), v.end());
	return v[v.size() / 2];
}

int main() {
	// 1. 打印设备信息，确认架构
	PrintDeviceInfo();

	const int n = 1 << 24;  // 元素
	const size_t bytes = n * sizeof(float);

	// 2. 主机内存：使用 pinned memory（页锁定），
	//    可被 GPU 直接访问/拷贝更快，释放时必须用 cudaFreeHost。
	float *h_a = nullptr, *h_b = nullptr;
	float *h_c = nullptr, *h_ref = nullptr;  // h_c: GPU 结果; h_ref: CPU 参考结果
	CUDA_CHECK(cudaMallocHost(&h_a, bytes));
	CUDA_CHECK(cudaMallocHost(&h_b, bytes));
	CUDA_CHECK(cudaMallocHost(&h_c, bytes));
	CUDA_CHECK(cudaMallocHost(&h_ref, bytes));

	for (int i = 0; i < n; ++i) {
		h_a[i] = static_cast<float>(i);
		h_b[i] = static_cast<float>(2.0f * i);
	}

	// 3. 分配设备内存（每步都用 CUDA_CHECK 兜底，出错即定位到具体文件行）
	float *d_a, *d_b, *d_c;
	CUDA_CHECK(cudaMalloc(&d_a, bytes));
	CUDA_CHECK(cudaMalloc(&d_b, bytes));
	CUDA_CHECK(cudaMalloc(&d_c, bytes));

	CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

	// 4. 预热：先各跑一次，消除冷启动/驱动加载影响；
	//    同时 CPU 参考结果写入 h_ref，供第 7 步做正确性校验。
	const int threads = 256;
	const int blocks = (n + threads - 1) / threads;
	VecAddCpu(h_a, h_b, h_ref, n);
	VecAddKernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
	CUDA_CHECK(cudaDeviceSynchronize());

	// 5. CPU 基线计时：单线程多次取中位数。
	//    CPU/GPU 统一使用 Start()/StopMs() 的 Timer 接口，便于对比。
	std::vector<double> cpu_times;
	CpuTimer cpu_timer;
	const int cpu_iters = 5;
	for (int it = 0; it < cpu_iters; ++it) {
		cpu_timer.Start();
		VecAddCpu(h_a, h_b, h_ref, n);  // 结果确定，重复写入 h_ref 即可
		cpu_times.push_back(cpu_timer.StopMs());
	}
	double cpu_ms = Median(cpu_times);

	// 6. GPU kernel 计时：kernel 耗时短，多跑几轮取中位数
	std::vector<float> gpu_times;
	GpuTimer timer;
	const int gpu_iters = 100;
	for (int it = 0; it < gpu_iters; ++it) {
		timer.Start();
		VecAddKernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
		gpu_times.push_back(timer.StopMs());  // 内部已同步，耗时即本轮 kernel 时间
	}
	float gpu_ms = Median(gpu_times);

	// 7. 回拷并做正确性校验（GPU 结果 vs CPU 参考，相对误差口径见文档）
	CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

	double max_err = 0.0;
	long long nerr = 0;
	for (int i = 0; i < n; ++i) {
		double err = std::abs(static_cast<double>(h_c[i]) - h_ref[i]) /
		             std::max(std::abs(static_cast<double>(h_ref[i])), 1e-30);
		if (!(err <= 1e-6)) ++nerr;  // NaN/Inf/超差均记为失败
		max_err = std::max(max_err, err);
	}
	printf("max rel err = %g, bad = %lld  ->  %s\n", max_err, nerr, nerr == 0 ? "PASS" : "FAIL");

	// 8. 打印耗时对比
	double speedup = cpu_ms / gpu_ms;
	double gbps = 3.0 * bytes / (gpu_ms * 1e-3) / 1e9;  // 读A + 读B + 写C
	printf("CPU 基线耗时 : %.3f ms (中位数, %d 轮)\n", cpu_ms, cpu_iters);
	printf("GPU kernel耗时: %.3f ms (中位数, %d 轮)\n", gpu_ms, gpu_iters);
	printf("加速比       : %.1fx\n", speedup);
	printf("GPU 有效带宽  : %.1f GB/s (读A+读B+写C)\n", gbps);

	// 9. 释放资源（pinned memory 用 cudaFreeHost，普通设备内存用 cudaFree）
	CUDA_CHECK(cudaFreeHost(h_a));
	CUDA_CHECK(cudaFreeHost(h_b));
	CUDA_CHECK(cudaFreeHost(h_c));
	CUDA_CHECK(cudaFreeHost(h_ref));
	CUDA_CHECK(cudaFree(d_a));
	CUDA_CHECK(cudaFree(d_b));
	CUDA_CHECK(cudaFree(d_c));
	return nerr == 0 ? 0 : 1;

	// nvcc -std=c++17 -O3 -I ../common/include -o demo_utils demo_utils.cu
}

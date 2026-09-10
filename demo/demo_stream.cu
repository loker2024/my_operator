// demo_stream.cu —— 在 demo_utils.cu 基础上演示两个进阶点：
//   1) VecAdd kernel 改为 grid-stride loop：线程通过固定 stride 遍历多个元素，
//      启动任意规模的 grid 都正确，因此可以不必按 n 铺满 block（便于填满设备）；
//   2) 主机端为 pinned memory，利用多条 CUDA stream 把“H2D 拷贝 / kernel /
//      D2H 回拷”切成小块做异步流水，让计算与传输尽量重叠。
// 正确性口径、CPU 基线、中位数统计均沿用 demo_utils.cu。
//
// 编译运行（同一套工具头，目录与 demo_utils.cu 相同）：
//   nvcc -std=c++17 -O3 -I ../common/include -o demo_stream demo_stream.cu
//   ./demo_stream
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "operator_common/CpuTimer.h"    // CpuTimer（CPU 计时）
#include "operator_common/GpuTimer.h"    // GpuTimer（CUDA event 计时）
#include "operator_common/cuda_check.h"  // CUDA_CHECK / PrintDeviceInfo

// 向量加法 kernel —— grid-stride loop 版本
// 每个线程以 blockDim.x*gridDim.x 为步长向后跳，保证任意 (blocks, threads)
// 组合都能处理完 n 个元素（grid 无需按 n 精确覆盖）。
__global__ void VecAddKernel(const float* a, const float* b, float* c, int n) {
	const int stride = gridDim.x * blockDim.x;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
		c[i] = a[i] + b[i];
	}
}

// CPU 基线：朴素单线程参考实现（兼作正确性基准）
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

// 在指定 stream 上提交“第 off 个元素起的 chunk 个元素”的完整流水：
//   流内顺序保证：H2D 拷贝 -> kernel -> D2H 回拷，天然有序，无需事件同步；
//   流间并发：stream A 的 kernel 可与 stream B 的拷贝重叠执行。
// off / chunk 都以元素为单位；h_* 为 pinned memory，cudaMemcpyAsync 才可能异步。
static void SubmitChunk(int off, int chunk, const float* h_a, const float* h_b, float* h_c,
                        float* d_a, float* d_b, float* d_c, int blocks, int threads,
                        cudaStream_t stream) {
	const size_t chunk_bytes = chunk * sizeof(float);
	CUDA_CHECK(cudaMemcpyAsync(d_a + off, h_a + off, chunk_bytes, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_b + off, h_b + off, chunk_bytes, cudaMemcpyHostToDevice, stream));
	VecAddKernel<<<blocks, threads, 0, stream>>>(d_a + off, d_b + off, d_c + off, chunk);
	CUDA_CHECK(cudaMemcpyAsync(h_c + off, d_c + off, chunk_bytes, cudaMemcpyDeviceToHost, stream));
}

// 用 n_chunks 条 stream 跑一遍完整流水，返回单轮 GPU 耗时(ms)。
// 计时：每条流在开工前记 start、收尾后记 stop，各流并发因此近似同时开始，
// 取所有流耗时最大值作为本轮总时间（已包含同步等待，返回时 h_c 结果就绪）。
static float RunPipelineOnce(int n_chunks, int n, const float* h_a, const float* h_b, float* h_c,
                             float* d_a, float* d_b, float* d_c, int blocks, int threads,
                             cudaStream_t* streams, cudaEvent_t* starts, cudaEvent_t* stops) {
	const int chunk = n / n_chunks;  // 要求 n 能被 n_chunks 整除
	for (int s = 0; s < n_chunks; ++s) {
		const int off = s * chunk;
		CUDA_CHECK(cudaEventRecord(starts[s], streams[s]));
		SubmitChunk(off, chunk, h_a, h_b, h_c, d_a, d_b, d_c, blocks, threads, streams[s]);
		CUDA_CHECK(cudaEventRecord(stops[s], streams[s]));
	}
	float total_ms = 0.0f;
	for (int s = 0; s < n_chunks; ++s) {
		CUDA_CHECK(cudaEventSynchronize(stops[s]));  // 等待该流全部完成
		float ms = 0.0f;
		CUDA_CHECK(cudaEventElapsedTime(&ms, starts[s], stops[s]));
		total_ms = std::max(total_ms, ms);
	}
	return total_ms;
}

// 用 n_chunks 条 stream 重复跑 iters 轮，取中位数作为该配置的耗时。
// 每轮之间流内异步操作已由事件同步收尾，h_c 在返回后为最新一轮结果。
static float RunPipelineMedian(int n_chunks, int iters, int n, const float* h_a, const float* h_b,
                               float* h_c, float* d_a, float* d_b, float* d_c, int blocks,
                               int threads) {
	std::vector<cudaStream_t> streams(n_chunks);
	std::vector<cudaEvent_t> starts(n_chunks), stops(n_chunks);
	for (int s = 0; s < n_chunks; ++s) {
		CUDA_CHECK(cudaStreamCreate(&streams[s]));
		CUDA_CHECK(cudaEventCreate(&starts[s]));
		CUDA_CHECK(cudaEventCreate(&stops[s]));
	}
	// 预热一轮，消除冷启动影响
	RunPipelineOnce(n_chunks, n, h_a, h_b, h_c, d_a, d_b, d_c, blocks, threads, streams.data(),
	                starts.data(), stops.data());

	std::vector<float> times;
	times.reserve(iters);
	for (int it = 0; it < iters; ++it) {
		times.push_back(RunPipelineOnce(n_chunks, n, h_a, h_b, h_c, d_a, d_b, d_c, blocks, threads,
		                                streams.data(), starts.data(), stops.data()));
	}

	for (int s = 0; s < n_chunks; ++s) {
		CUDA_CHECK(cudaEventDestroy(stops[s]));
		CUDA_CHECK(cudaEventDestroy(starts[s]));
		CUDA_CHECK(cudaStreamDestroy(streams[s]));
	}
	return Median(times);
}

int main() {
	// 1. 打印设备信息，读取 SM 数（用于确定 grid-stride 的 grid 规模）
	PrintDeviceInfo();
	int dev = 0;
	CUDA_CHECK(cudaGetDevice(&dev));
	cudaDeviceProp prop;
	CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

	const int n = 1 << 24;  // 元素（能被下面的 n_chunks 整除）
	const size_t bytes = n * sizeof(float);

	// 2. 主机内存使用 pinned memory —— cudaMemcpyAsync 的前提之一，
	//    异步拷贝要求主机指针页锁定，释放必须用 cudaFreeHost。
	float *h_a = nullptr, *h_b = nullptr;
	float *h_c = nullptr, *h_ref = nullptr;  // h_c: GPU 结果; h_ref: CPU 参考
	CUDA_CHECK(cudaMallocHost(&h_a, bytes));
	CUDA_CHECK(cudaMallocHost(&h_b, bytes));
	CUDA_CHECK(cudaMallocHost(&h_c, bytes));
	CUDA_CHECK(cudaMallocHost(&h_ref, bytes));

	for (int i = 0; i < n; ++i) {
		h_a[i] = static_cast<float>(i);
		h_b[i] = static_cast<float>(2.0f * i);
	}

	// 3. 分配设备内存（整块一次分配，各 stream 通过偏移访问自己的片段）
	float *d_a, *d_b, *d_c;
	CUDA_CHECK(cudaMalloc(&d_a, bytes));
	CUDA_CHECK(cudaMalloc(&d_b, bytes));
	CUDA_CHECK(cudaMalloc(&d_c, bytes));

	// 4. kernel 启动参数：grid-stride 不必按 n 铺满 grid，
	//    取约 8 wave（按 SM 数缩放）的小 grid，让每线程迭代多次更能体现 stride；
	//    无论取多大都正确，改 blocks 只影响占用与循环次数。
	const int threads = 256;
	const int blocks = prop.multiProcessorCount * 8;

	// CPU 参考结果 + 基线耗时（单线程多次取中位数）
	VecAddCpu(h_a, h_b, h_ref, n);
	std::vector<double> cpu_times;
	CpuTimer cpu_timer;
	const int cpu_iters = 5;
	for (int it = 0; it < cpu_iters; ++it) {
		cpu_timer.Start();
		VecAddCpu(h_a, h_b, h_ref, n);
		cpu_times.push_back(cpu_timer.StopMs());
	}
	const double cpu_ms = Median(cpu_times);

	// 5. kernel-only 参考时间：仅计算、不含任何主机拷贝（默认流，串行）
	std::vector<float> kernel_times;
	GpuTimer timer;
	const int gpu_iters = 100;
	for (int it = 0; it < gpu_iters; ++it) {
		timer.Start();
		VecAddKernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
		kernel_times.push_back(timer.StopMs());
	}
	const float kernel_ms = Median(kernel_times);

	// 6. 流水线（异步传输 + kernel）在不同 stream 并发度下的耗时。
	//    1 条 stream 即“无重叠”的串行对照；2/4/8 条流让各 chunk 的
	//    计算与其它 chunk 的拷贝重叠。
	printf("grid-stride grid : %d blocks x %d threads (SM=%d x 8 waves)\n\n", blocks, threads,
	       prop.multiProcessorCount);
	printf("流水线耗时(含 H2D 拷贝 + kernel + D2H 回拷, 中位数 %d 轮):\n", gpu_iters);
	const std::vector<int> stream_configs = {1, 2, 4, 8};
	std::vector<float> pipeline_ms;
	float single_stream_ms = 0.0f;
	for (int n_chunks : stream_configs) {
		const float ms = RunPipelineMedian(n_chunks, gpu_iters, n, h_a, h_b, h_c, d_a, d_b, d_c,
		                                   blocks, threads);
		pipeline_ms.push_back(ms);
		if (n_chunks == 1) single_stream_ms = ms;
		printf("  %d stream(s): %.3f ms   (每块 %d 元素)\n", n_chunks, ms, n / n_chunks);
	}

	// 7. 正确性校验：以最后一组配置的结果为准（每轮结束均已同步回拷完成），
	//    与 CPU 参考逐元素比对（相对误差口径同 demo_utils.cu）。
	double max_err = 0.0;
	long long nerr = 0;
	for (int i = 0; i < n; ++i) {
		double err = std::abs(static_cast<double>(h_c[i]) - h_ref[i]) /
		             std::max(std::abs(static_cast<double>(h_ref[i])), 1e-30);
		if (!(err <= 1e-6)) ++nerr;
		max_err = std::max(max_err, err);
	}
	printf("\nmax rel err = %g, bad = %lld  ->  %s\n", max_err, nerr, nerr == 0 ? "PASS" : "FAIL");

	// 8. 汇总对比（%s 位置用单流流水线做参照）
	const float best_pipeline = *std::min_element(pipeline_ms.begin(), pipeline_ms.end());
	const int best_streams =
	    stream_configs[std::min_element(pipeline_ms.begin(), pipeline_ms.end()) -
	                   pipeline_ms.begin()];
	const double cpu_speedup = cpu_ms / best_pipeline;
	const double gbps = 3.0 * bytes / (kernel_ms * 1e-3) / 1e9;  // 仅 kernel 内读A+读B+写C
	printf("CPU 基线耗时     : %.3f ms (中位数, %d 轮)\n", cpu_ms, cpu_iters);
	printf("kernel-only耗时  : %.3f ms (中位数, %d 轮, 不含主机拷贝)\n", kernel_ms, gpu_iters);
	printf("单流流水线耗时   : %.3f ms\n", single_stream_ms);
	printf("最优流水线耗时   : %.3f ms (%d streams)\n", best_pipeline, best_streams);
	printf("重叠相对收益     : 单流/最优 = %.2fx\n", single_stream_ms / best_pipeline);
	printf("CPU 加速比       : %.1fx (以最优流水线为分母)\n", cpu_speedup);
	printf("GPU 有效带宽     : %.1f GB/s (kernel 读A+读B+写C)\n", gbps);

	// 9. 释放资源（pinned 用 cudaFreeHost，设备内存用 cudaFree）
	CUDA_CHECK(cudaFreeHost(h_a));
	CUDA_CHECK(cudaFreeHost(h_b));
	CUDA_CHECK(cudaFreeHost(h_c));
	CUDA_CHECK(cudaFreeHost(h_ref));
	CUDA_CHECK(cudaFree(d_a));
	CUDA_CHECK(cudaFree(d_b));
	CUDA_CHECK(cudaFree(d_c));
	return nerr == 0 ? 0 : 1;
	// nvcc -std=c++17 -O3 -I ../common/include -o demo_stream demo_stream.cu
}

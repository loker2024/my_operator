// test.cu —— test_softmax_kernel 的实现（接口见 test.cuh），执行入口见 main.cu。
//
// 结构：主函数只串联流程，各步骤由下方匿名命名空间的小工具承担 ——
//   ① 生成确定性输入与主机参考（softmax_cpu，CpuTimer 计时）
//   ② 申请设备缓冲并拷入
//   ③ 预热 + 计时采样（GpuTimer，取中位数与 P5/P95）
//   ④ 拷回输出并逐元素比对（相对误差 + NaN/Inf）
//   ⑤ 渲染报告（英文文案：正确性、CPU/GPU 耗时、加速比、有效带宽、峰值带宽与利用率）
// 判据与口径（容差 1e-5、跳过 |ref| 过小元素、两档采样）见 test.cuh 与
// docs/benchmark-methodology.md。

#include <algorithm>  // std::sort
#include <cmath>      // std::fabs / std::isnan / std::isinf
#include <cstddef>    // std::size_t
#include <cstdint>    // std::int64_t
#include <cstdio>     // printf
#include <vector>     // std::vector

#include "include/test.cuh"
#include "operator_common/CpuTimer.h"    // 主机参考计时（std::chrono）
#include "operator_common/GpuTimer.h"    // 内核计时（CUDA event）
#include "operator_common/cuda_check.h"  // CUDA_CHECK 错误检查宏

namespace {

// 正确性判据（见 docs/benchmark-methodology.md §2）：逐元素相对误差 <= kRelTol，跳过
// |ref| < kSkipAbs 的元素（此时相对比较无意义）。
constexpr double kRelTol = 1e-5;
constexpr double kSkipAbs = 1e-30;

// 计时结果：median 为主指标，P5 / P95 为波动区间（见 docs/benchmark-methodology.md §3.2）。
struct TimingStats {
	float median_ms = 0.0f;
	float p5_ms = 0.0f;
	float p95_ms = 0.0f;
};

// 逐元素比对结果。
struct CompareStats {
	double max_rel_err = 0.0;
	std::int64_t worst_idx = -1;  // 最大相对误差所在下标（-1 = 无可比元素）
	std::int64_t bad_count = 0;   // 超差或 NaN/Inf 的元素数
	std::int64_t skipped = 0;     // |ref| 过小、跳过相对比较的元素数
	bool pass = true;
};

// 一次运行的全部结果，供 PrintReport 渲染（字段与输出项一一对应）。
struct RunReport {
	const char* kernel_name = nullptr;
	int rows = 0;
	int cols = 0;
	int grid = 0;
	int block = 0;
	std::size_t smem_bytes = 0;
	bool host_api = false;
	bool has_measurement = false;
	bool invalid_arguments = false;
	bool strict_benchmark = false;
	int warmup_iterations = 0;
	int iterations = 0;
	int sample_count = 0;
	double rel_tol = kRelTol;
	double cpu_ms = 0.0;          // 主机参考单次耗时
	double bandwidth_gbps = 0.0;  // 有效带宽（逻辑数据量：每元素读 1 次 + 写 1 次）
	TimingStats gpu{};
	CompareStats cmp{};
	const std::vector<float>* h_ref = nullptr;  // 失败定位用（仅 count > 0 时非空）
	const std::vector<float>* h_gpu = nullptr;
	const char* skip_reason = nullptr;  // 非空 = 无输出元素可比（空矩阵 / 空行）
};

// 设备静态属性快照：进程内查询一次、每份报告复用。动态功耗/温度不属于 CUDA Runtime
// 可查询范围，故不伪造成实时指标；峰值显存带宽为理论估算，口径与 cuda_check.h 一致。
struct DeviceInfo {
	char name[256] = "N/A";
	int major = 0;
	int minor = 0;
	int sm_count = 0;
	std::size_t global_memory_bytes = 0;
	std::size_t shared_memory_per_block = 0;
	int memory_clock_khz = 0;
	int memory_bus_width_bits = 0;
	double peak_dram_bandwidth_gbps = 0.0;
	bool available = false;
};

const DeviceInfo& GetDeviceInfo() {
	static const DeviceInfo cached = [] {
		DeviceInfo info;
		int dev = 0;
		if (cudaGetDevice(&dev) != cudaSuccess) return info;
		cudaDeviceProp prop{};
		if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return info;

		std::snprintf(info.name, sizeof(info.name), "%s", prop.name);
		info.major = prop.major;
		info.minor = prop.minor;
		info.sm_count = prop.multiProcessorCount;
		info.global_memory_bytes = prop.totalGlobalMem;
		info.shared_memory_per_block = prop.sharedMemPerBlock;
		info.memory_clock_khz = prop.memoryClockRate;
		info.memory_bus_width_bits = prop.memoryBusWidth;
		if (info.memory_clock_khz > 0 && info.memory_bus_width_bits > 0) {
			info.peak_dram_bandwidth_gbps =
			    2.0 * info.memory_clock_khz * (info.memory_bus_width_bits / 8.0) / 1e6;
		}
		info.available = true;
		return info;
	}();
	return cached;
}

// 生成可复现输入：元素取自 (row*31 + col*17) % 2001 - 1000 再除 100，落在 [-10, 10)。
// 确定性、与随机数实现无关；范围保证 max-shift 后 exp 参数 <= 0、行和 >= 1（含
// exp(0) 项），既不溢出 / 下溢，也不会让极小分母放大相对误差。
void FillDeterministicInput(std::vector<float>& h_input, int cols, std::int64_t count) {
	for (std::int64_t i = 0; i < count; ++i) {
		const int r = static_cast<int>(i / cols);  // cols == 0 时 count == 0，不进入
		const int c = static_cast<int>(i % cols);
		const int v = ((r * 31 + c * 17) % 2001) - 1000;
		h_input[static_cast<std::size_t>(i)] = static_cast<float>(v) / 100.0f;
	}
}

// 预热 + 计时采样：每组连续 iterations 次取平均，再取各组中位数（避免离群带偏）。
// 预热消除冷启动 / 驱动初始化 / 模块加载影响；内核轮间不加额外同步。
template <typename LaunchFn>
TimingStats MeasureKernel(const LaunchFn& launch, int warmup_iterations, int iterations,
                          int sample_count) {
	for (int i = 0; i < warmup_iterations; ++i) {
		launch();
	}
	CUDA_CHECK(cudaDeviceSynchronize());

	std::vector<float> samples_ms(static_cast<std::size_t>(sample_count));
	for (int s = 0; s < sample_count; ++s) {
		GpuTimer timer;  // 内部为 CUDA event，StopMs 前隐式同步
		timer.Start();
		for (int i = 0; i < iterations; ++i) {
			launch();
		}
		samples_ms[static_cast<std::size_t>(s)] = timer.StopMs() / static_cast<float>(iterations);
	}
	std::sort(samples_ms.begin(), samples_ms.end());

	TimingStats stats;
	stats.median_ms = samples_ms[static_cast<std::size_t>(sample_count / 2)];
	stats.p5_ms = samples_ms[static_cast<std::size_t>((sample_count - 1) * 5 / 100)];
	stats.p95_ms = samples_ms[static_cast<std::size_t>((sample_count - 1) * 95 / 100)];
	return stats;
}

// 逐元素比对：超差或 NaN/Inf 计入 bad_count，任一不合规即 pass = false；NaN/Inf 不参与
// 相对误差统计（无意义），|ref| 过小的元素跳过。
CompareStats CompareCellwise(const std::vector<float>& h_ref, const std::vector<float>& h_gpu,
                             std::int64_t count, double rel_tol, double skip_abs) {
	CompareStats stats;
	for (std::int64_t i = 0; i < count; ++i) {
		const double ref = h_ref[static_cast<std::size_t>(i)];
		const double got = h_gpu[static_cast<std::size_t>(i)];
		if (std::isnan(got) || std::isinf(got)) {
			++stats.bad_count;  // GPU 出现 NaN/Inf 是硬错误
			continue;
		}
		const double denom = std::fabs(ref);
		if (denom < skip_abs) {
			++stats.skipped;
			continue;
		}
		const double rel_err = std::fabs(got - ref) / denom;
		if (rel_err > stats.max_rel_err) {
			stats.max_rel_err = rel_err;
			stats.worst_idx = i;
		}
		if (rel_err > rel_tol) {
			++stats.bad_count;
		}
	}
	stats.pass = (stats.bad_count == 0);
	return stats;
}

// 失败定位：不合规元素数 + 最大误差位置及其前后各 2 个元素（便于按行定位）。
void PrintMismatchDetail(const RunReport& report) {
	const CompareStats& cmp = report.cmp;
	const std::vector<float>& h_ref = *report.h_ref;
	const std::vector<float>& h_gpu = *report.h_gpu;
	std::printf("    mismatched elements: %lld (over tolerance or NaN/Inf)\n",
	            static_cast<long long>(cmp.bad_count));
	if (cmp.worst_idx < 0) return;
	const std::int64_t count = static_cast<std::int64_t>(h_gpu.size());
	const std::int64_t from = cmp.worst_idx > 2 ? cmp.worst_idx - 2 : 0;
	const std::int64_t to = cmp.worst_idx + 3 < count ? cmp.worst_idx + 3 : count;
	for (std::int64_t j = from; j < to; ++j) {
		std::printf("      [%lld] GPU %.6e / CPU %.6e%s\n", static_cast<long long>(j),
		            h_gpu[static_cast<std::size_t>(j)], h_ref[static_cast<std::size_t>(j)],
		            j == cmp.worst_idx ? "   <- worst" : "");
	}
}

// 报告渲染：每次测试调用独立输出 GPU 静态配置、启动配置、截图七项指标和验证明细。
// 设备快照由 GetDeviceInfo 缓存，故重复打印不重复查询 CUDA Runtime。
void PrintReport(const RunReport& report) {
	const DeviceInfo& device = GetDeviceInfo();
	std::printf(
	    "================================================================================\n");
	std::printf("Test Case: %s\n\n", report.kernel_name);

	std::printf("GPU Configuration\n");
	std::printf("  Device Name: %s\n", device.available ? device.name : "N/A");
	if (device.available) {
		std::printf("  Compute Capability: %d.%d\n", device.major, device.minor);
		std::printf("  SM Count: %d\n", device.sm_count);
		std::printf("  Global Memory: %.1f GB\n",
		            device.global_memory_bytes / (1024.0 * 1024.0 * 1024.0));
		std::printf("  Shared Memory per Block: %zu B\n", device.shared_memory_per_block);
		std::printf("  Memory Clock: %d kHz\n", device.memory_clock_khz);
		std::printf("  Memory Bus Width: %d bit\n", device.memory_bus_width_bits);
	}
	if (device.peak_dram_bandwidth_gbps > 0.0) {
		std::printf("  Peak DRAM Bandwidth: %.2f GB/s\n", device.peak_dram_bandwidth_gbps);
	} else {
		std::printf("  Peak DRAM Bandwidth: N/A\n");
	}

	std::printf("\nLaunch Configuration\n");
	std::printf("  Matrix Shape: %d x %d\n", report.rows, report.cols);
	if (report.host_api) {
		std::printf("  Execution Path: Host API\n");
		std::printf("  Grid: N/A\n");
		std::printf("  Block: N/A\n");
		std::printf("  Dynamic Shared Memory: N/A\n");
	} else {
		std::printf("  Execution Path: CUDA Kernel\n");
		std::printf("  Grid: %d\n", report.grid);
		std::printf("  Block: %d\n", report.block);
		std::printf("  Dynamic Shared Memory: %zu B\n", report.smem_bytes);
	}

	std::printf("\nResults\n");
	std::printf("  CPU & GPU Results Match: %s\n", report.cmp.pass ? "Yes" : "No");
	if (report.has_measurement) {
		const double speedup =
		    report.gpu.median_ms > 0.0 ? report.cpu_ms / report.gpu.median_ms : 0.0;
		const double utilization =
		    device.peak_dram_bandwidth_gbps > 0.0
		        ? 100.0 * report.bandwidth_gbps / device.peak_dram_bandwidth_gbps
		        : 0.0;
		std::printf("  CPU Time: %.3f ms (host reference)\n", report.cpu_ms);
		std::printf("  GPU Time: %.6f ms (median)\n", report.gpu.median_ms);
		std::printf("  Speedup: %.2fx\n", speedup);
		std::printf("  Effective Bandwidth: %.2f GB/s (logical: 1 read + 1 write per element)\n",
		            report.bandwidth_gbps);
		if (device.peak_dram_bandwidth_gbps > 0.0) {
			std::printf("  Peak DRAM Bandwidth: %.2f GB/s\n", device.peak_dram_bandwidth_gbps);
			std::printf("  Bandwidth Utilization: %.2f %%\n", utilization);
		} else {
			std::printf("  Peak DRAM Bandwidth: N/A\n");
			std::printf("  Bandwidth Utilization: N/A\n");
		}
	} else {
		std::printf("  CPU Time: N/A\n");
		std::printf("  GPU Time: N/A\n");
		std::printf("  Speedup: N/A\n");
		std::printf("  Effective Bandwidth: N/A\n");
		if (device.peak_dram_bandwidth_gbps > 0.0) {
			std::printf("  Peak DRAM Bandwidth: %.2f GB/s\n", device.peak_dram_bandwidth_gbps);
		} else {
			std::printf("  Peak DRAM Bandwidth: N/A\n");
		}
		std::printf("  Bandwidth Utilization: N/A\n");
	}

	std::printf("\nValidation\n");
	if (report.invalid_arguments) {
		std::printf("  Status: FAIL (invalid arguments: rows=%d, cols=%d, grid=%d, block=%d)\n",
		            report.rows, report.cols, report.grid, report.block);
	} else if (report.skip_reason != nullptr) {
		std::printf("  Status: PASS (%s; no output elements to compare)\n", report.skip_reason);
	} else {
		std::printf("  Max Relative Error: %.3e (tolerance: %.0e)\n", report.cmp.max_rel_err,
		            report.rel_tol);
		if (report.cmp.skipped > 0) {
			std::printf("  Skipped |ref| < 1e-30: %lld elements\n",
			            static_cast<long long>(report.cmp.skipped));
		}
		if (!report.cmp.pass) {
			PrintMismatchDetail(report);
		}
		std::printf("  Sampling: %d warmup, %d x %d iterations (%s)\n", report.warmup_iterations,
		            report.sample_count, report.iterations,
		            report.strict_benchmark ? "strict" : "development");
		if (report.strict_benchmark) {
			std::printf("  P5 / P95: %.6f / %.6f ms\n", report.gpu.p5_ms, report.gpu.p95_ms);
		}
		std::printf("  Sample out[0][0]: GPU %.6e / CPU ref %.6e\n", (*report.h_gpu)[0],
		            (*report.h_ref)[0]);
	}
	std::printf(
	    "================================================================================\n");
}

}  // namespace

bool test_softmax_kernel(SoftmaxKernel kernel, const char* kernel_name, int rows, int cols,
                         int grid, int block, std::size_t smem_bytes, bool strict_benchmark,
                         SoftmaxHostKernel host_kernel) {
	// 迭代口径：开发（默认）vs 严格两档，与 reduce 测试文件一致
	// （见 docs/benchmark-methodology.md）。严格档面向本机 RTX 4060 Laptop 调低采样量
	// （100 预热 + 21 组 × 1000 次，总 2.1 万次，约为原口径 1/10）：慢内核（如 v0
	// 单次 ~5 ms）整表严格基准由数十分钟降到约十分钟，21 组保留 P5/P95 分位分辨率。
	const int warmup_iterations = strict_benchmark ? 100 : 1;
	const int iterations = strict_benchmark ? 1000 : 100;
	const int sample_count = strict_benchmark ? 21 : 1;
	RunReport report;
	report.kernel_name = kernel_name;
	report.rows = rows;
	report.cols = cols;
	report.grid = grid;
	report.block = block;
	report.smem_bytes = smem_bytes;
	report.host_api = host_kernel != nullptr;
	report.strict_benchmark = strict_benchmark;
	report.warmup_iterations = warmup_iterations;
	report.iterations = iterations;
	report.sample_count = sample_count;
	report.rel_tol = kRelTol;

	// 契约防御：非法参数判 FAIL 而非崩溃（rows/cols < 0 无意义，cudaMalloc(0)
	// 未定义，grid/block < 1 无法启动）。
	if (rows < 0 || cols < 0 || grid < 1 || block < 1) {
		report.invalid_arguments = true;
		report.cmp.pass = false;
		PrintReport(report);
		return false;
	}

	// 元素总数。启动网格由调用方按被测内核的行映射给出（见 main.cu 的 RowMap /
	// GridFor）；空矩阵 rows == 0 时 grid 也须 >= 1，各内核在 rows == 0 时空转。
	const std::int64_t count = static_cast<std::int64_t>(rows) * cols;

	// ① 输入 + 主机参考。参考用 double 计算，避免参考值自身的舍入误差吃掉 1e-5 容差；
	// 其单次耗时一并报告（同机同构建的 CPU 侧量级参考，不参与判据）。
	std::vector<float> h_input(static_cast<std::size_t>(count));
	std::vector<float> h_ref(static_cast<std::size_t>(count));
	FillDeterministicInput(h_input, cols, count);
	CpuTimer cpu_timer;
	cpu_timer.Start();
	softmax_cpu(h_input.data(), h_ref.data(), rows, cols);
	const double cpu_ms = cpu_timer.StopMs();
	report.cpu_ms = cpu_ms;

	// ② 设备缓冲。空矩阵 / 空行时 cudaMalloc(0) 未定义，统一按 max(count, 1) 申请；内核
	// 该路径全走越界 / 空行分支、不读写数据，输出缓冲区保持原样（无可比元素，天然通过）。
	float *d_input = nullptr, *d_output = nullptr;
	const std::size_t in_bytes = static_cast<std::size_t>(count > 0 ? count : 1) * sizeof(float);
	CUDA_CHECK(cudaMalloc(&d_input, in_bytes));
	CUDA_CHECK(cudaMalloc(&d_output, in_bytes));
	if (count > 0) {  // count == 0 时无可拷数据
		CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), in_bytes, cudaMemcpyHostToDevice));
	}

	// ③ 启动器。只有 cudaLaunchKernel 才能以“运行期内核函数指针”启动，从而一份驱动
	// 复用所有版本；args 中需放与形参 const 修饰严格匹配的指针。grid / block /
	// smem_bytes 由调用方按被测内核的行映射给出（见 main.cu 的 RowMap / GridFor /
	// SmemFor）。host_kernel 非空时走主机 API 通道（如 cuDNN 对照参考，见 test.cuh），
	// 它自行启动计算，上面的启动配置只作为未使用参数存在。
	const float* d_input_arg = d_input;
	float* d_output_arg = d_output;
	int rows_arg = rows;
	int cols_arg = cols;
	void* args[] = {&d_input_arg, &d_output_arg, &rows_arg, &cols_arg};
	const auto launch = [&]() {
		if (host_kernel != nullptr) {
			host_kernel(d_input_arg, d_output_arg, rows_arg, cols_arg);
			return;
		}
		CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(grid), dim3(block),
		                            args, smem_bytes));
	};

	// 空形状不做计时（无有意义数据量）：直接报告并返回。
	if (count == 0) {
		report.skip_reason = "empty matrix / empty rows";
		report.cmp.pass = true;
		PrintReport(report);
		CUDA_CHECK(cudaFree(d_input));
		CUDA_CHECK(cudaFree(d_output));
		return true;
	}

	// ④ 预热 + 计时采样；有效带宽 =（输入读 count 个 float + 输出写 count 个 float）/
	// 中位耗时（逻辑数据量，每元素计 1 读 1 写；各版本实际读行次数 / 访问模式各异，
	// 低效会直接反映为更低的“有效带宽”，见 README）。
	report.gpu = MeasureKernel(launch, warmup_iterations, iterations, sample_count);
	const double bytes_per_run = 2.0 * static_cast<double>(count) * sizeof(float);
	report.bandwidth_gbps = bytes_per_run / (report.gpu.median_ms * 1e6);
	report.has_measurement = true;

	// ⑤ 拷回 GPU 输出，逐元素比对后渲染报告。
	std::vector<float> h_gpu(static_cast<std::size_t>(count));
	CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_output, in_bytes, cudaMemcpyDeviceToHost));
	report.cmp = CompareCellwise(h_ref, h_gpu, count, kRelTol, kSkipAbs);
	report.h_ref = &h_ref;
	report.h_gpu = &h_gpu;
	PrintReport(report);

	CUDA_CHECK(cudaFree(d_input));
	CUDA_CHECK(cudaFree(d_output));
	return report.cmp.pass;
}

// ============================================================================
// main.cu —— GEMM 入口：默认对固定场景做正确性验证 + 性能基准（先 cuBLAS 对照，再各被测
// 内核）；`--bench` 走多尺寸性能扫描，只计时并把结果写成 CSV。
// 接入新内核：向 kKernels 追加一项（短名、描述、内核地址、启动配置），模板内核在此显式指定
// 模板实参。
// ============================================================================

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include "include/sgemm_v0.cuh"
#include "include/sgemm_v1.cuh"
#include "include/sgemm_v2.cuh"
#include "include/test.cuh"

namespace {

// sgemm_v0 的启动配置：block 16×16 铺 M×N，每线程一个输出元素。
constexpr int kBlockX = 16;
constexpr int kBlockY = 16;
// sgemm_v1 的启动配置：每 block 起满 TILE_SIZE² 个线性线程覆盖一个输出 tile。
constexpr int kSgemmV1TileSize = 32;
constexpr int kSgemmV1Threads = kSgemmV1TileSize * kSgemmV1TileSize;
// sgemm_v2 的启动配置：同 v1，共享内存为静态分配（不计入动态共享内存）。
constexpr int kSgemmV2TileSize = 32;
constexpr int kSgemmV2Threads = kSgemmV2TileSize * kSgemmV2TileSize;

// 一次启动的配置：物理 block、每 block 覆盖的输出 tile 与动态共享内存字节数。
struct LaunchConfig {
	int block_x;
	int block_y;
	int tile_rows;
	int tile_cols;
	std::size_t smem_bytes;
	bool row_uses_grid_y;
};

// plot_name 用于 --bench 的 CSV 首列与性能曲线图例（保持简短）；description 用于正确性
// 报告的测试用例名（带实现要点说明）。
struct KernelEntry {
	const char* plot_name;
	const char* description;
	const void* kernel;
	LaunchConfig launch;
};

const KernelEntry kKernels[] = {
    {"sgemm_v0",
     "sgemm_v0 (one thread per output element, global-memory baseline)",
     reinterpret_cast<const void*>(sgemm_v0),
     {kBlockX, kBlockY, kBlockX, kBlockY, 0, false}},
    {"sgemm_v1",
     "sgemm_v1 (TILE_SIZE=32, original global-memory implementation)",
     reinterpret_cast<const void*>(sgemm_v1<kSgemmV1TileSize>),
     {kSgemmV1Threads, 1, kSgemmV1TileSize, kSgemmV1TileSize, 0, false}},
    {"sgemm_v2",
     "sgemm_v2 (TILE_SIZE=32, shared-memory tiling, one thread per output)",
     reinterpret_cast<const void*>(sgemm_v2<kSgemmV2TileSize>),
     {kSgemmV2Threads, 1, kSgemmV2TileSize, kSgemmV2TileSize, 0, true}},
};

struct Scenario {
	const char* label;
	int M;
	int N;
	int K;
};

// 固定的正常测试场景：每个内核和 cuBLAS 对照只运行一次。
const Scenario kNormalScenarios[] = {
    {"normal: 512x512x512", 512, 512, 512},
};

// ---- 扫描模式（--bench）的默认配置 ----
const char* const kCublasPlotName = "cuBLAS";
// 默认尺寸序列：128…4096 的正方形矩阵（绘图脚本按等距分类轴处理这些尺寸）。
const int kDefaultSweepSizes[] = {128, 256, 512, 1024, 2048, 4096};
// 默认产物落点：operators/gemm/bench/<扫描时刻>/gemm_bench.csv，每次扫描单开一个目录。
const char* const kDefaultCsvDir = "operators/gemm/bench";
const char* const kCsvFileName = "gemm_bench.csv";
constexpr int kDefaultWarmup = 1;
// 每个 (内核, 尺寸) 采样点的计时预算上限（ms），每组迭代数据此自适应，见 include/test.cuh。
constexpr double kDefaultBudgetMs = 200.0;

// csv_path 为空表示未指定，由 RunSweep 生成带时间戳的默认路径（见 DefaultCsvPath）。
struct SweepOptions {
	std::vector<int> sizes;
	std::string csv_path;
	int warmup = kDefaultWarmup;
	double budget_ms = kDefaultBudgetMs;
};

struct BenchPoint {
	const char* label;
	int size;
	double median_ms;
	double gflops;
	int iters;
	bool ok;
};

int DivUp(int value, int divisor) {
	return (value + divisor - 1) / divisor;
}

int GridX(int M, int N, const LaunchConfig& launch) {
	return launch.row_uses_grid_y ? DivUp(N, launch.tile_cols) : DivUp(M, launch.tile_rows);
}

int GridY(int M, int N, const LaunchConfig& launch) {
	return launch.row_uses_grid_y ? DivUp(M, launch.tile_rows) : DivUp(N, launch.tile_cols);
}

// 默认 CSV 路径：operators/gemm/bench/<YYYYmmddHHMMSS>/gemm_bench.csv —— 按扫描时刻分目录。
std::string DefaultCsvPath() {
	const std::time_t now = std::time(nullptr);
	std::tm local_time = {};
	localtime_r(&now, &local_time);
	char stamp[32] = {};
	std::strftime(stamp, sizeof(stamp), "%Y%m%d%H%M%S", &local_time);
	return std::string(kDefaultCsvDir) + "/" + stamp + "/" + kCsvFileName;
}

void PrintUsage(const char* program) {
	std::printf(
	    "usage: %s [--bench [--sizes 128,256,...] [--csv <path>] [--warmup <n>] [--budget <ms>]]\n",
	    program);
	std::printf("  (no option): correctness + performance on the default 512x512x512 scenario\n");
	std::printf("  --bench    : sweep the sizes and write kernel timings as CSV\n");
	std::printf(
	    "               performance only, no correctness check; sizes are square matrices\n");
}

// 解析 --bench 的选项；缺参数或取值非法时返回 false（调用方打印用法）。
bool ParseSweepOptions(int argc, char** argv, SweepOptions* options) {
	for (const int size : kDefaultSweepSizes) options->sizes.push_back(size);

	for (int i = 2; i < argc; ++i) {
		const char* arg = argv[i];
		const char* value = (i + 1 < argc) ? argv[i + 1] : nullptr;
		if (std::strcmp(arg, "--sizes") == 0 && value != nullptr) {
			options->sizes.clear();
			const std::string list(value);
			std::size_t start = 0;
			while (true) {
				const std::size_t comma = list.find(',', start);
				const std::string token = list.substr(
				    start, comma == std::string::npos ? std::string::npos : comma - start);
				const int size = std::atoi(token.c_str());
				if (size < 1) return false;
				options->sizes.push_back(size);
				if (comma == std::string::npos) break;
				start = comma + 1;
			}
			++i;
		} else if (std::strcmp(arg, "--csv") == 0 && value != nullptr) {
			options->csv_path = value;
			++i;
		} else if (std::strcmp(arg, "--warmup") == 0 && value != nullptr) {
			options->warmup = std::atoi(value);
			if (options->warmup < 0) return false;
			++i;
		} else if (std::strcmp(arg, "--budget") == 0 && value != nullptr) {
			options->budget_ms = std::atof(value);
			if (options->budget_ms <= 0.0) return false;
			++i;
		} else {
			return false;
		}
	}
	return !options->sizes.empty();
}

// 记录一个采样点并打印进度：失败点（median_ms < 0）只打印警告，不写入 CSV。
void RecordPoint(std::vector<BenchPoint>* points, const char* label, int size, double median_ms,
                 int iters) {
	if (median_ms < 0.0 || iters < 1) {
		std::printf("  %-10s %5d^3: SKIPPED (launch or configuration failed)\n", label, size);
		points->push_back({label, size, 0.0, 0.0, 0, false});
		return;
	}
	const double gflops = 2.0 * size * size * size / (median_ms * 1e6);
	std::printf("  %-10s %5d^3: %9.4f ms  %8.2f GFLOP/s (x%d iters)\n", label, size, median_ms,
	            gflops, iters);
	points->push_back({label, size, median_ms, gflops, iters, true});
}

// 写出 CSV（表头 label,size,median_ms,gflops,iters）；先按需创建父目录，路径不可写时
// 退回标准输出。
void WriteCsv(const std::vector<BenchPoint>& points, const std::string& path) {
	std::error_code dir_error;
	const std::filesystem::path parent = std::filesystem::path(path).parent_path();
	if (!parent.empty()) std::filesystem::create_directories(parent, dir_error);

	std::FILE* file = std::fopen(path.c_str(), "w");
	if (file == nullptr) {
		std::printf("\n[warn] cannot open %s for writing; CSV follows on stdout\n", path.c_str());
		file = stdout;
	} else {
		std::printf("\nCSV written to %s\n", path.c_str());
	}
	std::fprintf(file, "label,size,median_ms,gflops,iters\n");
	for (const BenchPoint& point : points) {
		if (!point.ok) continue;
		std::fprintf(file, "%s,%d,%.4f,%.3f,%d\n", point.label, point.size, point.median_ms,
		             point.gflops, point.iters);
	}
	if (file != stdout) std::fclose(file);
}

int RunSweep(const SweepOptions& options) {
	std::printf("==== GEMM bench sweep (performance only, no correctness check) ====\n");
	std::printf("sizes:");
	for (const int size : options.sizes) std::printf(" %d", size);
	std::printf(
	    "\nsampling: %d warmup, adaptive iterations (budget %.0f ms per point, median of 3 "
	    "samples)\n\n",
	    options.warmup, options.budget_ms);

	std::vector<BenchPoint> points;
	// 与正确性报告保持同一顺序：先 cuBLAS 对照，再各被测内核。
	for (const int size : options.sizes) {
		int iters = 0;
		const double median_ms =
		    bench_cublas_sgemm(size, size, size, options.warmup, options.budget_ms, &iters);
		RecordPoint(&points, kCublasPlotName, size, median_ms, iters);
	}
	for (const KernelEntry& kernel : kKernels) {
		for (const int size : options.sizes) {
			int iters = 0;
			const double median_ms = bench_gemm_kernel(
			    kernel.kernel, size, size, size, GridX(size, size, kernel.launch),
			    GridY(size, size, kernel.launch), kernel.launch.block_x, kernel.launch.block_y,
			    kernel.launch.smem_bytes, options.warmup, options.budget_ms, &iters);
			RecordPoint(&points, kernel.plot_name, size, median_ms, iters);
		}
	}

	int skipped = 0;
	for (const BenchPoint& point : points) skipped += point.ok ? 0 : 1;
	const std::string csv_path = options.csv_path.empty() ? DefaultCsvPath() : options.csv_path;
	WriteCsv(points, csv_path);
	std::printf("points: %zu, skipped: %d\n", points.size(), skipped);
	// 扫描只采集性能数据、不作判据：部分点失败不视为整体失败，全部失败才返回错误码。
	return skipped == static_cast<int>(points.size()) ? 1 : 0;
}

int RunValidation() {
	std::printf("==== GEMM test: correctness (tolerance 1e-3) + performance ====\n");
	std::printf("kernels under test = %zu\n", sizeof(kKernels) / sizeof(kKernels[0]));
	std::printf("scenario: 512x512x512\n");
	std::printf("sampling: 1 warmup + 100 iterations\n\n");

	bool all_ok = true;
	int passed = 0;
	int total = 0;
	for (const Scenario& scenario : kNormalScenarios) {
		char name[192];
		std::snprintf(name, sizeof(name), "cublasSgemm (CUBLAS_PEDANTIC_MATH) | %s",
		              scenario.label);
		const bool ok = test_cublas_sgemm(name, scenario.M, scenario.N, scenario.K);
		all_ok = all_ok && ok;
		passed += ok ? 1 : 0;
		++total;
		std::printf("\n");
	}
	for (const KernelEntry& kernel : kKernels) {
		for (const Scenario& scenario : kNormalScenarios) {
			char name[192];
			std::snprintf(name, sizeof(name), "%s | %s", kernel.description, scenario.label);
			const bool ok = test_gemm_kernel(
			    kernel.kernel, name, scenario.M, scenario.N, scenario.K,
			    GridX(scenario.M, scenario.N, kernel.launch),
			    GridY(scenario.M, scenario.N, kernel.launch), kernel.launch.block_x,
			    kernel.launch.block_y, kernel.launch.smem_bytes);
			all_ok = all_ok && ok;
			passed += ok ? 1 : 0;
			++total;
			std::printf("\n");
		}
	}

	std::printf("==== Result: %d/%d items PASS%s ====\n", passed, total,
	            all_ok ? "" : ", FAIL present");
	return all_ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
	if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
		SweepOptions options;
		if (!ParseSweepOptions(argc, argv, &options)) {
			PrintUsage(argv[0]);
			return 2;
		}
		return RunSweep(options);
	}
	if (argc > 1) {
		PrintUsage(argv[0]);
		return 2;
	}
	return RunValidation();
}

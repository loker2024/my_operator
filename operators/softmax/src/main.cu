// ============================================================================
// main.cu —— Softmax 测试执行入口：把被测内核注册给可复用测试驱动
// test_softmax_kernel（声明 test.cuh / 实现 test.cu），按开关跑三类场景，退出码
// 0 = 全部通过。
//
// 接入新版本内核：在同名版本 .cuh/.cu 中添加声明与实现后，
// 添加声明与实现后，向下方 kKernels 表追加 {名字, 函数指针, RowMap} 一项即可自动
// 复用全部测试场景 —— RowMap 给出该内核的行映射，由 GridFor/SmemFor 推出每个场景
// 的启动配置（见各版本 .cuh）。
//
// 可选段：cuDNN 对照参考（厂商库 cudnnSoftmaxForward，实现见 softmax_cudnn.cu）分两层开关
// —— CMake 的 -DSOFTMAX_WITH_CUDNN=ON 决定“是否编译与链接 cuDNN”（链接期依赖，须在
// 构建前定，值由 CMake 缓存），main() 的 kEnableCudnnReference 决定“链接后跑不跑”。
// 它走主机 API、不是可注册的内核，故不进入 kKernels。
//
// 构建：cmake --build build --target softmax && ./build/operators/softmax/softmax
// ============================================================================

#include <cstddef>  // std::size_t
#include <cstdio>   // printf / std::snprintf

#include "include/online_softmax_v0.cuh"
#include "include/online_softmax_v1.cuh"
#include "include/online_softmax_v2.cuh"
#include "include/online_softmax_v3.cuh"
#include "include/online_softmax_v3_false.cuh"
#include "include/online_softmax_v4.cuh"
#include "operator_common/cuda_check.h"
#include "include/softmax_cudnn.cuh"
#include "include/softmax_reference.cuh"
#include "include/softmax_v0.cuh"
#include "include/softmax_v1.cuh"
#include "include/softmax_v2.cuh"
#include "include/softmax_v3.cuh"
#include "include/softmax_v4.cuh"
#include "include/softmax_v5.cuh"
#include "include/test.cuh"

namespace {

// 每 block 线程数（默认 256）：kThreadPerRow 用它铺满行号，其余行映射用作每行协作
// 线程数。256 同时满足各版本的“2 的幂 / 32 的倍数”约束（见各版本 .cuh）。
constexpr int kBlock = 256;

// 行映射 —— 决定每个被测内核的启动 grid 与动态共享内存（见 GridFor / SmemFor；各内核
// 的行映射以各版本 .cuh 的启动约束为准）。
enum class RowMap {
	kThreadPerRow,         // 每线程处理一行：grid = ceil(rows / block)，无共享内存
	kBlockPerRow,          // 每行一个 block + smem 折半树形归约：grid = rows
	kBlockPerRowShuffle,   // 每行一个 block + 两级 warp shuffle 归约：grid = rows
	kBlockPerRowRowCache,  // 每行一个 block + smem 缓存整行：grid = rows
	kGridStrideRow,        // 每 block grid-stride 处理多行：grid = min(rows, 上限)
};

// online-v4（kGridStrideRow）启动 block 数上限系数：grid = min(rows, SM 数 × 本值)。
// 取设备 SM 数的倍数而非固定值，避免硬编码设备；block 数落在“若干波驻留”区间 ——
// 既保证并行度足以逼近带宽，又让每个 block 经 grid-stride 复用同一份寄存器 / 静态
// __shared__ 连续处理多行（rows 远大于该值时不再启动 rows 个 block）。
constexpr int kGridStrideBlocksPerSm = 32;
int g_grid_stride_blocks = 1;  // 由 main() 按当前设备 SM 数初始化

// 被测内核表。row_map 为该内核的行映射方式。
struct KernelEntry {
	const char* name;      // 打印用名字
	SoftmaxKernel kernel;  // 内核函数指针
	RowMap row_map;        // 行映射方式（决定启动配置）
};

const KernelEntry kKernels[] = {
    {"softmax_v0 (one thread per row, 3 serial passes)", softmax_v0, RowMap::kThreadPerRow},
    {"softmax_v1 (one block per row, smem tree reduction)", softmax_v1, RowMap::kBlockPerRow},
    {"softmax_v2 (one block per row, warp shuffle reduction)", softmax_v2,
     RowMap::kBlockPerRowShuffle},
    {"softmax_v3 (one block per row, float4 vectorized)", softmax_v3, RowMap::kBlockPerRowShuffle},
    {"softmax_v4 (one block per row, whole-row smem cache of x)", softmax_v4,
     RowMap::kBlockPerRowRowCache},
    {"softmax_v5 (one block per row, 2 global reads + float4, exp in smem)", softmax_v5,
     RowMap::kBlockPerRowRowCache},
    {"online_softmax_v0 (one thread per row, single-pass online reduction)", online_softmax_v0,
     RowMap::kThreadPerRow},
    {"online_softmax_v1 (one block per row, online reduction + 2-level shuffle)", online_softmax_v1,
     RowMap::kBlockPerRowShuffle},
    {"online_softmax_v2 (one block per row, online reduction + float4)", online_softmax_v2,
     RowMap::kBlockPerRowShuffle},
    {"online_softmax_v3 (one block per row, online reduction + register tiling)", online_softmax_v3,
     RowMap::kBlockPerRowShuffle},
    {"online_softmax_v3_false (one block per row, runtime-indexed cache -> local memory)",
     online_softmax_v3_false, RowMap::kBlockPerRowShuffle},
    {"online_softmax_v4 (grid-stride multi-row, online reduction + float4)", online_softmax_v4,
     RowMap::kGridStrideRow},
};

// 测试场景。label 仅用于打印；rows × cols 为矩阵形状，grid/smem 由被测内核的
// 行映射方式推出。
struct Scenario {
	const char* label;
	int rows;
	int cols;
};

template <size_t N>
constexpr size_t CountOf(const Scenario (&)[N]) {
	return N;
}

// 场景组 A：正常流程 —— 大规模形状（行数远超 block，各版本的 grid 都够大）。
const Scenario kNormalScenarios[] = {
    {"normal: 4096x4096 (~64 MiB input)", 4096, 4096},
    {"normal: 16384x1024 (wide rows)", 16384, 1024},
};

// 场景组 B：边界条件 —— 覆盖各版本的关键路径：行数边界（block-1 / block / block+1 /
// 2*block-1）对应 kThreadPerRow 的越界空转与末 block 仅余 1 行；行宽边界（block±1 /
// 2*block-1，以及 warp 边界 31/32/33）对应 kBlockPerRow* 的多轮 stride、空转线程与
// warp shuffle 归约（仅前几个 warp 持有数据、其余以归约单位元参与）；float4 边界（4 的
// 倍数、向量主循环在 block 线程数附近）对应 kBlockPerRowShuffle / kBlockPerRowRowCache 的
// 向量化路径，非 4 倍列宽场景覆盖标量回退。kGridStrideRow 的「同一 block 处理多行」由
// 正常流程组（rows 4096 / 16384 远大于上限）覆盖，本组行数 ≤ 2*block 时退化为「一行一
// block」。各版本的完整覆盖矩阵见 README。
const Scenario kBoundaryScenarios[] = {
    {"boundary: 1x1, smallest non-empty", 1, 1},
    // 行数边界（v0 关键路径）
    {"boundary: (block-1)x3, one thread short of full", kBlock - 1, 3},
    {"boundary: blockx3, exactly one full block", kBlock, 3},
    {"boundary: (block+1)x3, one extra row needs a 2nd block", kBlock + 1, 3},
    {"boundary: (2*block-1)x3, last block holds 1 row", 2 * kBlock - 1, 3},
    // 行宽边界（v1/v2/v3/v4/v5 标量回退关键路径）
    {"boundary: 3x(block-1), one column short of full", 3, kBlock - 1},
    {"boundary: 3xblock, row fits exactly one round", 3, kBlock},
    {"boundary: 3x(block+1), row needs two rounds", 3, kBlock + 1},
    {"boundary: 3x(2*block-1), 2nd round holds 1 column", 3, 2 * kBlock - 1},
    // warp 边界（v2/v3/v4/v5 关键路径：行宽恰 1 个 warp、差 1 与超 1）
    {"boundary: 3x(32-1), one column short of a warp", 3, 31},
    {"boundary: 3x32, row is exactly one warp", 3, 32},
    {"boundary: 3x(32+1), 2nd warp holds 1 column", 3, 33},
    // float4 对齐边界（v3/v4/v5 向量化关键路径：列宽为 4 的倍数 → 行首 16 B 对齐，
    // 向量主循环在 block 线程数附近 —— n4 = block-1 / block / block+1 /
    // 2*block-1，对应空转 / 恰 1 轮满载 / 第 2 轮仅余 1 个 / 第 2 轮余 block-1 个）
    {"boundary: 3x(4*(block-1)), float4 one short of full", 3, 4 * (kBlock - 1)},
    {"boundary: 3x(4*block), float4 exactly one round", 3, 4 * kBlock},
    {"boundary: 3x(4*(block+1)), float4 2nd round holds 1", 3, 4 * (kBlock + 1)},
    {"boundary: 3x(4*(2*block-1)), float4 2nd round holds block-1", 3, 4 * (2 * kBlock - 1)},
};

// 场景组 C：异常 / 健壮性 —— 空矩阵、空行。
const Scenario kAbnormalScenarios[] = {
    {"abnormal: 0x1024, empty matrix", 0, 1024},
    {"abnormal: 3x0, one empty row per line", 3, 0},
};

// 按行映射求“覆盖全部行”的最小 grid；rows == 0 时也须 >= 1（内核在 rows == 0 时空转，
// 见各版本 .cuh）。
int GridFor(int rows, RowMap row_map) {
	if (rows == 0) return 1;
	switch (row_map) {
		case RowMap::kThreadPerRow:
			return (rows + kBlock - 1) / kBlock;
		case RowMap::kBlockPerRow:
		case RowMap::kBlockPerRowShuffle:
		case RowMap::kBlockPerRowRowCache:
			return rows;
		case RowMap::kGridStrideRow:
			return rows < g_grid_stride_blocks ? rows : g_grid_stride_blocks;
	}
	return 1;  // 不可达
}

// 按行映射求每 block 的动态共享内存字节数（见各版本 .cuh）：仅
// kBlockPerRow（block 个归约中间量）与 kBlockPerRowRowCache（整行缓存、随列宽增长）非
// 零，其余内核不依赖动态共享内存。
std::size_t SmemFor(RowMap row_map, int cols) {
	switch (row_map) {
		case RowMap::kThreadPerRow:
			return 0;  // 无共享内存
		case RowMap::kBlockPerRow:
			return static_cast<std::size_t>(kBlock) * sizeof(float);  // 归约中间量：block 个 float
		case RowMap::kBlockPerRowShuffle:
			return 0;  // 仅内部静态 __shared__ 中转
		case RowMap::kBlockPerRowRowCache:
			return static_cast<std::size_t>(cols) * sizeof(float);  // 整行缓存，随列宽
		case RowMap::kGridStrideRow:
			return 0;  // 仅内部静态 __shared__ 中转
	}
	return 0;  // 不可达
}

// 对单个内核跑一遍启用场景，返回该内核是否全部 PASS。
// 开关：enable_boundary 执行 B/C 组（默认只跑 A 组正常流程，避免小形状拖慢
// 日常迭代）；strict_benchmark 透传给 test_softmax_kernel 的严格采样口径。
bool RunScenarios(const KernelEntry& kern, bool enable_boundary = false,
                  bool strict_benchmark = false, int* passed = nullptr, int* total = nullptr) {
	bool all_ok = true;
	int local_passed = 0;
	int local_total = 0;

	const auto run_group = [&](const char* title, const Scenario* scenarios, size_t count) {
		std::printf("== %s ==\n", title);
		for (size_t i = 0; i < count; ++i) {
			const Scenario& s = scenarios[i];
			char full_name[192];
			std::snprintf(full_name, sizeof(full_name), "%s | %s", kern.name, s.label);
			const bool ok = test_softmax_kernel(kern.kernel, full_name, s.rows, s.cols,
			                                    GridFor(s.rows, kern.row_map), kBlock,
			                                    SmemFor(kern.row_map, s.cols), strict_benchmark);
			all_ok = ok && all_ok;
			local_passed += ok ? 1 : 0;
			local_total += 1;
		}
	};

	run_group("[A] normal", kNormalScenarios, CountOf(kNormalScenarios));
	if (enable_boundary) {
		run_group("[B] boundary", kBoundaryScenarios, CountOf(kBoundaryScenarios));
		run_group("[C] abnormal / robustness", kAbnormalScenarios, CountOf(kAbnormalScenarios));
	} else {
		std::printf("== [B] boundary / [C] abnormal & robustness ==\n");
		std::printf("    skipped (enable_boundary = false, off by default)\n");
	}

	if (passed != nullptr) *passed += local_passed;
	if (total != nullptr) *total += local_total;
	return all_ok;
}

}  // namespace

int main() {
	// 开关集中在此，默认关闭：只跑正常流程 + 快速性能。需要全量回归或严格基准时
	// 改为 true。
	constexpr bool kEnableBoundary = false;
	constexpr bool kStrictBenchmark = false;
#ifdef SOFTMAX_WITH_CUDNN
	// cuDNN 对照参考的运行开关（只在 -DSOFTMAX_WITH_CUDNN=ON 的构建里存在，见
	// CMakeLists.txt）：true 跑对照段并与自研内核同场对照，false 跳过。CMake 选项
	// 配一次即可（值会缓存），此后只改这里。
	constexpr bool kEnableCudnnReference = true;
#endif

	// online-v4 的 grid-stride 启动 block 数上限按当前设备 SM 数确定（见
	// kGridStrideBlocksPerSm）：避免硬编码设备，同时保证 block 数落在“若干波驻留”。
	int dev = 0;
	CUDA_CHECK(cudaGetDevice(&dev));
	int sm_count = 0;
	CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));
	if (sm_count < 1) sm_count = 1;
	g_grid_stride_blocks = sm_count * kGridStrideBlocksPerSm;

	std::printf("==== Softmax test: correctness (tolerance 1e-5) + performance ====\n");
	std::printf("block = %d; kernels under test = %zu\n", kBlock,
	            sizeof(kKernels) / sizeof(kKernels[0]));
	std::printf("grid-stride cap = min(rows, %d SMs x %d) = %d blocks (online-v4)\n", sm_count,
	            kGridStrideBlocksPerSm, g_grid_stride_blocks);
	std::printf("switches: enable_boundary = %s, strict_benchmark = %s\n",
	            kEnableBoundary ? "true" : "false", kStrictBenchmark ? "true" : "false");
#ifdef SOFTMAX_WITH_CUDNN
	std::printf("          cudnn_reference = %s (cuDNN reference section)\n",
	            kEnableCudnnReference ? "true" : "false");
#endif
	std::printf("\n");

	bool all_ok = true;
	int passed = 0;
	int total = 0;

	for (const KernelEntry& kern : kKernels) {
		std::printf("---------------- %s ----------------\n", kern.name);
		const bool ok = RunScenarios(kern, kEnableBoundary, kStrictBenchmark, &passed, &total);
		all_ok = ok && all_ok;
		std::printf("\n");
	}

#ifdef SOFTMAX_WITH_CUDNN
	// cuDNN 对照参考（跑不跑由上面的 kEnableCudnnReference 决定，能否编译进来取决于
	// 构建时的 -DSOFTMAX_WITH_CUDNN=ON）：跑同一批场景、复用同一套 1e-5 判据与计时
	// 口径，用于与自研内核做同场量级 / 性能对照。它不注册进 kKernels（主机 API、内部
	// 自行启动，无 RowMap 概念），故按场景组单列一段。
	// 注意：ACCURATE 与本仓库内核的累加顺序不同，max_err 只作量级对照（见 README）。
	if (kEnableCudnnReference) {
		constexpr const char* kCudnnName = "cudnn_softmax (ACCURATE, MODE_INSTANCE)";
		const auto run_cudnn = [&](const char* title, const Scenario* scenarios, size_t count) {
			std::printf("== %s ==\n", title);
			for (size_t i = 0; i < count; ++i) {
				char full_name[192];
				std::snprintf(full_name, sizeof(full_name), "%s | %s", kCudnnName,
				              scenarios[i].label);
				// grid / block / smem 对主机 API 无意义，填合法值即可（见 test.cuh）。
				const bool ok =
				    test_softmax_kernel(nullptr, full_name, scenarios[i].rows, scenarios[i].cols, 1,
				                        kBlock, 0, kStrictBenchmark, softmax_cudnn);
				all_ok = ok && all_ok;
				passed += ok ? 1 : 0;
				total += 1;
			}
		};

		std::printf(
		    "---------------- [reference] cuDNN cudnnSoftmaxForward (ACCURATE) ----------------\n");
		run_cudnn("[A] normal", kNormalScenarios, CountOf(kNormalScenarios));
		if (kEnableBoundary) {
			run_cudnn("[B] boundary", kBoundaryScenarios, CountOf(kBoundaryScenarios));
			run_cudnn("[C] abnormal / robustness", kAbnormalScenarios, CountOf(kAbnormalScenarios));
		} else {
			std::printf("== [B] boundary / [C] abnormal & robustness ==\n");
			std::printf("    skipped (enable_boundary = false, off by default)\n");
		}
		std::printf("\n");
	} else {
		std::printf(
		    "---------------- [reference] cuDNN cudnnSoftmaxForward (ACCURATE) ----------------\n");
		std::printf("    skipped (kEnableCudnnReference = false)\n\n");
	}
#endif

	// 正确性全部通过则退出码 0，否则 1（便于脚本化判断）。
	std::printf("==== Result: %d/%d items PASS%s ====\n", passed, total,
	            all_ok ? "" : ", FAIL present");
	return all_ok ? 0 : 1;
}

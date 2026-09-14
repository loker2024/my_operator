// ============================================================================
// main.cu —— Softmax 测试执行入口：把被测内核注册给可复用测试驱动
// test_softmax_kernel（声明 test.cuh / 实现 test.cu），固定运行两组正常场景，退出码
// 0 = 全部通过。
//
// 接入新版本内核：在同名版本 .cuh/.cu 中添加声明与实现后，
// 添加声明与实现后，向下方 kKernels 表追加 {名字, 函数指针, RowMap} 一项即可自动
// 复用全部测试场景 —— RowMap 给出该内核的行映射，由 GridFor/SmemFor 推出每个场景
// 的启动配置（见各版本 .cuh）。
//
// cuDNN 对照参考（厂商库 cudnnSoftmaxForward，实现见 softmax_cudnn.cu）仅在 CMake 的
// -DSOFTMAX_WITH_CUDNN=ON 构建中编译、链接并自动运行。它走主机 API、不是可注册的
// 内核，故不进入 kKernels。
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

// 正常测试场景：大规模形状（行数远超 block，各版本的 grid 都够大）。
const Scenario kNormalScenarios[] = {
    {"normal: 4096x4096 (~64 MiB input)", 4096, 4096},
    {"normal: 16384x1024 (wide rows)", 16384, 1024},
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

// 对单个内核跑两组正常场景，返回该内核是否全部 PASS。
// 固定使用快速采样口径：1 次预热与 100 次计时迭代。
bool RunScenarios(const KernelEntry& kern, int* passed = nullptr, int* total = nullptr) {
	bool all_ok = true;
	int local_passed = 0;
	int local_total = 0;

	for (size_t i = 0; i < CountOf(kNormalScenarios); ++i) {
		const Scenario& s = kNormalScenarios[i];
		char full_name[192];
		std::snprintf(full_name, sizeof(full_name), "%s | %s", kern.name, s.label);
		const bool ok = test_softmax_kernel(kern.kernel, full_name, s.rows, s.cols,
		                                    GridFor(s.rows, kern.row_map), kBlock,
		                                    SmemFor(kern.row_map, s.cols), false);
		all_ok = ok && all_ok;
		local_passed += ok ? 1 : 0;
		local_total += 1;
	}

	if (passed != nullptr) *passed += local_passed;
	if (total != nullptr) *total += local_total;
	return all_ok;
}

}  // namespace

int main() {
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
	std::printf("scenarios: 4096x4096 and 16384x1024; sampling: 1 warmup + 100 iterations\n\n");

	bool all_ok = true;
	int passed = 0;
	int total = 0;

	for (const KernelEntry& kern : kKernels) {
		std::printf("---------------- %s ----------------\n", kern.name);
		const bool ok = RunScenarios(kern, &passed, &total);
		all_ok = ok && all_ok;
		std::printf("\n");
	}

#ifdef SOFTMAX_WITH_CUDNN
	// cuDNN 对照参考在 -DSOFTMAX_WITH_CUDNN=ON 构建中自动运行同一批正常场景，复用
	// 同一套 1e-5 判据与快速计时口径，用于与自研内核做同场量级 / 性能对照。它不注册进
	// kKernels（主机 API、内部自行启动，无 RowMap 概念），故单列一段。
	// 注意：ACCURATE 与本仓库内核的累加顺序不同，max_err 只作量级对照（见 README）。
	constexpr const char* kCudnnName = "cudnn_softmax (ACCURATE, MODE_INSTANCE)";
	std::printf("---------------- cuDNN cudnnSoftmaxForward (ACCURATE) ----------------\n");
	for (size_t i = 0; i < CountOf(kNormalScenarios); ++i) {
		char full_name[192];
		std::snprintf(full_name, sizeof(full_name), "%s | %s", kCudnnName, kNormalScenarios[i].label);
		// grid / block / smem 对主机 API 无意义，填合法值即可（见 test.cuh）。
		const bool ok = test_softmax_kernel(nullptr, full_name, kNormalScenarios[i].rows,
		                                    kNormalScenarios[i].cols, 1, kBlock, 0, false, softmax_cudnn);
		all_ok = ok && all_ok;
		passed += ok ? 1 : 0;
		total += 1;
	}
	std::printf("\n");
#endif

	// 正确性全部通过则退出码 0，否则 1（便于脚本化判断）。
	std::printf("==== Result: %d/%d items PASS%s ====\n", passed, total,
	            all_ok ? "" : ", FAIL present");
	return all_ok ? 0 : 1;
}

// ============================================================================
// main.cu —— Softmax 测试执行入口：把被测内核注册给可复用测试驱动
// test_softmax_kernel（声明 test.cuh / 实现 test.cu），按开关跑三类场景，退出码
// 0 = 全部通过。
//
// 接入新版本内核：在 softmax.cuh/.cu（或独立实现单元如 online_softmax.cuh/.cu）
// 添加声明与实现后，向下方 kKernels 表追加 {名字, 函数指针, RowMap} 一项即可自动
// 复用全部测试场景 —— RowMap 给出该内核的行映射，由 GridFor/SmemFor 推出每个场景
// 的启动配置（见 softmax.cuh 各版本）。
//
// 构建：cmake --build build --target softmax && ./build/operators/softmax/softmax
// ============================================================================

#include <cstddef>  // std::size_t
#include <cstdio>   // printf / std::snprintf
#include <cstring>  // std::strcmp

#include "online_softmax.cuh"
#include "operator_common/BenchConfig.h"
#include "operator_common/cuda_check.h"
#include "softmax.cuh"
#include "test.cuh"

namespace {

// 每 block 线程数（默认 256）：v0/online-v0 用它铺满行号，v1/v2/v3/v4/v5/online-v1/
// online-v2/online-v3/online-v3_false/online-v4 用作每行的协作线程数。256 同时满足 v1 的"2 的幂"（折半归约）与 v2/v3/
// v4/v5/online-v1/online-v2/online-v3/online-v3_false/online-v4 的"32 的倍数"（warp shuffle）约束，见 softmax.cuh /
// online_softmax.cuh 各版本的启动约束。
constexpr int kBlock = 256;

// 行映射与启动配置 —— 决定每个场景的启动 grid 与动态共享内存，见 softmax.cuh
// 各版本的启动约束。
enum class RowMap {
	kThreadPerRow,  // v0/online-v0：每线程处理一行，grid = ceil(rows / block)，无共享内存
	kBlockPerRow,         // v1：每行一个 block，grid = rows，smem = block * sizeof(float)
	kBlockPerRowShuffle,  // v2/v3/online-v1/online-v2/online-v3/online-v3_false：每行一个 block +
	                      //     两级 warp shuffle 归约，grid = rows，无动态共享内存（内部仅静态
	                      //     __shared__ 中转）。v3 与 online-v2 的 float4 向量化只
	                      //     影响行内访问，启动配置同 v2；online-v1 的在线归约同理
	                      //     （见 online_softmax.cuh v1/v2）
	kBlockPerRowRowCache,  // v4/v5：每行一个 block + 动态共享内存缓存整行（v4 缓存
	                       //     x、v5 缓存 exp），grid = rows，smem = cols * sizeof
	                       //     (float)（随行宽，启动约束见 softmax.cuh v4/v5）
	kGridStrideRow,        // online-v4：每 block 经 grid-stride 循环处理多行，
	                       //     grid = min(rows, SM 数 × kGridStrideBlocksPerSm)，
	                       //     无动态共享内存（内部仅静态 __shared__ 中转，见
	                       //     online_softmax.cuh v4）
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
    {"softmax_v0 (每线程处理一行, 行内串行三遍)", softmax_v0, RowMap::kThreadPerRow},
    {"softmax_v1 (每行一个 block, 块内树形归约)", softmax_v1, RowMap::kBlockPerRow},
    {"softmax_v2 (每行一个 block, warp shuffle 归约)", softmax_v2, RowMap::kBlockPerRowShuffle},
    {"softmax_v3 (每行一个 block, float4 向量化)", softmax_v3, RowMap::kBlockPerRowShuffle},
    {"softmax_v4 (每行一个 block, 整行 smem 缓存 x 一遍读)", softmax_v4,
     RowMap::kBlockPerRowRowCache},
    {"softmax_v5 (每行一个 block, 全局读2遍 float4, smem 存 exp)", softmax_v5,
     RowMap::kBlockPerRowRowCache},
    {"online_softmax_v0 (每线程处理一行, 单趟在线归约)", online_softmax_v0, RowMap::kThreadPerRow},
    {"online_softmax_v1 (每行一个 block, 单趟在线归约 + 两级 shuffle 合并)", online_softmax_v1,
     RowMap::kBlockPerRowShuffle},
    {"online_softmax_v2 (每行一个 block, 单趟在线归约 + float4)", online_softmax_v2,
     RowMap::kBlockPerRowShuffle},
    {"online_softmax_v3 (每行一个 block, 单趟在线归约 + 寄存器分片缓存)", online_softmax_v3,
     RowMap::kBlockPerRowShuffle},
    {"online_softmax_v3_false (每行一个 block, 单趟在线归约 + 运行期下标缓存→local memory)",
     online_softmax_v3_false, RowMap::kBlockPerRowShuffle},
    {"online_softmax_v4 (grid-stride 多行处理, 单趟在线归约 + float4)", online_softmax_v4,
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
    {"正常: 4096x4096 (约 64 MiB 输入)", 4096, 4096},
    {"正常: 16384x1024 (宽行场景)", 16384, 1024},
};

// 场景组 B：边界条件 —— 行数在 v0 的"block 线程铺满行号"覆盖边界附近（差 1 /
// 恰满载 / 超 1 / 末 block 仅余 1 行），列宽在 v1/v2/v3/v4/v5 与 online-v1/online-v2/
// online-v4 的"行内协作"边界附近（online-v1 的覆盖路径同 v2/v3：stride 扫行 + 块内
// shuffle 合并，无向量化分派；online-v2 与 online-v4 的覆盖路径同 v3：另按列宽分派
// float4；差 1 / 恰满载 / 超 1 / 第 2 轮仅余 1 列，以及 warp 边界附近的 31/32/33 ——
// v2/v3/v4/v5 与 online-v1/online-v2/online-v4 warp shuffle 归约的关键路径：仅前几个
// warp 持有数据、其余 warp 以归约单位元参与）；两种维度分别覆盖各版本的越界空转、
// 空转线程与多轮 stride 等路径。行数边界组行数 ≤ 2*block < grid-stride 上限（768），
// 对 online-v4 走「grid = rows、一行一 block」的退化路径；grid-stride「同一 block
// 处理多行」路径由正常流程组（rows 4096 / 16384 > 768）覆盖。
// 非 4 倍列宽场景对 v3/v4/v5 与 online-v2/online-v4 走标量回退；末尾另补 4 个 float4
// 对齐边界（列宽为 4 的倍数、向量主循环在 block 线程数附近）专测 v3/v4/v5 与
// online-v2/online-v4 的向量化路径，对 v0/v1/v2 与 online-v1 只是多一轮 stride 冗余
// 覆盖（无副作用）。
const Scenario kBoundaryScenarios[] = {
    {"边界: 1x1, 最小非空", 1, 1},
    // 行数边界（v0 关键路径）
    {"边界: (block-1)x3, 线程差 1 满载", kBlock - 1, 3},
    {"边界: blockx3, 恰 1 个 block 满线程", kBlock, 3},
    {"边界: (block+1)x3, 多 1 行需第 2 个 block", kBlock + 1, 3},
    {"边界: (2*block-1)x3, 末 block 仅余 1 行", 2 * kBlock - 1, 3},
    // 行宽边界（v1/v2/v3/v4/v5 标量回退关键路径）
    {"边界: 3x(block-1), 行宽差 1 满载", 3, kBlock - 1},
    {"边界: 3xblock, 行宽恰 1 轮满载", 3, kBlock},
    {"边界: 3x(block+1), 行宽需 2 轮", 3, kBlock + 1},
    {"边界: 3x(2*block-1), 第 2 轮仅余 1 列", 3, 2 * kBlock - 1},
    // warp 边界（v2/v3/v4/v5 关键路径：行宽恰 1 个 warp、差 1 与超 1）
    {"边界: 3x(32-1), warp 内差 1 列满载", 3, 31},
    {"边界: 3x32, 行宽恰 1 个 warp 满载", 3, 32},
    {"边界: 3x(32+1), 第 2 个 warp 仅 1 列有效", 3, 33},
    // float4 对齐边界（v3/v4/v5 向量化关键路径：列宽为 4 的倍数 → 行首 16 B 对齐，
    // 向量主循环在 block 线程数附近 —— n4 = block-1 / block / block+1 /
    // 2*block-1，对应空转 / 恰 1 轮满载 / 第 2 轮仅余 1 个 / 第 2 轮余 block-1 个）
    {"边界: 3x(4*(block-1)), float4 差 1 满载", 3, 4 * (kBlock - 1)},
    {"边界: 3x(4*block), float4 恰 1 轮满载", 3, 4 * kBlock},
    {"边界: 3x(4*(block+1)), float4 第 2 轮仅余 1 个", 3, 4 * (kBlock + 1)},
    {"边界: 3x(4*(2*block-1)), float4 第 2 轮余 block-1 个", 3, 4 * (2 * kBlock - 1)},
};

// 场景组 C：异常 / 健壮性 —— 空矩阵、空行。
const Scenario kAbnormalScenarios[] = {
    {"异常: 0x1024, 空矩阵", 0, 1024},
    {"异常: 3x0, 每行 0 列(空行)", 3, 0},
};

// 按行映射方式求“覆盖全部行”的最小 grid；rows == 0 时也须 >= 1（内核以
// row >= rows 越界空转，见 softmax.cuh / online_softmax.cuh）。kBlockPerRow /
// kBlockPerRowShuffle（含 online-v1/online-v2/online-v3/online-v3_false）/ kBlockPerRowRowCache 都是每行一个
// block → grid = rows；kGridStrideRow（online-v4）用 grid-stride 循环处理多行 →
// grid = min(rows, SM 数 × kGridStrideBlocksPerSm)。
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

// 按行映射方式求每 block 的动态共享内存（字节数，见 softmax.cuh / online_softmax.cuh）；
// 除 v4/v5（kBlockPerRowRowCache 缓存整行、随列宽 cols 增长）外均与行宽无关。
std::size_t SmemFor(RowMap row_map, int cols) {
	switch (row_map) {
		case RowMap::kThreadPerRow:
			return 0;  // v0/online-v0 无共享内存
		case RowMap::kBlockPerRow:
			return static_cast<std::size_t>(kBlock) * sizeof(float);  // v1 动态共享内存
		case RowMap::kBlockPerRowShuffle:
			return 0;  // v2/v3/online-v1/online-v2/online-v3/online-v3_false 仅用内部静态
			           // __shared__ 中转，无需动态共享内存
		case RowMap::kBlockPerRowRowCache:
			return static_cast<std::size_t>(cols) * sizeof(float);  // v4/v5 整行缓存，随行宽
		case RowMap::kGridStrideRow:
			return 0;  // online-v4 仅用内部静态 __shared__ 中转，无需动态共享内存
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

	run_group("[A] 正常流程", kNormalScenarios, CountOf(kNormalScenarios));
	if (enable_boundary) {
		run_group("[B] 边界条件", kBoundaryScenarios, CountOf(kBoundaryScenarios));
		run_group("[C] 异常与健壮性", kAbnormalScenarios, CountOf(kAbnormalScenarios));
	} else {
		std::printf("== [B] 边界条件 / [C] 异常与健壮性 ==\n");
		std::printf("    已跳过（enable_boundary = false，默认关闭）\n");
	}

	if (passed != nullptr) *passed += local_passed;
	if (total != nullptr) *total += local_total;
	return all_ok;
}

}  // namespace

int main(int argc, char** argv) {
	// 命令行开关（默认开发档：只跑正常流程 + 快速性能；口径见
	// docs/benchmark-methodology.md）：
	//   --boundary 追加 [B] 边界条件 / [C] 异常与健壮性场景（全量回归）
	//   --strict   启用严格采样口径（采样量按设备算力分档）
	//   --full     等价于 --boundary --strict（全量回归 + 严格基准）
	bool enable_boundary = false;
	bool strict_benchmark = false;
	for (int i = 1; i < argc; ++i) {
		if (std::strcmp(argv[i], "--boundary") == 0) {
			enable_boundary = true;
		} else if (std::strcmp(argv[i], "--strict") == 0) {
			strict_benchmark = true;
		} else if (std::strcmp(argv[i], "--full") == 0) {
			enable_boundary = true;
			strict_benchmark = true;
		} else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
			std::printf("用法: %s [--boundary] [--strict] [--full]\n", argv[0]);
			return 0;
		} else {
			std::printf("未知参数: %s\n用法: %s [--boundary] [--strict] [--full]\n", argv[i], argv[0]);
			return 2;
		}
	}

	PrintDeviceInfo();
	std::printf("\n");

	// online-v4 的 grid-stride 启动 block 数上限按当前设备 SM 数确定（见
	// kGridStrideBlocksPerSm）：避免硬编码设备，同时保证 block 数落在“若干波驻留”。
	int dev = 0;
	CUDA_CHECK(cudaGetDevice(&dev));
	int sm_count = 0;
	CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));
	if (sm_count < 1) sm_count = 1;
	g_grid_stride_blocks = sm_count * kGridStrideBlocksPerSm;

	const StrictBenchConfig strict = MakeStrictBenchConfig();
	std::printf("==== Softmax 测试：正确性(容差 1e-5) + 性能 ====\n");
	std::printf("block = %d；被测内核 %zu 个\n", kBlock, sizeof(kKernels) / sizeof(kKernels[0]));
	std::printf("grid-stride 上限 = min(rows, %d SM × %d) = %d block（online-v4）\n", sm_count,
	            kGridStrideBlocksPerSm, g_grid_stride_blocks);
	std::printf("严格档采样: %d 次预热 + %d 组 x %d 次（基准 %d SM，本机 %d SM）\n",
	            strict.warmup_iterations, strict.sample_count, strict.iterations,
	            kBenchRefSmCount, BenchDeviceSmCount());
	std::printf("开关: enable_boundary = %s, strict_benchmark = %s\n\n",
	            enable_boundary ? "true" : "false", strict_benchmark ? "true" : "false");

	bool all_ok = true;
	int passed = 0;
	int total = 0;

	for (const KernelEntry& kern : kKernels) {
		std::printf("---------------- %s ----------------\n", kern.name);
		const bool ok = RunScenarios(kern, enable_boundary, strict_benchmark, &passed, &total);
		all_ok = ok && all_ok;
		std::printf("\n");
	}

	// 正确性全部通过则退出码 0，否则 1（便于脚本化判断）。
	std::printf("==== 结果：%d/%d 项 PASS，%s ====\n", passed, total,
	            all_ok ? "全部通过" : "存在 FAIL");
	return all_ok ? 0 : 1;
}

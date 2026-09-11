// ============================================================================
// main.cu —— Reduce 测试执行入口：把被测内核注册给可复用测试驱动
// test_reduce_kernel（声明 test.cuh / 实现 test.cu），按开关跑三类场景，退出码
// 0 = 全部通过。
//
// 接入新版本内核：在 reduce.cuh/.cu 添加声明与实现后，只需向下方 kKernels 表
// 追加 {名字, 函数指针, 每线程元素数} 一项，即可自动复用全部测试场景。模板
// 内核（reduce_v5）定义在 reduce.cuh，注册时取固定实例 reduce_v5<kBlock>
// 作为函数指针。
//
// 构建：cmake --build build --target reduce && ./build/operators/reduce/reduce
// ============================================================================

#include <cstddef>  // std::size_t
#include <cstdio>   // printf / std::snprintf
#include <cstring>  // std::strcmp

#include "operator_common/BenchConfig.h"
#include "operator_common/cuda_check.h"
#include "reduce.cuh"
#include "test.cuh"

namespace {

// 每 block 线程数（各内核均要求为 2 的幂；v4/v5 另要求 >= 64，v6/v7 要求 >= 32
// 且 <= 1024）。reduce_v5 以 reduce_v5<kBlock> 注册；修改本值时需同时确认模板
// 的 block 约束与测试场景仍匹配。
constexpr int kBlock = 256;

// 被测内核表。elems_per_thread：每线程加载的输入元素数，决定“覆盖 n 所需的
// grid”——v0/v1/v2 为 1，v3/v4/v5/v6 为 2（grid 减半），v7 为 4（float4 向量化，
// grid 再减半），GridFor 据此计算。v7 为 grid-stride 扫描，任意 grid >= 1 都完整
// 覆盖输入，该值只用于给出“多数线程单轮完成”的推荐网格。
struct KernelEntry {
	const char* name;      // 打印用名字
	ReduceKernel kernel;   // 内核函数指针
	int elems_per_thread;  // 每线程加载的元素数
};

const KernelEntry kKernels[] = {
    {"reduce_v0 (交错寻址)", reduce_v0, 1},
    {"reduce_v1 (连续寻址)", reduce_v1, 1},
    {"reduce_v2 (折半步长)", reduce_v2, 1},
    {"reduce_v3 (每线程 2 元素)", reduce_v3, 2},
    {"reduce_v4 (每线程 2 元素 + warp 归约)", reduce_v4, 2},
    {"reduce_v5 (常量 block + warp 归约)", reduce_v5<kBlock>, 2},
    {"reduce_v6 (每线程 2 元素 + warp shuffle)", reduce_v6, 2},
    {"reduce_v7 (float4 向量化 + warp shuffle)", reduce_v7, 4},
};

// 测试场景。label 仅用于打印；n 为输入元素个数；extra_grid 为在“恰好覆盖 n 的
// block 数”基础上额外多配的 block 数（验证“超配安全”）。
struct Scenario {
	const char* label;
	int n;
	int extra_grid;
};

template <size_t N>
constexpr size_t CountOf(const Scenario (&)[N]) {
	return N;
}

// 由场景求实际启动的 grid：
//   base = ceil(n / (block * elems_per_thread))，每 block 覆盖
//   block * elems_per_thread 个连续元素；n == 0 时 base 至少为 1；
//   再叠加 extra_grid 个冗余 block。
// 对 v7（grid-stride 扫描）该值为“推荐网格”而非“精确覆盖所需”：其循环按
// gridDim 联合步进，grid >= 1 即完整覆盖，超出部分只空转（写 0），仍安全。
int GridFor(int n, int block, int elems_per_thread, int extra_grid) {
	const int span = block * elems_per_thread;
	int base = (n + span - 1) / span;
	if (base < 1) base = 1;
	return base + extra_grid;
}

// 场景组 A：正常流程 —— 大规模形状（grid 恰好覆盖输入）。
const Scenario kNormalScenarios[] = {
    {"正常: n=2^20, 对齐", 1 << 20, 0},
    {"正常: n=2^20+1000, 尾部非对齐", (1 << 20) + 1000, 0},
};

// 场景组 B：边界条件 —— 极小规模与 block 边界附近 / 整 block 满载的形状，
// 覆盖补 0、单 block、双 block（第二个 block 只有少量有效元素）等路径。
// 对 v7 额外补足 n % 4 尾部余数的向量化路径（余 2 的形状，配合 kBlock±1 的余
// 1 / 余 3 与 kBlock 的整除，覆盖 float4 主循环 + 标量尾部的全部组合）。
const Scenario kBoundaryScenarios[] = {
    {"边界: n=1, 单元素", 1, 0},
    {"边界: n=block, 恰 1 个 block 满载", kBlock, 0},
    {"边界: n=block-1, 差 1 满载", kBlock - 1, 0},
    {"边界: n=block+1, 需 2 个 block", kBlock + 1, 0},
    {"边界: n=2*block-1, 第 2 个 block 仅 1 个有效元素", 2 * kBlock - 1, 0},
    {"边界: n=block-2, float4 尾部余 2 元素", kBlock - 2, 0},
};

// 场景组 C：异常 / 健壮性 —— 空输入、超配 grid。
const Scenario kAbnormalScenarios[] = {
    {"异常: n=0, 空输入 (期望和=0)", 0, 0},
    {"健壮: n=2^18, grid 超配 +3 个冗余 block", 1 << 18, 3},
};

// 对单个内核跑一遍启用场景，返回该内核是否全部 PASS。
// 开关：enable_boundary 执行 B/C 组（默认只跑 A 组正常流程，避免极小形状拖慢
// 日常迭代）；strict_benchmark 透传给 test_reduce_kernel 的严格采样口径。
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
			const bool ok =
			    test_reduce_kernel(kern.kernel, full_name, s.n,
			                       GridFor(s.n, kBlock, kern.elems_per_thread, s.extra_grid),
			                       kBlock, strict_benchmark);
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

	const StrictBenchConfig strict = MakeStrictBenchConfig();
	std::printf("==== Reduce 测试：正确性(容差 1e-3) + 性能 ====\n");
	std::printf("block = %d；被测内核 %zu 个\n", kBlock, sizeof(kKernels) / sizeof(kKernels[0]));
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

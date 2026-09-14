// ============================================================================
// main.cu —— Reduce 测试执行入口：把被测内核注册给可复用测试驱动
// test_reduce_kernel（声明 test.cuh / 实现 test.cu），固定运行两组正常场景，退出码
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

// 正常测试场景：大规模对齐与尾部非对齐形状（grid 恰好覆盖输入）。
const Scenario kNormalScenarios[] = {
    {"正常: n=2^20, 对齐", 1 << 20, 0},
    {"正常: n=2^20+1000, 尾部非对齐", (1 << 20) + 1000, 0},
};

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
		const bool ok = test_reduce_kernel(
		    kern.kernel, full_name, s.n, GridFor(s.n, kBlock, kern.elems_per_thread, s.extra_grid),
		    kBlock, false);
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
	PrintDeviceInfo();
	std::printf("\n");

	std::printf("==== Reduce 测试：正确性(容差 1e-3) + 性能 ====\n");
	std::printf("block = %d；被测内核 %zu 个\n", kBlock, sizeof(kKernels) / sizeof(kKernels[0]));
	std::printf("场景：大规模对齐、尾部非对齐；快速采样：1 次预热 + 100 次迭代\n\n");

	bool all_ok = true;
	int passed = 0;
	int total = 0;

	for (const KernelEntry& kern : kKernels) {
		std::printf("---------------- %s ----------------\n", kern.name);
		const bool ok = RunScenarios(kern, &passed, &total);
		all_ok = ok && all_ok;
		std::printf("\n");
	}

	// 正确性全部通过则退出码 0，否则 1（便于脚本化判断）。
	std::printf("==== 结果：%d/%d 项 PASS，%s ====\n", passed, total,
	            all_ok ? "全部通过" : "存在 FAIL");
	return all_ok ? 0 : 1;
}

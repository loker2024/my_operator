// ============================================================================
// main.cu —— Softmax 测试执行入口：把被测内核注册给可复用测试驱动
// test_softmax_kernel（声明 test.cuh / 实现 test.cu），按开关跑三类场景，退出码
// 0 = 全部通过。
//
// 接入新版本内核：在 softmax.cuh/.cu 添加声明与实现后，只需向下方 kKernels 表
// 追加 {名字, 函数指针} 一项，即可自动复用全部测试场景（各版本输出约定与启动
// 约束一致：grid = rows、每 block 处理一行，见 softmax.cuh）。
//
// 构建：cmake --build build --target softmax && ./build/operators/softmax/softmax
// ============================================================================

#include <cstddef>  // std::size_t
#include <cstdio>   // printf / std::snprintf

#include "softmax.cuh"
#include "test.cuh"
#include "operator_common/cuda_check.h"

namespace {

// 每 block 线程数（2 的幂，默认 256；每行 1 个 block，行内列维由 blockDim 个
// 线程协同遍历，见 softmax.cuh 的启动约束）。
constexpr int kBlock = 256;

// 被测内核表。各版本共享同一启动/输出约定，无需逐版本记录额外配置。
struct KernelEntry {
  const char* name;      // 打印用名字
  SoftmaxKernel kernel;  // 内核函数指针
};

const KernelEntry kKernels[] = {
    {"softmax_v0 (每行一个 block, 朴素两遍规约)", softmax_v0},
};

// 测试场景。label 仅用于打印；rows × cols 为矩阵形状。
struct Scenario {
  const char* label;
  int rows;
  int cols;
};

template <size_t N>
constexpr size_t CountOf(const Scenario (&)[N]) {
  return N;
}

// 场景组 A：正常流程 —— 大规模形状（每 block 处理一整行，grid = rows）。
const Scenario kNormalScenarios[] = {
    {"正常: 4096x4096 (约 64 MiB 输入)", 4096, 4096},
    {"正常: 16384x1024 (宽行场景)", 16384, 1024},
};

// 场景组 B：边界条件 —— 极小形状、行宽在 block 边界（恰满载 / 差 1 / 超 1）与
// 列块轮换边界附近，覆盖 strided 列遍历、越界补空转、多 block 各行独立归约等路径。
const Scenario kBoundaryScenarios[] = {
    {"边界: 1x1, 最小非空", 1, 1},
    {"边界: 1x(block-1), 列差 1 满载", 1, kBlock - 1},
    {"边界: 1xblock, 恰 1 个列块满载", 1, kBlock},
    {"边界: 1x(block+1), 列需 2 轮", 1, kBlock + 1},
    {"边界: 1x(2*block-1), 第 2 列块仅 1 列", 1, 2 * kBlock - 1},
    {"边界: 3x(2*block), 多行多列块", 3, 2 * kBlock},
};

// 场景组 C：异常 / 健壮性 —— 空矩阵、空行。
const Scenario kAbnormalScenarios[] = {
    {"异常: 0x1024, 空矩阵", 0, 1024},
    {"异常: 3x0, 每行 0 列(空行)", 3, 0},
};

// 对单个内核跑一遍启用场景，返回该内核是否全部 PASS。
// 开关：enable_boundary 执行 B/C 组（默认只跑 A 组正常流程，避免小形状拖慢
// 日常迭代）；strict_benchmark 透传给 test_softmax_kernel 的严格采样口径。
bool RunScenarios(const KernelEntry& kern, bool enable_boundary = false,
                  bool strict_benchmark = false, int* passed = nullptr,
                  int* total = nullptr) {
  bool all_ok = true;
  int local_passed = 0;
  int local_total = 0;

  const auto run_group = [&](const char* title, const Scenario* scenarios,
                             size_t count) {
    std::printf("== %s ==\n", title);
    for (size_t i = 0; i < count; ++i) {
      const Scenario& s = scenarios[i];
      char full_name[192];
      std::snprintf(full_name, sizeof(full_name), "%s | %s", kern.name, s.label);
      const bool ok = test_softmax_kernel(kern.kernel, full_name, s.rows, s.cols,
                                          kBlock, strict_benchmark);
      all_ok = ok && all_ok;
      local_passed += ok ? 1 : 0;
      local_total += 1;
    }
  };

  run_group("[A] 正常流程", kNormalScenarios, CountOf(kNormalScenarios));
  if (enable_boundary) {
    run_group("[B] 边界条件", kBoundaryScenarios, CountOf(kBoundaryScenarios));
    run_group("[C] 异常与健壮性", kAbnormalScenarios,
              CountOf(kAbnormalScenarios));
  } else {
    std::printf("== [B] 边界条件 / [C] 异常与健壮性 ==\n");
    std::printf("    已跳过（enable_boundary = false，默认关闭）\n");
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

  PrintDeviceInfo();
  std::printf("\n");

  std::printf("==== Softmax 测试：正确性(容差 1e-5) + 性能 ====\n");
  std::printf("block = %d；被测内核 %zu 个\n", kBlock,
              sizeof(kKernels) / sizeof(kKernels[0]));
  std::printf("开关: enable_boundary = %s, strict_benchmark = %s\n\n",
              kEnableBoundary ? "true" : "false",
              kStrictBenchmark ? "true" : "false");

  bool all_ok = true;
  int passed = 0;
  int total = 0;

  for (const KernelEntry& kern : kKernels) {
    std::printf("---------------- %s ----------------\n", kern.name);
    const bool ok = RunScenarios(kern, kEnableBoundary, kStrictBenchmark,
                                 &passed, &total);
    all_ok = ok && all_ok;
    std::printf("\n");
  }

  // 正确性全部通过则退出码 0，否则 1（便于脚本化判断）。
  std::printf("==== 结果：%d/%d 项 PASS，%s ====\n", passed, total,
              all_ok ? "全部通过" : "存在 FAIL");
  return all_ok ? 0 : 1;
}

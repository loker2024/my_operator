// ============================================================================
// main.cu —— Softmax 测试执行入口：把被测内核注册给可复用测试驱动
// test_softmax_kernel（声明 test.cuh / 实现 test.cu），按开关跑三类场景，退出码
// 0 = 全部通过。
//
// 接入新版本内核：在 softmax.cuh/.cu 添加声明与实现后，向下方 kKernels 表追加
// {名字, 函数指针, RowMap} 一项即可自动复用全部测试场景 —— RowMap 给出该内核的
// 行映射，由 GridFor/SmemFor 推出每个场景的启动配置（见 softmax.cuh 各版本）。
//
// 构建：cmake --build build --target softmax && ./build/operators/softmax/softmax
// ============================================================================

#include <cstddef>  // std::size_t
#include <cstdio>   // printf / std::snprintf

#include "softmax.cuh"
#include "test.cuh"
#include "operator_common/cuda_check.h"

namespace {

// 每 block 线程数（默认 256；应为 2 的幂，见 softmax.cuh 的 v1 约束）：
// v0 用它铺满行号，v1 用它作为每行的协作线程数。
constexpr int kBlock = 256;

// 行映射方式 —— 决定每个场景的启动 grid 与动态共享内存，见 softmax.cuh 各版本。
enum class RowMap {
  kThreadPerRow,  // v0：每线程处理一行，grid = ceil(rows / block)，无共享内存
  kBlockPerRow,   // v1：每行一个 block，grid = rows，smem = block * sizeof(float)
};

// 被测内核表。row_map 为该内核的行映射方式。
struct KernelEntry {
  const char* name;      // 打印用名字
  SoftmaxKernel kernel;  // 内核函数指针
  RowMap row_map;        // 行映射方式（决定启动配置）
};

const KernelEntry kKernels[] = {
    {"softmax_v0 (每线程处理一行, 行内串行三遍)", softmax_v0,
     RowMap::kThreadPerRow},
    {"softmax_v1 (每行一个 block, 块内树形归约)", softmax_v1,
     RowMap::kBlockPerRow},
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

// 场景组 A：正常流程 —— 大规模形状（行数远超 block，两版本的 grid 都够大）。
const Scenario kNormalScenarios[] = {
    {"正常: 4096x4096 (约 64 MiB 输入)", 4096, 4096},
    {"正常: 16384x1024 (宽行场景)", 16384, 1024},
};

// 场景组 B：边界条件 —— 行数在 v0 的“block 线程铺满行号”覆盖边界附近（差 1 /
// 恰满载 / 超 1 / 末 block 仅余 1 行），列宽在 v1 的“行内协作”边界附近（差 1 /
// 恰满载 / 超 1 / 第 2 轮仅余 1 列）；两种维度分别覆盖两版本的越界空转、空转
// 线程与多轮 stride 等路径。
const Scenario kBoundaryScenarios[] = {
    {"边界: 1x1, 最小非空", 1, 1},
    // 行数边界（v0 关键路径）
    {"边界: (block-1)x3, 线程差 1 满载", kBlock - 1, 3},
    {"边界: blockx3, 恰 1 个 block 满线程", kBlock, 3},
    {"边界: (block+1)x3, 多 1 行需第 2 个 block", kBlock + 1, 3},
    {"边界: (2*block-1)x3, 末 block 仅余 1 行", 2 * kBlock - 1, 3},
    // 行宽边界（v1 关键路径）
    {"边界: 3x(block-1), 行宽差 1 满载", 3, kBlock - 1},
    {"边界: 3xblock, 行宽恰 1 轮满载", 3, kBlock},
    {"边界: 3x(block+1), 行宽需 2 轮", 3, kBlock + 1},
    {"边界: 3x(2*block-1), 第 2 轮仅余 1 列", 3, 2 * kBlock - 1},
};

// 场景组 C：异常 / 健壮性 —— 空矩阵、空行。
const Scenario kAbnormalScenarios[] = {
    {"异常: 0x1024, 空矩阵", 0, 1024},
    {"异常: 3x0, 每行 0 列(空行)", 3, 0},
};

// 按行映射方式求“覆盖全部行”的最小 grid；rows == 0 时也须 >= 1（内核以
// row >= rows 越界空转，见 softmax.cuh）。
int GridFor(int rows, RowMap row_map) {
  if (rows == 0) return 1;
  switch (row_map) {
    case RowMap::kThreadPerRow:
      return (rows + kBlock - 1) / kBlock;
    case RowMap::kBlockPerRow:
      return rows;
  }
  return 1;  // 不可达
}

// 按行映射方式求每 block 的动态共享内存（字节数，见 softmax.cuh）。
std::size_t SmemFor(RowMap row_map) {
  switch (row_map) {
    case RowMap::kThreadPerRow:
      return 0;
    case RowMap::kBlockPerRow:
      return static_cast<std::size_t>(kBlock) * sizeof(float);
  }
  return 0;  // 不可达
}

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
                                          GridFor(s.rows, kern.row_map), kBlock,
                                          SmemFor(kern.row_map),
                                          strict_benchmark);
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

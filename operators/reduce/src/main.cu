// ============================================================================
// main.cu —— 一维归约算子的执行入口：注册被测内核并运行测试（正确性 + 性能）
//
// 职责
//   本文件只做“执行”：把被测归约内核（reduce_v0 / reduce_v1 / reduce_v2…）
//   注册给可复用测试驱动 test_reduce_kernel（声明见 test.cuh，实现见
//   test.cu），按开关覆盖 正常流程 / 边界条件 / 异常与健壮性 三类场景，
//   并以退出码汇总结果（0 = 全部通过，供脚本化使用）。
//
// 新增内核版本的接入方式（可复用测试的关键）：
//   只需在下方 kernels 表追加一项 { 名字, 内核指针 }，即可自动对所有
//   场景复用同一套测试代码；无需改动测试驱动 test.cu 或场景表。
//
// 两个开关（均为函数参数，默认关闭；见 RunScenarios 的注释）：
//   enable_boundary   是否执行边界条件与异常/健壮性场景；
//   strict_benchmark  是否按严格口径采样性能（与 test_reduce_kernel 同名参数一致）。
//   默认状态下只跑“正常流程 + 快速性能口径”，日常开发迭代最快；需要全量
//   回归或出严格基准数字时，把 main() 中的开关改为 true 即可。
//
// 场景设计（各类别的意图与覆盖点，详见各场景组的注释）：
//   A. 正常流程 —— 常规大规模形状，验证基本正确性与性能；
//   B. 边界条件 —— 极小规模、恰好落在 block 边界附近 / 整 block 满载的
//      形状，覆盖“末尾 block 不满、多数线程补 0”与“grid 恰好为 1”等路径；
//   C. 异常与健壮性 —— 空输入（n == 0）、非法参数契约（由 test.cu 参数
//      防御判 FAIL 而不崩溃）、超配 grid（多余 block 全补 0 不影响结果）。
//
// 构建与运行（仓库根目录）：
//   CMake：  cmake -S . -B build && cmake --build build --target reduce
//            ./build/operators/reduce/reduce
//   或直接 nvcc（无需触碰工程构建目录）：
//   nvcc -std=c++17 -arch=native -I common/include \
//        operators/reduce/src/reduce.cu operators/reduce/src/test.cu \
//        operators/reduce/src/main.cu -o /tmp/reduce_run && /tmp/reduce_run
//
// 各文件职责总览
//   reduce.cuh / reduce.cu   算子接口声明与实现（被测试对象）
//   test.cuh  / test.cu      可复用测试驱动的声明与实现
//   main.cu                  执行入口（本文件）
// ============================================================================

#include <cstddef>  // std::size_t
#include <cstdio>   // printf / std::snprintf

#include "reduce.cuh"  // ReduceKernel 统一签名、reduce_v0 / reduce_v1 / reduce_v2 声明
#include "test.cuh"    // test_reduce_kernel 声明（内部经 reduce.cuh 引入算子接口）

namespace {

// ---------------------------------------------------------------------------
// 全局配置
// ---------------------------------------------------------------------------

// 每 block 线程数：所有内核都要求为 2 的幂（本文件所有场景共用）。
constexpr int kBlock = 256;

// ---------------------------------------------------------------------------
// 被测内核表
// ---------------------------------------------------------------------------
// 每个条目对应一个符合 ReduceKernel 签名的归约内核。新增版本（reduce_v2…）
// 时只需在数组末尾追加一项，下方所有场景会自动对新内核各跑一遍。
struct KernelEntry {
  const char* name;     // 打印用名字（区分版本与寻址方式）
  ReduceKernel kernel;  // 内核函数指针
};

const KernelEntry kKernels[] = {
    {"reduce_v0 (交错寻址)", reduce_v0},
    {"reduce_v1 (连续寻址)", reduce_v1},
    {"reduce_v2 (折半步长)", reduce_v2},
};

// ---------------------------------------------------------------------------
// 测试场景表
// ---------------------------------------------------------------------------
// label       仅用于打印，帮助区分场景意图；
// n           输入元素个数；
// extra_grid  在“恰好覆盖 n 所需的 block 数”基础上额外多启动的 block 数，
//             用于验证“超配 grid”时多余 block 全部补 0、不改变归约结果。
struct Scenario {
  const char* label;
  int n;
  int extra_grid;
};

// 编译期取场景表长度（配合下方基于数组的场景表，避免手写个数）。
template <size_t N>
constexpr size_t CountOf(const Scenario (&)[N]) {
  return N;
}

// 由场景参数计算实际启动的 grid 大小：
//   base = ceil(n / block)；n == 0 时也必须至少 1（空 block 全走补 0 分支，
//   结果恒为 0，可安全启动）；最后叠加 extra_grid 个冗余 block。
int GridFor(int n, int block, int extra_grid) {
  int base = (n + block - 1) / block;
  if (base < 1) base = 1;
  return base + extra_grid;
}

// 场景组 A：正常流程 —— 常规大规模形状（grid 恰好覆盖输入）。
const Scenario kNormalScenarios[] = {
    {"正常: n=2^20, 对齐", 1 << 20, 0},
    {"正常: n=2^20+1000, 尾部非对齐", (1 << 20) + 1000, 0},
};

// 场景组 B：边界条件 —— 极小规模与 block 边界附近 / 整 block 满载的形状，
// 覆盖补 0、单 block、双 block（第二个 block 只有少量有效元素）等路径。
const Scenario kBoundaryScenarios[] = {
    {"边界: n=1, 单元素", 1, 0},
    {"边界: n=block, 恰 1 个 block 满载", kBlock, 0},
    {"边界: n=block-1, 差 1 满载", kBlock - 1, 0},
    {"边界: n=block+1, 需 2 个 block", kBlock + 1, 0},
    {"边界: n=2*block-1, 第 2 个 block 仅 1 个有效元素", 2 * kBlock - 1, 0},
};

// 场景组 C：异常 / 健壮性 —— 空输入、超配 grid、非法契约。
const Scenario kAbnormalScenarios[] = {
    {"异常: n=0, 空输入 (期望和=0)", 0, 0},
    {"健壮: n=2^18, grid 超配 +3 个冗余 block", 1 << 18, 3},
};

// ---------------------------------------------------------------------------
// 场景执行
// ---------------------------------------------------------------------------
// 对单个内核跑一遍场景，返回 true 表示该内核全部 PASS。
//
// 开关（均为函数参数，默认关闭）：
//   enable_boundary   true  时执行 B 组（边界条件）与 C 组（异常/健壮性）；
//                     false（默认）时只跑 A 组正常流程，并在报告中标注跳过，
//                     避免日常迭代被大量极小形状（无性能意义的用例）拖慢。
//   strict_benchmark  true  时按严格口径采样（1000 预热 + 21 组 × 10000 次，
//                     输出中位数与 P5/P95）；false（默认）时快速模式（1 预热
//                     + 100 次）。该开关透传给 test_reduce_kernel 的同名参数，
//                     两处口径保持一致。
//
// passed / total 可选：非空时累加本次执行的 PASS 数与用例总数，便于 main 汇总。
bool RunScenarios(const KernelEntry& kern, bool enable_boundary = false,
                  bool strict_benchmark = false, int* passed = nullptr,
                  int* total = nullptr) {
  bool all_ok = true;
  int local_passed = 0;
  int local_total = 0;

  const auto run_group = [&](const char* title, const Scenario* scenarios, size_t count) {
    std::printf("== %s ==\n", title);
    for (size_t i = 0; i < count; ++i) {
      const Scenario& s = scenarios[i];
      char full_name[192];
      std::snprintf(full_name, sizeof(full_name), "%s | %s", kern.name, s.label);
      const bool ok = test_reduce_kernel(kern.kernel, full_name, s.n,
                                         GridFor(s.n, kBlock, s.extra_grid), kBlock,
                                         strict_benchmark);
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

int main() {
  // 开关集中在此处，默认均关闭：只跑正常流程 + 快速性能口径。
  // 需要全量回归（含边界条件 / 异常与健壮性）或严格基准数字时，改为 true。
  constexpr bool kEnableBoundary = false;
  constexpr bool kStrictBenchmark = false;

  std::printf("==== Reduce 测试：正确性(容差 1e-3) + 性能 ====\n");
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
    const bool ok = RunScenarios(kern, kEnableBoundary, kStrictBenchmark, &passed, &total);
    all_ok = ok && all_ok;
    std::printf("\n");
  }

  // 汇总：正确性全部通过则退出码 0，否则 1（便于脚本化判断）。
  std::printf("==== 结果：%d/%d 项 PASS，%s ====\n", passed, total,
              all_ok ? "全部通过" : "存在 FAIL");
  return all_ok ? 0 : 1;
}

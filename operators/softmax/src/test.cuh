#pragma once
// ============================================================================
// test.cuh —— Softmax 可复用测试驱动：接口声明（实现见 test.cu）
//   以函数指针接收任意符合 SoftmaxKernel 签名的 softmax 内核，使 v0… 共用同一
//   套正确性 + 性能测试；内核注册与场景编排见 main.cu。
//   口径：正确性相对误差 <= 1e-5（逐元素，跳过 |ref| 过小的元素）；性能采样分
//   “开发 / 严格”两档（见 docs/benchmark-methodology.md）。
// ============================================================================

#include "softmax.cuh"  // SoftmaxKernel 统一签名及被测算子接口

// 对单个内核跑单场景：生成输入并算 CPU 参考 → 分配/拷入 → 预热 → 计时采样 →
// 拷回比对逐元素（相对误差 + NaN/Inf）→ 输出报告。
//
// 参数：
//   kernel / kernel_name  被测 softmax 内核及其打印名
//   rows / cols           矩阵形状（rows >= 0, cols >= 0；空矩阵/空行期望不写
//                         任何元素、直接通过）
//   block                 每 block 线程数（2 的幂，默认 256；每行 1 个 block）
//   strict_benchmark     严格采样开关：false（默认）1 次预热 + 100 次迭代；
//                        true 时 1000 次预热 + 21 组 × 2000 次并输出 P5/P95
//                        （softmax 单次内核开销远大于 reduce，严格档迭代数相对
//                        reduce 的 21×10000 折半为 21×2000，避免全量基准过慢）
//
// 启动网格固定为 grid = max(rows, 1)：每 block 处理一行（见 softmax.cuh 的
// 启动约束），rows == 0 时靠越界判定空转。
//
// 返回：正确性通过返回 true（供 main 汇总并决定退出码）。
bool test_softmax_kernel(SoftmaxKernel kernel, const char* kernel_name, int rows,
                         int cols, int block, bool strict_benchmark = false);

#pragma once
// ============================================================================
// test.cuh —— Reduce 可复用测试驱动：接口声明（实现见 test.cu）
//   以函数指针接收任意符合 ReduceKernel 签名的归约内核，使 v0…v7 共用同一套
//   正确性 + 性能测试；内核注册与场景编排见 main.cu。
//   口径：正确性相对误差 <= 1e-3；性能采样分“开发 / 严格”两档（见
//   docs/benchmark-methodology.md）。
// ============================================================================

#include "reduce.cuh"  // ReduceKernel 统一签名及被测算子接口

// 对单个内核跑单场景：生成输入并算 CPU 参考 → 分配/拷入 → 预热 → 计时采样 →
// 拷回部分和并二次汇总 → 相对误差判据 → 输出报告。
//
// 参数：
//   kernel / kernel_name  被测归约内核及其打印名
//   n                    输入长度（>= 0；n == 0 为空输入，期望结果 0）
//   grid                 启动 block 数（>= 1；建议按内核覆盖口径给足——多余
//                        block 全补 0、不影响结果，由 main.cu 的 GridFor 计算）
//   block                每 block 线程数（2 的幂，默认 256）
//   strict_benchmark     严格采样开关：false（默认）1 次预热 + 100 次迭代；
//                        true 时 1000 次预热 + 21 组 × 10000 次并输出 P5/P95
//
// 返回：正确性通过返回 true（供 main 汇总并决定退出码）。
bool test_reduce_kernel(ReduceKernel kernel, const char* kernel_name, int n,
                        int grid, int block, bool strict_benchmark = false);

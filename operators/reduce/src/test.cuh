#pragma once
// ============================================================================
// test.cuh —— 一维归约算子的可复用测试驱动：接口声明
//
// 文件职责
//   只放置测试驱动的声明。三件套分工如下：
//     test.cuh  测试函数声明（本文件）
//     test.cu   测试函数实现（同一目录）
//     main.cu   执行入口：main() 中把各版本内核传给 test_reduce_kernel 运行
//
// 可复用性设计
//   测试函数把“归约内核函数本身”作为参数传入（类型 ReduceKernel，定义于
//   reduce.cuh），因此 v0 及后续新增的 v1/v2/… 内核只需在 main.cu 中换一个
//   函数名即可复用同一份测试代码，无需为每个版本复制整套测试逻辑。
//   该模式对应旧版 Reduce/src/reduce.cu 的 testReduceKernel（学习笔记 §14）。
//
// 正确性与性能口径（与 docs/benchmark-methodology.md 保持一致）：
//   * 相对误差 <= 1e-3（Reduce 容差）；
//   * 迭代口径 §3.2：开发模式 预热 1 次 + 100 次；严格模式 预热 1000 次 +
//     21 组 × 10000 次，输出中位数与 P5/P95。
// ============================================================================

#include "reduce.cuh"  // ReduceKernel 统一签名（以及被测算子接口）

// ---------------------------------------------------------------------------
// test_reduce_kernel —— 对任意符合 ReduceKernel 签名的归约内核做统一验证
// ---------------------------------------------------------------------------
// 流程：生成输入并算 CPU 参考 -> 设备端分配/拷贝 -> 预热 -> event 采样计时
//       -> 拷回部分和、主机汇总 -> 相对误差判据 -> 输出报告。
//
// 参数：
//   kernel            待测归约内核，如 reduce_v0
//   kernel_name       打印时显示的内核名，如 "reduce_v0"
//   n                 输入元素个数
//   grid              启动的 block 数，须 >= ceil(n / block)
//   block             每 block 线程数，2 的幂（默认 256）
//   strict_benchmark  true  时按严格口径采样并输出 P5/P95；
//                     false 时快速模式（默认）：1 次预热 + 100 次迭代
//
// 返回值：true 表示正确性校验通过（供 main 汇总并决定退出码）。
bool test_reduce_kernel(ReduceKernel kernel, const char* kernel_name, int n,
                        int grid, int block, bool strict_benchmark = false);

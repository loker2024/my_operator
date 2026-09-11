#pragma once
// ============================================================================
// test.cuh —— Softmax 可复用测试驱动：接口声明（实现见 test.cu）
//   以函数指针接收任意符合 SoftmaxKernel 签名的 softmax 内核，使 v0/v1/v2/v3/v4/v5
//   （及后续同签名内核）共用同一套正确性 + 性能测试；内核注册、场景编排与各
//   版本的启动配置见 main.cu。
//   口径：正确性相对误差 <= 1e-5（逐元素，跳过 |ref| 过小的元素）；性能采样分
//   “开发 / 严格”两档（见 docs/benchmark-methodology.md，与 reduce 测试一致）。
// ============================================================================

#include <cstddef>  // std::size_t

#include "softmax.cuh"  // SoftmaxKernel 统一签名及被测算子接口

// 对单个内核跑单场景：生成输入并算 CPU 参考 → 分配/拷入 → 预热 → 计时采样 →
// 拷回比对逐元素（相对误差 + NaN/Inf）→ 输出报告。
//
// 参数：
//   kernel / kernel_name  被测 softmax 内核及其打印名
//   rows / cols           矩阵形状（rows >= 0, cols >= 0；空矩阵/空行期望不写
//                         任何元素、直接通过）
//   grid / block          启动网格 / 每 block 线程数（均 >= 1）。各版本按自己的
//                         行映射启动（见 softmax.cuh）：v0 每线程处理一行 →
//                         grid = ceil(rows/block)；v1/v2/v3/v4/v5 每行一个 block →
//                         grid = rows
//   smem_bytes            每 block 动态共享内存字节数（v0 = 0；v1 =
//                         blockDim.x * sizeof(float)；v2/v3 = 0 —— 内部仅静态
//                         __shared__ 中转，与行宽无关；v4/v5 = cols *
//                         sizeof(float) —— v4 按行宽缓存整行 x、v5 缓存整行 exp，
//                         均随列宽增长）
//   strict_benchmark      严格采样开关：false（默认）1 次预热 + 100 次迭代；
//                         true 时按设备算力分档采样（基准机 24 SM = 100 次预热
//                         + 21 组 × 1000 次，更强 GPU 按 SM 数放大）并输出
//                         P5/P95，见 operator_common/BenchConfig.h
//
// 返回：正确性通过返回 true（供 main 汇总并决定退出码）。
bool test_softmax_kernel(SoftmaxKernel kernel, const char* kernel_name, int rows, int cols,
                         int grid, int block, std::size_t smem_bytes,
                         bool strict_benchmark = false);

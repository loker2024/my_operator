#pragma once
// ============================================================================
// reduce.cuh —— Reduce 算子接口声明（一维 fp32 数组整体求和）
//   实现见同目录 reduce.cu；可复用测试驱动见 test.cuh / test.cu，
//   测试执行入口见 main.cu。
//
// 两阶段归约约定：每 block 把负责的连续段归约为 1 个部分和，写入
// output[blockIdx.x]（故 output 至少要有 grid 个元素）；最终标量由调用方对
// output[0, grid) 再汇总一次。各版本均保持 ReduceKernel 签名与上述输出约定，
// 使 test_reduce_kernel 能以函数指针统一驱动 v0…v4。
//
// GPU 内核版本演进（详细推导与实测结论见 operators/reduce/README.md）：
//   v0 交错寻址 → v1 连续寻址 → v2 折半步长：共享内存树形归约逐档改进；
//   v3/v4 在归约前加入“每线程 2 元素”展开，v4 再把归约尾部换为 warp 展开。
// ============================================================================

#include <cuda_runtime.h>  // __global__、cudaError_t 等 CUDA 基本定义

// ---------------------------------------------------------------------------
// CPU 参考实现（主机端正确性基线）
// ---------------------------------------------------------------------------
// 逐元素累加。内部用 double 累加抑制 fp32 顺序累加的舍入误差，返回前转回
// float，便于与 GPU 的 fp32 结果同类型比较。
float reduce_cpu(const float* input, int n);

// ---------------------------------------------------------------------------
// GPU 归约内核（v0/v1/v2 的公共启动约束）
// ---------------------------------------------------------------------------
//   * grid >= ceil(n / blockDim.x)：越界元素补 0；多配的 block 全补 0、部分和
//     为 0，不影响最终结果（“超配安全”）；
//   * blockDim.x 为 2 的幂（默认 256），保证树形归约各轮均匀配对；
//   * 动态共享内存 = blockDim.x * sizeof(float)：内核用 extern __shared__
//     声明，大小由启动配置的第三参数给出。

// v0 交错寻址树形归约（正确性基线）：step 从 1 倍增，活跃线程按
//   tid % (2*step) == 0 交错合并，warp 内分歧明显。
__global__ void reduce_v0(const float* input, float* output, int n);

// v1 连续寻址：活跃线程改为连续前缀（index = tid*2*step < blockDim.x），
//   整条 warp 全活跃或全空闲，消除 v0 的 warp 内分歧；其余模型同 v0。
__global__ void reduce_v1(const float* input, float* output, int n);

// v2 折半步长：stride 自 blockDim.x/2 每轮折半到 1，线程 tid 合并
//   smem[tid] 与 smem[tid+stride]；部分和就地落回数组前端连续槽，读写下标
//   连续 → 无共享内存 bank 冲突。其余模型同 v0/v1。
__global__ void reduce_v2(const float* input, float* output, int n);

// v3 每线程 2 元素（标量加载）：block 覆盖 2*blockDim.x 个连续元素，线程先
//   寄存器预加和相距 blockDim.x 的两元素（越界跳过），再做 v2 折半归约。
//   与 v0/v1/v2 的差异：grid = ceil(n / (2*blockDim.x))（n == 0 时也须 >= 1），
//   smem 大小仍为 blockDim.x * sizeof(float)（存每线程 1 个预加和）。
__global__ void reduce_v3(const float* input, float* output, int n);

// v4 v3 + 末 warp 展开：折半归约到 stride = 32 即停，随后由 warp 0 对残留在
//   smem 前端的 32 个部分和做展开合并，省去剩余 5 轮 __syncthreads。
//   加载与覆盖口径同 v3（grid = ceil(n / (2*blockDim.x))）；另要求 blockDim.x
//   为 2 的幂且 >= 64（展开第一步需读取 smem[tid+32]）。
__global__ void reduce_v4(const float* input, float* output, int n);

// ---------------------------------------------------------------------------
// 归约内核统一签名
// ---------------------------------------------------------------------------
// 各版本输出约定一致（每 block 1 个部分和），故可共用同一测试驱动。
using ReduceKernel = void (*)(const float* input, float* output, int n);

#pragma once
// ============================================================================
// softmax.cuh —— Softmax 算子接口声明（行主序 fp32 矩阵，逐行数值稳定 softmax）
//   实现见同目录 softmax.cu；可复用测试驱动见 test.cuh/test.cu；执行入口见
//   main.cu。推导与实测结论见 operators/softmax/README.md。
//
// 数学定义（每行独立归一化）：
//   m_i    = max_j x[i,j]                    行最大值（max-shift，数值稳定用）
//   y[i,j] = exp(x[i,j] - m_i) / Σ_k exp(x[i,k] - m_i)
//   先减行最大使 exp 参数 <= 0：e^0 = 1 保证行和 >= 1、不会上溢，是大动态范围
//   输入下标准的稳定写法（减常数不改变 softmax 结果）。
//
// GPU 版本演进：
//   v0 每线程处理一行、行内串行三遍 —— 正确性基线（无共享内存 / 同步）；
//   v1 每行一个 block、行内由 blockDim.x 个线程协作 + 共享内存树形归约 ——
//      读合并、Σexp 累加误差由整行串行降为块内分段 + 树形量级；
//   v2 每行一个 block、行内遍历同 v1，块内归约改两级 warp shuffle —— 归约在
//      寄存器内完成、共享内存只做跨 warp 中转，块内同步降为 ~2 次 __syncthreads；
//   v3 行遍历与归约同 v2，行内访问按列宽分派：列宽为 4 的倍数时以 float4 向量
//      化（16 B 对齐，读/写指令数为标量 1/4），否则整行回退标量 —— 任意列宽
//      均正确，向量化收益限于对齐宽行；
// ============================================================================

#include <cuda_runtime.h>  // __global__、cudaError_t 等 CUDA 基本定义

// ---------------------------------------------------------------------------
// CPU 参考实现（主机端正确性基线）
// ---------------------------------------------------------------------------
// 逐行同公式，内部用 double 求最大值 / 指数 / 行和（参考值不自带 fp32 舍入
// 误差），返回前转回 float 便于与 GPU 的 fp32 结果同类型比较。
void softmax_cpu(const float* input, float* output, int M, int N);

// ---------------------------------------------------------------------------
// GPU 内核。公共输出契约：整矩阵算好并写满 output[0, rows*cols)，故
// test_softmax_kernel 能以函数指针统一驱动各版本；但各版本的行映射 / 启动配置
// 不同，见各自声明。行宽 N 均可任意（含 N == 0 的空行：遍历循环 0 次、不读不写）。
// ---------------------------------------------------------------------------

// v0 每线程处理一行（正确性基线）：
//   * row = blockIdx.x * blockDim.x + threadIdx.x，行内由该线程串行三遍遍历
//     （求行最大 → Σexp → 归一化写回）；每元素读行 3 次、warp 内各线程读不同
//     行 → 访存不合并；
//   * grid = ceil(M / blockDim.x)（M == 0 时也须 >= 1，由 row >= M 越界空转）；
//   * 无共享内存 / 同步（动态共享内存 = 0）。Σexp 为 fp32 串行累加，舍入误差
//     随行宽增长，是 v1 树形归约要解决的问题。
__global__ void softmax_v0(const float* input, float* output, const int M,
                           const int N);

// v1 每行一个 block，行内列维由 blockDim.x 个线程以 stride = blockDim.x 协同
//   遍历三次：行最大与行和各经一次共享内存折半树形归约，写回时第三次读行重算
//   exp（不再经共享内存）：
//   * row = blockIdx.x，grid = M（M == 0 时也须 >= 1，由 row >= M 越界空转）；
//   * warp 内各线程同轮访问相邻列 → 全局读合并；N 不足 blockDim.x 时多余线程
//     空转（不影响结果，-inf/0 为归约单位元）；N 超过 blockDim.x 时多轮 stride；
//   * 动态共享内存 = blockDim.x * sizeof(float)（只装规约中间量，与行宽无关）；
//   * blockDim.x 应为 2 的幂（默认 256），保证折半归约各轮均匀配对。
__global__ void softmax_v1(const float* input, float* output, const int M,
                           const int N);

// v2 每行一个 block，行内遍历与归约语义同 v1（三次 stride 扫行），仅把共享内存
//   折半树形归约换成两级 warp shuffle（helper 与实现见 softmax.cu）：
//   * row = blockIdx.x，grid = M（M == 0 时也须 >= 1，由 row >= M 越界空转）；
//   * 读合并 / 空转线程 / 多轮 stride 等语义同 v1；额外要求 blockDim.x 为 32 的
//     倍数（默认 256）且 <= 1024 —— shuffle 需整 warp 收敛，且 warp 部分值要能
//     装进归约中转的 warp_results[32]；
//   * 无动态共享内存（内部仅静态 __shared__ 做跨 warp 中转，与行宽无关）：
//     启动配置的 smem_bytes = 0。
__global__ void softmax_v2(const float* input, float* output, const int M,
                           const int N);

// v3 行遍历与归约同 v2（每行一个 block、两级 warp shuffle、无动态共享内存），
//   行内访问按列宽分派以支持任意列宽：
//   * N % 4 == 0：行首必然 16 B 对齐（row*N 为 4 的倍数），主循环每轮 stride 取
//     1 个 float4（4 列）向量化 —— 读/写指令数为标量的 1/4；无标量尾部；
//   * 否则：整行回退 v2 式标量三遍 —— 非 4 倍列宽时第 row>=1 行的行首 16 B 不
//     对齐，float4 重解释是未定义行为，尾列处理救不了跨行对齐（正确性不受
//     影响，仅无向量化收益）；
//   * 启动约束同 v2：row = blockIdx.x、grid = M、blockDim.x 为 32 的倍数（默认
//     256）且 <= 1024；smem_bytes = 0。
__global__ void softmax_v3(const float* input, float* output, const int M,
                           const int N);

// ---------------------------------------------------------------------------
// softmax 内核统一签名（仅输出约定一致；启动配置随内核版本由 main.cu 给出）
// ---------------------------------------------------------------------------
using SoftmaxKernel = void (*)(const float* input, float* output, int M, int N);

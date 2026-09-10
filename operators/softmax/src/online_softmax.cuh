#pragma once
// ============================================================================
// online_softmax.cuh —— online softmax 接口声明（行主序 fp32 矩阵，逐行归一化）
//   独立实现单元：实现见 online_softmax.cu，测试驱动见 test.cuh/test.cu，注册与
//   场景见 main.cu；推导与实测见 operators/softmax/README.md。
//
//   与 softmax.cuh（v0~v5）的差异：v0~v5 是「先求行最大 m、再求 Σexp」的两趟归约，
//   本单元改单趟 online 归约 —— 用二元组 (m, d) 在同一趟遍历里增量维护行最大与
//   分母：m' = max(m, x)、d' = d * exp(m - m') + exp(x - m')（d 先按新旧基准之差
//   缩放回新基准，再加新元素贡献），把求 m 与求 Σexp 合并为一趟全局读。
//
//   单元内各版本的差别只在行内 / 块内的协作方式与访存宽度：v0 每线程独占一行
//   （无协作），v1 每行一个 block、块内两级 warp shuffle 合并各线程的 (m, d)
//   （读合并，是后续「分块 + 在线合并」的 FlashAttention 形态的雏形），v2 在 v1
//   基础上按列宽分派 float4 向量化（仅影响行内访问宽度，归约与启动约束同 v1），
//   v3 用编译期定长 + 静态下标的寄存器分片缓存省掉第 2 遍全局读，v3_false 是 v3
//   的反面对照 —— 同样想缓存行，但用运行期下标写入定长数组，被 ptxas 降级为
//   local memory（“假寄存器”）。
// ============================================================================

#include <cuda_runtime.h>  // __global__、cudaError_t 等 CUDA 基本定义

// online_softmax_v0 每线程处理一行：
//   * row = blockIdx.x * blockDim.x + threadIdx.x；行内单趟 online 归约求 (m, d)，
//     再第二趟读行重算 exp(x - m) / d 写回；
//   * grid = ceil(M / blockDim.x)（M == 0 时也须 >= 1，由 row >= M 越界空转）；
//   * 无共享内存 / 同步（smem_bytes = 0）；行宽 N 任意，N == 0 时空行不读不写；
//   * warp 内各线程处理不同行 → 访存不合并，行内 fp32 串行累加误差随行宽增长。
__global__ void online_softmax_v0(const float* input, float* output, const int M, const int N);

// online_softmax_v1 每行一个 block，行内在线归约 + 块内两级 warp shuffle 合并：
//   * row = blockIdx.x，grid = M（M == 0 时也须 >= 1，由 row >= M 越界空转）；
//   * 每线程以 stride = blockDim.x 单趟在线归约自己的列子集得局部 (m, d)，再由
//     两级 warp shuffle 合并出整行 (m, d) —— 每 warp 先 shfl 归约为 1 个二元组，
//     warp 0 再合并 num_warps 个 warp 值并广播回全体（归约语义同 softmax.cuh
//     v2）；warp 内同轮访问相邻列 → 全局读合并；
//   * blockDim.x 须为 32 的倍数且 <= 1024（默认 256）：shuffle 需整 warp 收敛，
//     且各 warp 的二元组要装进中转用的 warp_m[32] / warp_d[32]；中转用静态
//     __shared__，无动态共享内存（smem_bytes = 0）；
//   * 未分到元素的线程 / warp（N 不整除或小于 blockDim.x）以 (m = -inf, d = 0)
//     作归约单位元参与合并 —— 合并时须跳过该侧的缩放，否则 (-inf) - (-inf) 的
//     NaN 会经 expf 污染整行分母；N == 0 的空行不读不写。
__global__ void online_softmax_v1(const float* input, float* output, const int M, const int N);

// online_softmax_v2 行遍历与归约同 v1（每行一个 block、单趟在线归约 + 两级 warp
//   shuffle 合并、无动态共享内存），行内访问按列宽分派以支持任意列宽：
//   * N % 4 == 0：行首必然 16 B 对齐（row*N 为 4 的倍数），单趟在线归约与写回的
//     主循环每轮 stride 取 1 个 float4（4 列，逐分量 mergeOnline / 算 exp 后整写
//     回）—— 读/写指令数为标量的 1/4，无标量尾部；
//   * 否则：整行回退 v1 式标量两遍 —— 非 4 倍列宽时第 row>=1 行的行首 16 B 不
//     对齐，float4 重解释是未定义行为，尾列处理救不了跨行对齐（正确性不受影响，
//     仅无向量化收益）；
//   * 启动约束同 v1：row = blockIdx.x、grid = M（M == 0 时也须 >= 1，由 row >= M
//     越界空转）、blockDim.x 为 32 的倍数且 <= 1024；smem_bytes = 0；N == 0 的空行
//     不读不写。
__global__ void online_softmax_v2(const float* input, float* output, const int M, const int N);

// online_softmax_v3 行遍历 / 归约 / 启动约束同 v1/v2（每行一个 block、单趟在线归约 +
//   两级 warp shuffle 合并；row = blockIdx.x、grid = M（M == 0 时也须 >= 1，由
//   row >= M 越界空转）、blockDim.x 为 32 的倍数且 <= 1024、无动态共享内存、
//   N == 0 的空行不读不写），仅把写回遍的 x 来源从「重读全局」改为「寄存器分片缓存」：
//   * 每线程元素数 ceil(N / blockDim.x) <= REG_TILE（常数，当前 16，见
//     online_softmax.cu 的 kRegTile）时走寄存器路径 —— 第 1 遍在线归约的同时把本线程
//     负责的列缓存进**编译期定长、静态下标**的寄存器数组（#pragma unroll 整体展开，
//     reg[k] 静态寻址故留在寄存器、不会降级为 local memory），写回遍直接取寄存器，
//     省掉第 2 遍全局读；
//   * 否则（列宽过大、寄存器分片装不下）自动回退 v1/v2 式两遍重读 —— 与 v0~v2 一样
//     对任意列宽均正确。分派条件只依赖 N 与 blockDim、对整 block 一致，两条分支内的
//     __syncthreads 均安全。
__global__ void online_softmax_v3(const float* input, float* output, const int M, const int N);

// online_softmax_v3_false 行遍历 / 归约 / 启动约束均同 v3，唯一区别在「寄存器分片」的写法
//   —— 本版刻意用**运行期下标**缓存本线程列（`reg_cache[count++]`）。寄存器不可被运行期
//   索引，ptxas 会把该定长数组整体降级为 **local memory**（物理是显存、仅靠 L1 缓存），
//   数组名字叫 reg 但并未落在寄存器 —— 即「假寄存器」。
//   * **不做列宽分派、无回退路径**，始终缓存：要求 ceil(N / blockDim.x) <= kRegTile（16，
//     见 online_softmax.cu；block = 256 时 N <= 4096）。超过则该定长数组越界写入（未定义
//     行为），由调用方保证 —— 去掉分派是为了与 v3 形成最干净的对照。
//   * 第 1 遍在线归约的同时把本线程负责的列以运行期下标写入定长数组，写回遍按同一顺序
//     取回，省掉第 2 遍全局读，但每次存取走 local memory。
//   本版是 v3 的反面对照：与 v3 逐项同构，只差下标是否静态，用于量化「local memory 缓存」
//   相对「真寄存器缓存」/「重读全局」的代价，实测见 README.md。
__global__ void online_softmax_v3_false(const float* input, float* output, const int M, const int N);

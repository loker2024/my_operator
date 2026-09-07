#pragma once
// ============================================================================
// reduce.cuh —— Reduce 算子接口声明（一维 fp32 数组整体求和）
//   实现见同目录 reduce.cu；可复用测试驱动见 test.cuh / test.cu，
//   测试执行入口见 main.cu。
//
// 两阶段归约约定：每 block 把负责的连续段归约为 1 个部分和，写入
// output[blockIdx.x]（故 output 至少要有 grid 个元素）；最终标量由调用方对
// output[0, grid) 再汇总一次。各版本均保持 ReduceKernel 签名与上述输出约定，
// 使 test_reduce_kernel 能以函数指针统一驱动 v0…v7。
//
// GPU 内核版本演进（详细推导与实测结论见 operators/reduce/README.md）：
//   v0 交错寻址 → v1 连续寻址 → v2 折半步长：共享内存树形归约逐档改进；
//   v3/v4 在归约前加入“每线程 2 元素”展开，v4 再把归约尾部换为 warp 展开；
//   v5 把 block 尺寸常量化（模板参数），让归约步骤在编译期整体展开；
//   v6 块内归约改两级 warp shuffle（寄存器直传），块内只剩一次 __syncthreads；
//   v7 加载改 float4 向量化 + grid-stride 扫描，块内归约沿用 v6。
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

// v5 v4 + 编译期常量化 block 尺寸：block 大小改由模板参数 BLOCK_SIZE 固定，
//   归约各级 if (BLOCK_SIZE >= ...) 均为编译期常量条件，编译器只保留当前
//   BLOCK_SIZE 需要的步骤并整体展开（无运行时归约循环，指令序列更短）。
//   覆盖口径同 v3/v4：grid = ceil(n / (2*BLOCK_SIZE))；要求 BLOCK_SIZE 为
//   2 的幂且 >= 64。模板定义必须内联在头文件：各实例化点（main.cu 等）据此
//   自行生成 reduce_v5<kBlock> 的设备代码，避免 -rdc=false 下跨翻译单元引用
//   __global__ 模板特化（nvcc 已对该用法发出弃用警告）。注册表取
//   reduce_v5<256> 实例（与 main.cu 的 kBlock = 256 对应）作函数指针。
template <int BLOCK_SIZE>
__global__ void reduce_v5(const float* input, float* output, int n) {
  extern __shared__ float smem[];

  const int tid = threadIdx.x;
  const int gid = blockIdx.x * (2 * BLOCK_SIZE) + threadIdx.x;

  // 每线程预加和相距 BLOCK_SIZE 的两个元素（越界跳过，等价补 0）。
  float val = 0.0f;
  if (gid < n) val += input[gid];
  if (gid + BLOCK_SIZE < n) val += input[gid + BLOCK_SIZE];
  smem[tid] = val;
  __syncthreads();

  // 编译期常量归约级：每级把部分和数量折半、就地落回 smem 前端连续槽（无
  // bank 冲突，同 v2 说明）；BLOCK_SIZE 已知使未命中的整级分支可被消除。
  if (BLOCK_SIZE >= 512) {
    if (tid < 256) smem[tid] += smem[tid + 256];  // 512 -> 256 个部分和
    __syncthreads();
  }
  if (BLOCK_SIZE >= 256) {
    if (tid < 128) smem[tid] += smem[tid + 128];  // 256 -> 128
    __syncthreads();
  }
  if (BLOCK_SIZE >= 128) {
    if (tid < 64) smem[tid] += smem[tid + 64];  // 128 -> 64
    __syncthreads();
  }

  // 剩余 <= 64 个部分和收进 warp 0 展开合并（volatile 保证每次读写真实落内存，
  // 免去其后所有 __syncthreads，语义与实现细节同 reduce.cu 的 warpReduce 注释）。
  if (tid < 32) {
    volatile float* vsmem = smem;
    if (BLOCK_SIZE >= 64) vsmem[tid] += vsmem[tid + 32];  // 64 -> 32
    vsmem[tid] += vsmem[tid + 16];
    vsmem[tid] += vsmem[tid + 8];
    vsmem[tid] += vsmem[tid + 4];
    vsmem[tid] += vsmem[tid + 2];
    vsmem[tid] += vsmem[tid + 1];
  }

  if (tid == 0) {
    output[blockIdx.x] = smem[0];
  }
}

// v6 见 reduce.cu：两级 warp shuffle 归约版，覆盖口径同 v3/v4/v5，要求
// blockDim.x 为 2 的幂且 32 <= blockDim.x <= 1024。
__global__ void reduce_v6(const float* input, float* output, int n);

// v7 见 reduce.cu：v6 的归约结构 + float4 向量化加载与 grid-stride 扫描 —— 每
//   线程每次读 1 个 float4（16 B，input 需 16 字节对齐），全体线程以
//   gridDim.x * blockDim.x 为步长联合覆盖输入；n % 4 的尾部元素以标量路径补齐。
//   覆盖口径与 v0…v6 的“每 block 覆盖固定 span”不同：grid-stride 保证任意
//   grid >= 1 都完整覆盖输入，故 grid 只需按性能推荐（见 reduce.cu 注释），
//   冗余 block 不读数据、写 0，仍满足“超配安全”。块内归约（两级 warp
//   shuffle）与 block 约束同 v6（2 的幂且 32 ~ 1024）。
__global__ void reduce_v7(const float* input, float* output, int n);

// ---------------------------------------------------------------------------
// 归约内核统一签名
// ---------------------------------------------------------------------------
// 各版本输出约定一致（每 block 1 个部分和），故可共用同一测试驱动。
using ReduceKernel = void (*)(const float* input, float* output, int n);

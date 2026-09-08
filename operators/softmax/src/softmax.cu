// softmax.cu —— softmax.cuh 声明的实现：CPU 参考 + softmax_v0/v1/v2 内核。
// 接口契约 / 启动约束 / 版本差异见 softmax.cuh，推导与实测见 README.md。

#include <cmath>    // expf / fmaxf / INFINITY / std::exp
#include <cstddef>  // std::size_t

#include "softmax.cuh"

// softmax_cpu —— 主机端参考（测试的正确性基线）
void softmax_cpu(const float* input, float* output, int M, int N) {
  // 逐行 softmax：求行最大 m → Σexp(x-m) → 归一化。全程 double：参考值不自带
  // fp32 舍入误差（GPU 侧还要经受 1e-5 判据），返回前转回 float。
  for (int row = 0; row < M; ++row) {
    const float* x = input + static_cast<std::size_t>(row) * N;
    float* y = output + static_cast<std::size_t>(row) * N;
    if (N <= 0) continue;  // 空行：无元素可写

    double m = -INFINITY;
    for (int c = 0; c < N; ++c) {
      m = (static_cast<double>(x[c]) > m) ? static_cast<double>(x[c]) : m;
    }

    double sum = 0.0;
    for (int c = 0; c < N; ++c) {
      sum += std::exp(static_cast<double>(x[c]) - m);
    }

    const double inv_sum = 1.0 / sum;
    for (int c = 0; c < N; ++c) {
      y[c] = static_cast<float>(std::exp(static_cast<double>(x[c]) - m) * inv_sum);
    }
  }
}

// softmax_v0 —— 每线程处理一行，行内串行三遍（正确性基线）
__global__ void softmax_v0(const float* input, float* output, const int M,
                           const int N) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;  // 线程铺满行号
  if (row >= M) return;  // 空矩阵 / 超配 grid：越界行空转

  const float* x = input + row * N;
  float* y = output + row * N;

  // ① 行最大 m（max-shift 使 exp 参数 <= 0、行和 >= 1，数值稳定）
  float max_val = -INFINITY;
  for (size_t i = 0; i < N; ++i) {
    max_val = fmaxf(max_val, x[i]);
  }

  // ② Σexp(x - m)
  float sum = 0.0f;
  for (size_t i = 0; i < N; ++i) {
    sum += expf(x[i] - max_val);
  }

  // ③ 归一化写回 y = exp(x - m) / 行和
  const float inv_sum = 1.0f / sum;
  for (size_t i = 0; i < N; ++i) {
    y[i] = expf(x[i] - max_val) * inv_sum;
  }
}

// softmax_v1 —— 每行一个 block，行内协作 + 共享内存树形归约
__global__ void softmax_v1(const float* input, float* output, const int M,
                           const int N) {
  extern __shared__ float smem[];  // 动态共享内存：blockDim.x * sizeof(float)

  const int row = blockIdx.x;  // 每 block 处理一行
  const int tid = threadIdx.x;
  if (row >= M) return;  // 空矩阵 / 超配 grid：空转

  const float* x = input + row * N;
  float* y = output + row * N;

  // ① 各线程 stride 扫行求局部最大 → 折半 fmaxf 树形归约出行最大（warp 内
  //    同轮访问相邻列 → 读合并；N 不整除 blockDim.x 时多余线程以 -inf 空转）
  float local_max = -INFINITY;
  for (int i = tid; i < N; i += blockDim.x) {
    local_max = fmaxf(local_max, x[i]);
  }
  smem[tid] = local_max;
  __syncthreads();  // 槽位全部就绪后才能开始归约
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
    }
    __syncthreads();  // 下一轮读本轮刚写入的局部最大
  }
  const float row_max = smem[0];
  __syncthreads();  // 全部线程读走 m 后，smem 才能被 ② 复用

  // ② 同样的 stride 扫行累加局部 Σexp → 树形加法归约出行和（块内分段累加把
  //    v0 整行串行的舍入误差降到 ~(每线程元素数 + log2(blockDim)) · ulp 量级）
  float local_sum = 0.0f;
  for (int i = tid; i < N; i += blockDim.x) {
    local_sum += expf(x[i] - row_max);
  }
  smem[tid] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }
  const float row_sum = smem[0];

  // ③ 再读一次行重算 exp 并归一化写回（行宽任意，smem 无需装整行）
  const float inv_sum = 1.0f / row_sum;
  for (int i = tid; i < N; i += blockDim.x) {
    y[i] = expf(x[i] - row_max) * inv_sum;
  }
}

// ---------------------------------------------------------------------------
// warpReduceMax / warpReduceSum —— 单 warp shuffle 归约（每 lane 1 个局部值 →
// 归约值收敛到 lane 0）
// ---------------------------------------------------------------------------
// 5 轮 __shfl_down_sync（offset 16 → 1）在寄存器间两两归约，不碰共享内存、无
// 同步开销；mask 0xffffffff 要求完整 warp 收敛调用（调用点不可让 warp 内部分
// 线程提前退出，否则行为未定义）。
__device__ float warpReduceMax(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

__device__ float warpReduceSum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// ---------------------------------------------------------------------------
// blockReduceMaxShuffle / blockReduceSumShuffle —— 块内归约（两级 warp shuffle，
// 结果经共享内存广播回全体线程）
// ---------------------------------------------------------------------------
// ① warpReduce{Max,Sum} 先把每 warp 归为 1 个值，lane 0 写入 warp_results[wid]；
// ② __syncthreads 后 warp 0 再归约 num_warps 个 warp 值（lane >= num_warps 以归约
//    单位元 -INFINITY / 0 空转，不影响结果），块归约值收敛到 tid 0；
// ③ 经 block_result + __syncthreads 广播：softmax 的 ②③ 遍需要每个线程都拿到
//    行最大 / 行和，不能只由 tid 0 持有（区别于 reduce 只需 tid 0 写输出）。
// 约束：blockDim.x 为 32 的倍数（默认 256）且 num_warps <= 32（即 blockDim.x
// <= 1024），warp 值才装得进 warp_results[32]。
__device__ float blockReduceMaxShuffle(float val) {
  __shared__ float warp_results[32];  // 各 warp 的归约值（每 warp 由 lane 0 写入）
  __shared__ float block_result;      // 广播用块归约值

  const int lane = threadIdx.x % 32;
  const int wid = threadIdx.x / 32;

  val = warpReduceMax(val);  // ① warp 内归约：lane 0 持该 warp 归约值
  if (lane == 0) warp_results[wid] = val;
  __syncthreads();  // 全部 warp 值就绪后才能被 warp 0 读取

  const int num_warps = blockDim.x / 32;
  val = (lane < num_warps) ? warp_results[lane] : -INFINITY;
  if (wid == 0) {
    val = warpReduceMax(val);  // ② warp 0 归约 num_warps 个值 → 块最大值
  }
  if (threadIdx.x == 0) block_result = val;  // tid 0（warp 0 lane 0）持有块结果
  __syncthreads();  // ③ 广播：全体读到块归约值后才允许继续（写回前读行）

  return block_result;
}

__device__ float blockReduceSumShuffle(float val) {
  __shared__ float warp_results[32];
  __shared__ float block_result;

  const int lane = threadIdx.x % 32;
  const int wid = threadIdx.x / 32;

  val = warpReduceSum(val);
  if (lane == 0) warp_results[wid] = val;
  __syncthreads();

  const int num_warps = blockDim.x / 32;
  val = (lane < num_warps) ? warp_results[lane] : 0.0f;
  if (wid == 0) {
    val = warpReduceSum(val);
  }
  if (threadIdx.x == 0) block_result = val;
  __syncthreads();

  return block_result;
}

// ---------------------------------------------------------------------------
// softmax_v2 —— 每行一个 block，块内两级 warp shuffle 归约
// ---------------------------------------------------------------------------
// 行遍历与 v1 相同（blockDim.x 个线程 stride 扫行 → 全局读合并；N 不足 blockDim.x
// 时多余线程空转、超出时多轮 stride），差异只在块内归约：v1 的共享内存折半树形
// 归约换成两级 warp shuffle —— 每 warp 先经 warpReduce{Max,Sum} 归为 1 个值，再由
// warp 0 归约 num_warps 个值并广播回全体（见上方 helper 注释）。归约在寄存器间
// 完成、共享内存只做跨 warp 中转，块内同步由 v1 的 ~2×(log2(blockDim.x)+2) 次
// 降为每个 blockReduce* 内部 2 次 __syncthreads。
// 启动约束同 v1：row = blockIdx.x、grid = M（M == 0 时也须 >= 1，row >= M 越界
// 空转）；额外要求 blockDim.x 为 32 的倍数（默认 256）且 <= 1024。无动态共享
// 内存（仅 helper 内静态 __shared__ 中转，与行宽无关）：smem_bytes = 0。
__global__ void softmax_v2(const float* input, float* output, const int M,
                           const int N) {
  const int row = blockIdx.x;  // 每 block 处理一行
  if (row >= M) return;  // 空矩阵 / 超配 grid：越界行空转
  const int tid = threadIdx.x;

  const float* x = input + row * N;
  float* y = output + row * N;

  // ① 行最大 m：stride 扫行局部最大 → 块内两级 warp shuffle 归约（max-shift
  //    使 exp 参数 <= 0、行和 >= 1，数值稳定）
  float local_max = -INFINITY;
  for (int i = tid; i < N; i += blockDim.x) {
    local_max = fmaxf(local_max, x[i]);
  }
  const float row_max = blockReduceMaxShuffle(local_max);

  // ② Σexp(x - m)：同样的 stride 扫行分段累加 → 块内归约（块内分段 + 归约累加
  //    把 v0 整行串行的舍入误差降到 ~(每线程元素数 + log2(blockDim))·ulp 量级）
  float local_sum = 0.0f;
  for (int i = tid; i < N; i += blockDim.x) {
    local_sum += expf(x[i] - row_max);
  }
  const float row_sum = blockReduceSumShuffle(local_sum);

  // ③ 归一化写回 y = exp(x - m) / 行和（第三次读行重算 exp，行宽任意）
  const float inv_sum = 1.0f / row_sum;
  for (int i = tid; i < N; i += blockDim.x) {
    y[i] = expf(x[i] - row_max) * inv_sum;
  }
}
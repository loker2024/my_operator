// softmax.cu —— softmax.cuh 声明的实现：CPU 参考 + softmax_v0/v1 内核。
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

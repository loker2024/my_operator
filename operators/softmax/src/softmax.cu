// ============================================================================
// softmax.cu —— softmax.cuh 声明的算子实现（CPU 参考 + GPU 内核 v0）。
// 接口契约、版本差异与启动约束见 softmax.cuh；详细推导与实测结论见
// operators/softmax/README.md。此处只保留实现侧的必要说明。
// ============================================================================

#include <cmath>    // expf / fmaxf / INFINITY / std::exp
#include <cstddef>  // std::size_t

#include "softmax.cuh"

// ---------------------------------------------------------------------------
// softmax_cpu —— 主机端参考实现
// ---------------------------------------------------------------------------
// 逐行按 softmax 公式实现：先求行最大值 m，再以 (x - m) 求 Σexp，最后归一化。
// 全程用 double：参考值不自带 fp32 舍入误差（GPU 侧还要经受 ≤1e-5 的相对误差
// 判据，参考再引入误差会吃掉容差）；返回前转回 float 便于同类型比较。
void softmax_cpu(const float* input, float* output, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* x = input + static_cast<std::size_t>(r) * cols;
    float* y = output + static_cast<std::size_t>(r) * cols;
    if (cols <= 0) continue;  // 空行：无元素可写

    double m = -INFINITY;
    for (int c = 0; c < cols; ++c) {
      m = (static_cast<double>(x[c]) > m) ? static_cast<double>(x[c]) : m;
    }

    double sum = 0.0;
    for (int c = 0; c < cols; ++c) {
      sum += std::exp(static_cast<double>(x[c]) - m);
    }

    const double inv_sum = 1.0 / sum;
    for (int c = 0; c < cols; ++c) {
      y[c] = static_cast<float>(std::exp(static_cast<double>(x[c]) - m) * inv_sum);
    }
  }
}

// ---------------------------------------------------------------------------
// softmax_v0 —— 朴素两遍规约（正确性基线）
// ---------------------------------------------------------------------------
// 每 block 处理一行（grid = rows）：线程 tid 以 stride = blockDim.x 步进整行，
// warp 内各线程同一轮访问相邻列 → 全局读合并。共享内存只存规约中间量：
//   ① 每线程维护局部最大，写 smem 后按 reduce_v2 的折半形态做 fmaxf 树形归约
//     收敛到 smem[0]，得到行最大 m；
//   ② 以 expf(x - m) 累加局部行和，同样的树形加法归约出 Σexp；
//   ③ 每线程再读行、算 expf(x - m) / 行和写回 y。
// 折半归约每轮后必须 __syncthreads（下轮读本轮刚写入的局部和）；①→② 复用同一
// 段 smem 前也要先同步，保证所有线程都已读走 m。局部和/树形归约把 Σexp 的
// fp32 舍入误差压在 ~(块内每线程元素数 + log2(block)) · ulp 量级，满足 1e-5
// 的容差口径（逐元素、串行 4096 项累加会超差，故不做单线程整行归约）。
__global__ void softmax_v0(const float* input, float* output, int rows,
                           int cols) {
  if (cols <= 0) return;  // 空行防御：避免 0 元素时的除零 / 空转
  const int row = blockIdx.x;
  if (row >= rows) return;  // 空矩阵 / 超配 grid 防御

  extern __shared__ float smem[];  // 动态共享内存：blockDim.x * sizeof(float)
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const float* x = input + static_cast<std::size_t>(row) * cols;
  float* y = output + static_cast<std::size_t>(row) * cols;

  // ① 行最大 m：strided 遍历求局部最大 → 共享内存树形 fmaxf 归约。
  float local_max = -INFINITY;
  for (int col = tid; col < cols; col += nthreads) {
    local_max = fmaxf(local_max, x[col]);
  }

  smem[tid] = local_max;
  __syncthreads();  // 槽位全部就绪后才能开始归约
  for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
    }
    __syncthreads();  // 下一轮读本轮刚写入的局部最大
  }
  const float row_max = smem[0];
  __syncthreads();  // 所有线程读走 m 后，smem 才能被第②阶段复用

  // ② Σexp(x - m)：以 m 减平移（数值稳定），strided 累加 → 同样的树形归约。
  float local_sum = 0.0f;
  for (int col = tid; col < cols; col += nthreads) {
    local_sum += expf(x[col] - row_max);
  }

  smem[tid] = local_sum;
  __syncthreads();
  for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }
  const float row_sum = smem[0];
  __syncthreads();

  // ③ 归一化写回：再读一次行以重算 exp（省去用 smem 存整行；行宽大时 smem
  // 只装规约中间量，不受行宽限制）。exp(0) = 1 使行和 >= 1，不会除零。
  const float inv_sum = 1.0f / row_sum;
  for (int col = tid; col < cols; col += nthreads) {
    y[col] = expf(x[col] - row_max) * inv_sum;
  }
}

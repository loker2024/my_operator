// ============================================================================
// reduce.cu —— reduce.cuh 声明的算子实现（CPU 参考 + GPU 内核 v0…v4、v6）。
// 接口契约、版本差异与启动约束见 reduce.cuh；详细推导与实测结论见
// operators/reduce/README.md。此处只保留实现侧的必要说明。
// reduce_v5 为模板内核，因实例化点需可见其定义，实现整体内联于 reduce.cuh。
// ============================================================================

#include "reduce.cuh"

// ---------------------------------------------------------------------------
// reduce_cpu —— 主机端参考实现
// ---------------------------------------------------------------------------
// 内部用 double 累加：fp32 顺序累加的舍入误差随项数增长（长序列可达 1e-4 量级，
// 见 docs/benchmark-methodology.md），double 可避免参考值自身引入额外误差；
// 返回前转回 float，便于与 GPU 的 fp32 结果同类型比较。
float reduce_cpu(const float* input, int n) {
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += static_cast<double>(input[i]);
  }
  return static_cast<float>(sum);
}

// ---------------------------------------------------------------------------
// reduce_v0 —— 交错寻址树形归约（正确性基线）
// ---------------------------------------------------------------------------
// 每线程搬 1 个元素到 smem[tid]（越界 gid >= n 补 0，不影响和）；随后 step 从
// 1 倍增，满足 tid % (2*step) == 0 的线程把 smem[tid+step] 并入 smem[tid]，
// log2(blockDim.x) 轮后收敛到 smem[0]，由 tid 0 写入 output[blockIdx.x]。
__global__ void reduce_v0(const float* input, float* output, int n) {
  extern __shared__ float smem[];

  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;

  smem[tid] = (gid < n) ? input[gid] : 0.0f;
  __syncthreads();  // 槽位全部就绪后才能开始归约

  for (size_t step = 1; step < blockDim.x; step *= 2) {
    if (tid % (2 * step) == 0) {
      smem[tid] += smem[tid + step];
    }
    __syncthreads();  // 下一轮读本轮刚写入的局部和，防止读到旧值
  }

  if (tid == 0) {
    output[blockIdx.x] = smem[0];
  }
}

// ---------------------------------------------------------------------------
// reduce_v1 —— 连续寻址树形归约
// ---------------------------------------------------------------------------
// 与 v0 同网格/输出模型，仅把活跃线程从“交错”改为“连续前缀”：每轮
// index = tid*2*step，index < blockDim.x 时合并 smem[index] += smem[index+step]，
// 即参与线程为连续前缀，消除 v0 的 warp 内分歧。
__global__ void reduce_v1(const float* input, float* output, int n) {
  extern __shared__ float smem[];

  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;

  smem[tid] = (gid < n) ? input[gid] : 0.0f;
  __syncthreads();

  for (size_t step = 1; step < blockDim.x; step *= 2) {
    const int index = 2 * static_cast<int>(step) * tid;
    if (index < blockDim.x) {
      smem[index] += smem[index + static_cast<int>(step)];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[blockIdx.x] = smem[0];
  }
}

// ---------------------------------------------------------------------------
// reduce_v2 —— 折半步长树形归约
// ---------------------------------------------------------------------------
// 步长方向与 v1 相反：stride 自 blockDim.x/2 每轮折半到 1，线程 tid（tid <
// stride）合并 smem[tid] 与 smem[tid+stride]。部分和就地落回数组最前端的连续
// 槽，故读写下标在活跃段内连续 → 无共享内存 bank 冲突；活跃线程同为连续前缀，
// 无 warp 内分歧。
__global__ void reduce_v2(const float* input, float* output, int n) {
  extern __shared__ float smem[];

  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;

  smem[tid] = (gid < n) ? input[gid] : 0.0f;
  __syncthreads();

  for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (static_cast<size_t>(tid) < stride) {
      smem[tid] += smem[tid + static_cast<int>(stride)];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[blockIdx.x] = smem[0];
  }
}

// ---------------------------------------------------------------------------
// reduce_v3 —— 每线程 2 元素（标量加载）
// ---------------------------------------------------------------------------
// 每 block 覆盖 2*blockDim.x 个连续元素：线程 tid 以 gid = blockIdx.x *
// (2*blockDim.x) + tid 为“段内前半”下标，寄存器预加和 gid 与 gid+blockDim.x
// 两个元素（越界跳过，等价补 0），再走 v2 的折半步长归约。两次加载在 warp 内
// 各自连续且互不依赖，可提升内存级并行。grid 口径见 reduce.cuh 的 v3 说明。
__global__ void reduce_v3(const float* input, float* output, int n) {
  extern __shared__ float smem[];

  const int tid = threadIdx.x;
  const int gid = blockIdx.x * (2 * blockDim.x) + threadIdx.x;

  float val = 0.0f;
  if (gid < n) val += input[gid];
  if (gid + blockDim.x < n) val += input[gid + blockDim.x];
  smem[tid] = val;  // smem 存“每线程 2 元素预加和”而非原始元素
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (static_cast<unsigned int>(tid) < s) {
      smem[tid] += smem[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[blockIdx.x] = smem[0];
  }
}

// ---------------------------------------------------------------------------
// warpReduce —— 单 warp 归约（smem 前端 32 个部分和 → smem[0]）
// ---------------------------------------------------------------------------
// 同一 warp 内指令按 SIMT 同步推进，展开写可省去 __syncthreads；volatile 强制
// 每次读写真实落内存，保证各线程读到的是其他线程刚写入的值（编译器不会把中间
// 结果缓存在寄存器里而错过别人的写入）。
__device__ void warpReduce(volatile float* smem, int tid) {
  smem[tid] += smem[tid + 32];
  smem[tid] += smem[tid + 16];
  smem[tid] += smem[tid + 8];
  smem[tid] += smem[tid + 4];
  smem[tid] += smem[tid + 2];
  smem[tid] += smem[tid + 1];
}

// ---------------------------------------------------------------------------
// reduce_v4 —— 每线程 2 元素 + 末 warp 展开归约
// ---------------------------------------------------------------------------
// 加载与覆盖口径同 v3，仅改归约尾部：折半做到 stride = 32 即停，剩余 5 轮
// 改由 warp 0 调 warpReduce 展开完成（见其上方注释），省去这些轮次的同步。
__global__ void reduce_v4(const float* input, float* output, int n) {
  extern __shared__ float smem[];

  const int tid = threadIdx.x;
  const int gid = blockIdx.x * (2 * blockDim.x) + threadIdx.x;

  float val = 0.0f;
  if (gid < n) val += input[gid];
  if (gid + blockDim.x < n) val += input[gid + blockDim.x];
  smem[tid] = val;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (static_cast<unsigned int>(tid) < s) {
      smem[tid] += smem[tid + s];
    }
    __syncthreads();
  }

  if (static_cast<unsigned int>(tid) < 32) {
    warpReduce(smem, tid);
  }

  if (tid == 0) {
    output[blockIdx.x] = smem[0];
  }
}

// ---------------------------------------------------------------------------
// reduce_v5 —— 模板常量化版本（实现见 reduce.cuh，含设计说明）
// ---------------------------------------------------------------------------
// v5 与 v4 算法一致，仅把 block 尺寸以模板参数常量化、让归约步骤在编译期
// 整体展开；因 -rdc=false 下跨翻译单元引用 __global__ 模板特化已被 nvcc 弃用，
// 其模板定义整体内联在 reduce.cuh 中（main.cu 以 reduce_v5<kBlock> 实例化）。

// ---------------------------------------------------------------------------
// warpReduceSum —— 单 warp shuffle 归约（每 lane 1 个局部和 → 全 lane 同值）
// ---------------------------------------------------------------------------
// 5 轮 __shfl_down_sync（offset 16 → 1）在寄存器间归约，不碰共享内存、无需同步；
// mask 0xffffffff 要求完整 warp 收敛调用，调用点不可只留单条 lane。
__device__ float warpReduceSum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// ---------------------------------------------------------------------------
// reduce_v6 —— 每线程 2 元素 + 两级 warp shuffle 归约
// ---------------------------------------------------------------------------
// 加载与覆盖口径同 v3/v4/v5（每 block 覆盖 2*blockDim.x 个连续元素），仅把块内
// 折半树归约换成 warp shuffle：warpReduceSum 先把每 warp 归为 1 个部分和（lane 0
// 写入 warp_results[wid]），__syncthreads 后 warp 0 再归约 numWarps 个部分和。
// 约束：blockDim.x 为 2 的幂且 32 <= blockDim.x <= 1024，部分和须装得进
// warp_results[32]。
__global__ void reduce_v6(const float* input, float* output, int n) {
  __shared__ float warp_results[32];  // 各 warp 的部分和

  const int tid = threadIdx.x;
  const int gid = blockIdx.x * (2 * blockDim.x) + threadIdx.x;
  const int lane = tid % 32;
  const int wid = tid / 32;

  float val = 0.0f;
  if (gid < n) val += input[gid];
  if (gid + blockDim.x < n) val += input[gid + blockDim.x];

  val = warpReduceSum(val);  // ① warp 内归约
  if (lane == 0) {
    warp_results[wid] = val;
  }

  __syncthreads();

  // ② 每 lane 取 1 个 warp 的部分和再归约一次（lane >= numWarps 视为 0）。
  // shuffle 需整 warp 参与，故仍让整个 warp 0 执行。
  const int numWarps = blockDim.x / 32;
  if (wid == 0) {
    val = (lane < numWarps) ? warp_results[lane] : 0.0f;
    val = warpReduceSum(val);
  }

  if (tid == 0) {
    output[blockIdx.x] = val;
  }
}

// ============================================================================
// reduce.cu —— 一维归约（求和）算子的实现
//
// 与 reduce.cuh 配对：本文件包含该头文件中全部函数的定义。
// 当前包含：
//   * reduce_cpu ：主机端参考实现（正确性基线）
//   * reduce_v0  ：GPU 归约内核 v0（交错寻址共享内存归约，每 block 一个部分和）
//   * reduce_v1  ：GPU 归约内核 v1（连续寻址共享内存归约，每 block 一个部分和）
//   * reduce_v2  ：GPU 归约内核 v2（折半步长共享内存归约，每 block 一个部分和）
//
// v0/v1/v2 的网格/块模型与启动约束完全相同（同一份测试驱动可直接复用），
// 区别仅在于块内树形归约的“活跃线程寻址方式 / 步长方向”，详见各函数前的注释。
//
// 本文件不包含 main()，也不包含测试代码。算子测试按职责拆分：
//   test.cuh / test.cu  可复用测试驱动（声明 + 实现）
//   main.cu             执行入口（main() 中运行测试）
// ============================================================================

#include "reduce.cuh"

// ============================================================================
// reduce_cpu —— 主机端 CPU 参考实现
// ============================================================================
float reduce_cpu(const float* input, int n) {
  // 用 double 累加：fp32 顺序累加时舍入误差随项数线性增长（长序列可达
  // 1e-4 量级，见 docs/benchmark-methodology.md），double 可把参考值误差
  // 压低到可忽略水平，避免把 GPU 的 fp32 误差和参考值自身的误差混在一起。
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += static_cast<double>(input[i]);
  }
  // 参考值返回 float，方便与 GPU 结果做同类型比较；double 累加的精度
  // 在转回 float 时仅损失约半个 float ulp，不影响 1e-3 容差判定。
  return static_cast<float>(sum);
}

// ============================================================================
// reduce_v0 —— GPU 归约内核（交错寻址，interleaved addressing）
// ============================================================================
// 一个 block 处理 blockDim.x 个连续元素，最终收敛出 1 个部分和。
// 整段数据的最终标量和还需要调用方把各 block 的部分和再汇总一次。
// ============================================================================
__global__ void reduce_v0(const float* input, float* output, int n) {
  // 动态共享内存：大小 = 启动配置的第三参数（block * sizeof(float)），
  // 每个 block 拥有自己的私有副本，block 之间不可见。
  extern __shared__ float smem[];

  const int tid = threadIdx.x;                              // 块内线程编号 [0, blockDim.x)
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;    // 全局元素下标

  // --- 阶段 1：数据加载 ------------------------------------------------
  // 每个线程把自己负责的 1 个元素搬入 smem[tid]（tid 唯一，槽位恰好写一次）。
  // 当 n 不是 blockDim.x 的整数倍时，末尾 block 的越界线程补 0：
  // 0 不改变求和结果，但可以保证下面每轮归约的读写下标都安全。
  smem[tid] = (gid < n) ? input[gid] : 0.0f;
  // 同步：所有线程都写完 smem 后才能开始归约，否则可能读到其他线程
  // 尚未写入的旧值（读到未初始化数据）。
  __syncthreads();

  // --- 阶段 2：树形归约（交错寻址） -------------------------------------
  // 每轮 step 翻倍，只有 tid % (2 * step) == 0 的线程把两个部分和合并：
  //   step=1 -> tid=0,2,4,… 合并相邻两个元素  smem[0]+smem[1]、smem[2]+smem[3] …
  //   step=2 -> tid=0,4,8,… 合并跨度 2 的局部和
  //   以此类推，log2(blockDim.x) 轮后结果收敛到 smem[0]。
  // 每轮结束后必须再次同步：下一轮读的是本轮刚写入的局部和，若不同步
  // 可能读到上一轮（甚至更早）的旧值。缺点：每轮活跃线程减半，后半段
  // 出现大量空转线程与 warp 分歧——这正是 v0 只作为正确性基线的理由。
  for (size_t step = 1; step < blockDim.x; step *= 2) {
    if (tid % (2 * step) == 0) {
      smem[tid] += smem[tid + step];
    }
    __syncthreads();
  }

  // --- 阶段 3：写回部分和 ----------------------------------------------
  // 归约结束后，整个 block 负责段的和在 smem[0]，由 tid 0 写入
  // output[blockIdx.x]；最终标量和由调用方对 output[0, grid) 二次求和。
  if (tid == 0) {
    output[blockIdx.x] = smem[0];
  }
}

// ============================================================================
// reduce_v1 —— GPU 归约内核（连续寻址，consecutive addressing）
// ============================================================================
// 与 v0 的差异：块内树形归约不再让“交错间隔”的线程活跃，而是让每轮
// 活跃线程构成**连续前缀**，从而避免 v0 那种同一 warp 内活跃/空闲线程
// 交错排布带来的线程分歧；网格划分、补 0、动态共享内存与输出约定均同 v0。
//
// 逻辑说明（以 blockDim.x = 8 为例）：
//   smem 初始： [a0 a1 a2 a3 a4 a5 a6 a7]   （每个槽为对应输入元素）
//   第 1 轮 (step=1)：index = tid*2，活跃 tid ∈ [0,4)
//     tid0: smem[0] += smem[1]  → a0+a1          tid1: smem[2] += smem[3]  → a2+a3
//     tid2: smem[4] += smem[5]  → a4+a5          tid3: smem[6] += smem[7]  → a6+a7
//     结果： [a0+a1 .. a2+a3 .. a4+a5 .. a6+a7 ..]（槽 0/2/4/6 保存局部和）
//   第 2 轮 (step=2)：index = tid*4，活跃 tid ∈ [0,2)
//     tid0: smem[0] += smem[2] → Σa0..a3        tid1: smem[4] += smem[6] → Σa4..a7
//   第 3 轮 (step=4)：index = tid*8，仅 tid0 满足 index < blockDim.x
//     tid0: smem[0] += smem[4] → Σa0..a7（收敛）
//   log2(blockDim.x) 轮后总和收敛到 smem[0]。
//
// 活跃线程形态对比（blockDim.x = 256）：
//   v0 交错寻址：active 条件为 tid % (2*step) == 0，活跃 lane 在 warp 内
//     均匀间隔分布，如第 1 轮每个 warp 只动用 0/2/4/…/30 共 16 个 lane，
//     其余 lane 空转，形成 warp 内分歧；
//   v1 连续寻址：active 条件为 index < blockDim.x，即 tid 属于连续前缀
//     [0, blockDim.x/(2*step))，故“整条 warp 要么全活跃、要么全空闲”，
//     消除了 v0 的 warp 内分歧，指令调度更规整。
//   注意：两者每轮参与线程总数相同（blockDim.x/2 递减），v1 并不减少
//   计算量，只改善线程占用形态，是“代价最小”的 v0 直接改进。
//
// 参数：
//   input  —— 设备端输入数组 input[0, n)，须为已分配的有效全局内存；
//   output —— 设备端输出数组，本 block 的部分和写入 output[blockIdx.x]，
//             故 output 至少要有 grid 个元素（调用方负责分配）；
//   n      —— 输入元素个数，须 >= 0 且 grid * blockDim.x 覆盖的范围可
//             以超出 n（越界部分由补 0 逻辑兜底，见下）。
// 返回值：无（结果写入 output[blockIdx.x]）。
//
// 启动约束（由调用方保证，与 v0 相同）：
//   * grid >= ceil(n / block)，使每个输入元素恰好被一个线程读到；若实际
//     启动的 grid 大于该值，多余的 block 因 gid >= n 全部补 0，其部分和
//     为 0，不会改变最终结果（该“超配安全”性质被测试显式覆盖）；
//   * block 应为 2 的幂（默认 256）：本内核靠 step 倍增逼近 blockDim.x，
//     只有 2 的幂才能让每轮下标不重叠、最终恰好收敛到 smem[0]；
//   * 动态共享内存 = block * sizeof(float) 字节，由启动配置第三参数给出。
// ============================================================================
__global__ void reduce_v1(const float* input, float* output, int n) {
  // 动态共享内存：每个 block 的私有副本，大小由启动配置第三参数指定。
  extern __shared__ float smem[];

  const int tid = threadIdx.x;                            // 块内线程编号
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;  // 全局元素下标

  // --- 阶段 1：数据加载（与 v0 相同） ------------------------------------
  // 每个线程搬 1 个元素到 smem[tid]；越界（gid >= n）补 0，既保证求和
  // 结果不变，也保证后续各轮归约读写不越界、不读未初始化数据。
  smem[tid] = (gid < n) ? input[gid] : 0.0f;
  // 等待全部槽位就绪后才能开始归约。
  __syncthreads();

  // --- 阶段 2：树形归约（连续寻址） --------------------------------------
  // 每轮 step 倍增（1, 2, 4, …），线程 tid 负责把槽位
  //   index     = tid * 2 * step
  //   index+step
  // 两个局部和合并到 smem[index]。活跃条件 index < blockDim.x 等价于
  // tid < blockDim.x / (2 * step)，即活跃线程构成连续前缀。合并结果写入
  // 偶数下标槽，最终收敛到 smem[0]。
  // 每轮之间必须 __syncthreads：下一轮要读本轮刚写入的局部和，不同步会
  // 读到旧值。与 v0 不同之处仅在于活跃线程从“交错”变为“连续前缀”。
  for (size_t step = 1; step < blockDim.x; step *= 2) {
    const int index = 2 * static_cast<int>(step) * tid;
    if (index < blockDim.x) {
      smem[index] += smem[index + static_cast<int>(step)];
    }
    __syncthreads();
  }

  // --- 阶段 3：写回部分和（与 v0 相同） ----------------------------------
  // tid 0 把收敛在 smem[0] 的块内和写入 output[blockIdx.x]；
  // 最终标量由调用方对 output[0, grid) 再做一次轻量求和得到。
  if (tid == 0) {
    output[blockIdx.x] = smem[0];
  }
}

// ============================================================================
// reduce_v2 —— GPU 归约内核（折半步长，stride halving）
// ============================================================================
// 与 v0/v1 的差异：块内树形归约的“步长方向”从“step 从 1 倍增逼近 blockDim.x”
// 反转为“stride 从 blockDim.x/2 逐轮折半到 1”。部分和每轮就地落回“最靠前的
// stride 个连续槽”，因此：
//   * 活跃线程仍是连续前缀 tid < stride（与 v1 一样消除 warp 内分歧）；
//   * 读写 smem 的下标在活跃前缀内是**连续地址**，天然无共享内存 bank 冲突
//     （v1 的 index = tid*2*step 使同一 warp 内地址按 2*step 间隔排布，
//     在大多数轮次会命中 2 路及以上的 bank 冲突）。
// 网格划分、补 0、动态共享内存、输出约定与启动约束均同 v0/v1。
//
// 逻辑说明（以 blockDim.x = 8 为例）：
//   smem 初始： [a0 a1 a2 a3 a4 a5 a6 a7]   （每个槽为对应输入元素）
//   第 1 轮 (stride=4)：活跃 tid ∈ [0,4)
//     tid0: smem[0]+=smem[4] → a0+a4      tid1: smem[1]+=smem[5] → a1+a5
//     tid2: smem[2]+=smem[6] → a2+a6      tid3: smem[3]+=smem[7] → a3+a7
//     结果：前 4 个槽保存两两相隔 4 个元素的局部和
//   第 2 轮 (stride=2)：活跃 tid ∈ [0,2)
//     tid0: smem[0]+=smem[2] → Σa0..a3    tid1: smem[1]+=smem[3] → Σa4..a7
//   第 3 轮 (stride=1)：仅 tid0
//     tid0: smem[0]+=smem[1] → Σa0..a7（收敛）
//   每轮都把上一轮保存在 [0, stride) 的局部和两两配对合并，结果继续留在
//   [0, stride/2)，stride 折半 log2(blockDim.x) 轮后收敛到 smem[0]。
//
// 与 v1 的对照（blockDim.x = 256）：
//   v1 连续寻址：step 从 1 倍增，第 1 轮 index = tid*2，同一 warp 内线程访问
//     的地址相差 2 个 float，跨 64 个连续槽 → 每轮存在 2 路 bank 冲突，且随
//     step 增大冲突路数上升；
//   v2 折半步长：每轮线程 tid 读 smem[tid] 与 smem[tid+stride]（均为连续段），
//     同一 warp 内地址彼此相邻 → 全程无 bank 冲突，是 v1 之外另一种“代价最小”
//     的改进：不减少计算量，只改善共享内存访问形态。
//   注意：v2 不做每线程多元素 / float4 向量化，那是 v0/v1/v2 之后规划中的
//   “寄存器多元素 + 向量加载”方向（见 reduce.cuh 的版本规划）。
//
// 参数、返回值与启动约束与 v1 完全一致，见 reduce_v1 定义处注释：
//   * grid >= ceil(n / block)（多余 block 全部补 0、部分和为 0，超配安全）；
//   * block 应为 2 的幂（默认 256）：折半归约需要每轮区间恰好一分为二，
//     非 2 的幂会在中间轮次出现无法配对/下标重叠；
//   * 动态共享内存 = block * sizeof(float) 字节。
// ============================================================================
__global__ void reduce_v2(const float* input, float* output, int n) {
  // 动态共享内存：每个 block 的私有副本，大小由启动配置第三参数指定。
  extern __shared__ float smem[];

  const int tid = threadIdx.x;                            // 块内线程编号
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;  // 全局元素下标

  // --- 阶段 1：数据加载（与 v0/v1 相同） ------------------------------------
  // 每个线程搬 1 个元素到 smem[tid]；越界（gid >= n）补 0，既保证求和
  // 结果不变，也保证后续各轮归约读写不越界、不读未初始化数据。
  smem[tid] = (gid < n) ? input[gid] : 0.0f;
  // 等待全部槽位就绪后才能开始归约。
  __syncthreads();

  // --- 阶段 2：树形归约（折半步长） ----------------------------------------
  // 与 v1 相反：stride 从 blockDim.x/2 出发每轮折半（>>= 1）直至 1。线程
  // tid 负责把槽位
  //   tid     （保存前半个区间的局部和）
  //   tid+stride（后半个区间）
  // 合并回 smem[tid]。活跃条件 tid < stride 即“连续前缀”，合并结果恰好留在
  // 数组最前端的 stride 个槽，供下一轮继续配对，因此每轮读写的都是连续地址。
  // 每轮之间必须 __syncthreads：下一轮要读本轮刚写入的局部和，不同步会读到
  // 旧值。
  for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (static_cast<size_t>(tid) < stride) {
      smem[tid] += smem[tid + static_cast<int>(stride)];
    }
    __syncthreads();
  }

  // --- 阶段 3：写回部分和（与 v0/v1 相同） ----------------------------------
  // tid 0 把收敛在 smem[0] 的块内和写入 output[blockIdx.x]；
  // 最终标量由调用方对 output[0, grid) 再做一次轻量求和得到。
  if (tid == 0) {
    output[blockIdx.x] = smem[0];
  }
}
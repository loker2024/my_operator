#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

#define CHECK(call) do { \
    cudaError_t status = (call); \
    if (status != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (false)

//============================================================================
// reduce_v0: 最朴素的归约内核（CUDA C Programming Guide 的 v0 版本）
//
// 每个 block 独立归约它负责的一段数据，把"部分和"写入 d_output[blockIdx.x]。
// 整个数组的总和还需要主机（或第二个内核）把各 block 的部分和再累加一次。
//
// 三个阶段:
//   1. 数据加载   - 每个线程把 d_input[gid] 搬进共享内存（越界补 0）
//   2. 树形归约   - 交错寻址(interleaved addressing)，每轮参与线程减半
//   3. 写回结果   - tid==0 把收敛在 shmem[0] 的部分和写入 d_output[blockIdx.x]
//
// 缺点: 每轮大量线程空转、warp 分歧严重，最后几步几乎退化为单线程串行。
//============================================================================
__global__ void reduce_v0(float* d_input, float* d_output, int n)
{
    extern __shared__ float shmem[];      // 动态共享内存，每个 block 一份私有副本

    int tid = threadIdx.x;                                // 块内局部编号 [0, blockDim.x)
    int gid = blockIdx.x * blockDim.x + threadIdx.x;      // 全局编号

    //--- 阶段 1: 数据加载 -------------------------------------------------
    // 每个线程只搬运自己那一个元素到 shmem[tid]；tid 唯一，故每个槽位恰好写一次。
    // gid < n 越界检查: 当 n 不是 blockDim.x 整数倍时，末尾线程补 0（不影响求和）。
    shmem[tid] = (gid < n) ? d_input[gid] : 0;
    __syncthreads();                     // 等所有线程写完才能开始归约

    //--- 阶段 2: 树形归约（交错寻址）-------------------------------------
    // 每轮 step 翻倍，只有 tid % (2*step) == 0 的线程活跃：
    //   step=1 -> tid=0,2,4..  把相邻两个元素合并
    //   step=2 -> tid=0,4,8..  合并跨度 2 的局部和
    //   以此类推，log2(blockDim.x) 轮后收敛到 shmem[0]。
    // 不活跃的线程只是空转，但每轮都必须 __syncthreads()，防止活跃线程
    // 读到上一轮还没写完的旧值。
    for (int step = 1; step < blockDim.x; step *= 2)
    {
        if (tid % (2 * step) == 0)
        {
            shmem[tid] += shmem[tid + step];
        }
        __syncthreads();
    }

    //--- 阶段 3: 写回部分和 ---------------------------------------------
    // 归约完成后整块数据的和收敛在 shmem[0]，由 tid==0 写入 d_output[blockIdx.x]。
    if (tid == 0)
    {
        d_output[blockIdx.x] = shmem[0];
    }
}


//============================================================================
// reduce_v1: 连续寻址的树形归约
//
// 与 reduce_v0 的计算结果相同，但把“是否活跃”的判断转换为连续的线程编号。
// 第 step 轮中，tid=0,1,... 的活跃线程分别处理连续的 index：
//   step=1: tid 0,1,2... -> index 0,2,4...，合并相邻元素
//   step=2: tid 0,1,2... -> index 0,4,8...，合并两个局部和
// 这种写法避免了 v0 中 tid % (2 * step) 的模运算；活跃线程本身也是连续的。
//
// 与 v0 一样，每个 block 只输出一个部分和；启动时必须按
// blockDim.x * sizeof(float) 为 extern __shared__ 申请动态共享内存。
//============================================================================
__global__ void reduce_v1(float* d_input, float* d_output, int n)
{
    // [] 没有静态长度：实际字节数由 kernel 启动配置的第三个参数在运行时指定。
    // 每个 block 都拥有自己的这一段共享内存，block 间不会共享 shmem。
    extern __shared__ float shmem[];

    int tid = threadIdx.x;                               // 块内线程编号 [0, blockDim.x)
    int gid = blockIdx.x * blockDim.x + tid;             // 对应的全局输入下标

    // 每个线程搬运一个元素；末尾不足一个 block 的位置填 0，保证可以照常归约。
    shmem[tid] = (gid < n) ? d_input[gid] : 0.0f;
    __syncthreads();                                     // 保证本轮读取前所有数据均已写入

    // step 是两个待合并部分和之间的距离。index = 2 * step * tid
    // 使每个活跃线程访问一对不重叠的元素：shmem[index] 和 shmem[index + step]。
    // 每轮完成后必须同步，下一轮才能读取本轮生成的局部和。
    for (int step = 1; step < blockDim.x; step *= 2)
    {
        int index = 2 * step * tid;
        if (index < blockDim.x)
        {
            shmem[index] += shmem[index + step];
        }
        __syncthreads();
    }

    // 所有元素最终收敛在 shmem[0]，每个 block 仅由一个线程写出部分和。
    if (tid == 0)
    {
        d_output[blockIdx.x] = shmem[0];
    }
}

//============================================================================
// reduce_v2: 反向步长（sequential addressing）归约
//
// 与 v0/v1 的“步长从小到大”不同，v2 从 blockDim.x / 2 开始，每轮减半。
// 因此活跃线程始终是连续的低编号线程：
//   step=128 -> tid 0~127 合并 [0,128]、[1,129] ...
//   step=64  -> tid 0~63  合并前一轮得到的两个半区部分和
// 最终由 tid 0 得到整个 block 的部分和。
//
// 这样避免了取模运算，也让活跃线程保持连续，减少控制开销并改善 warp
// 的执行组织。注意：当活跃线程数小于一个 warp 时，仍会存在非活跃 lane，
// 但它们不再像 v0 那样交错分布。
//============================================================================
__global__ void reduce_v2(float* d_input, float* d_output, int n)
{
    extern __shared__ float shmem[];  // 每个 block 私有的动态共享内存

    int tid = threadIdx.x;                        // 块内线程编号
    int gid = blockIdx.x * blockDim.x + tid;      // 全局输入下标

    // 每个线程搬运一个元素；最后一个不完整 block 的越界位置补 0。
    shmem[tid] = (gid < n) ? d_input[gid] : 0.0f;
    __syncthreads();

    // 每轮把后半段的局部和加到前半段。step 每次减半，参与线程数也减半；
    // tid < step 保证参与线程连续且读写位置不会越界。
    for (int step = blockDim.x / 2; step > 0; step /= 2)
    {
        if (tid < step)
        {
            shmem[tid] += shmem[tid + step];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_output[blockIdx.x] = shmem[0];
    }
}

//============================================================================
// reduce_v3: 每线程预归约两个元素
//
// v0~v2 中一个 block 仅处理 blockDim.x 个元素；v3 则让每个线程先在
// 寄存器中累加两个相距 blockDim.x 的元素。因此一个 block 覆盖
// 2 * blockDim.x 个输入元素，所需 grid 数量约减半，进入共享内存归约的
// 数据量也从 2 * blockDim.x 个输入压缩为 blockDim.x 个线程局部和。
//
// 启动该内核时，gridSize 必须按 ceil(n / (2 * blockSize)) 计算；不能
// 沿用 v0~v2 的 ceil(n / blockSize)，否则会启动额外 block（结果仍正确，
// 但会产生不必要开销）。
//============================================================================
__global__ void reduce_v3(float* input, float* output, int n)
{
    extern __shared__ float smem[];

    const int tid = threadIdx.x;
    // 每个 block 的起点间隔为 2 * blockDim.x，避免相邻 block 重复处理数据。
    const int gid = blockIdx.x * (blockDim.x * 2) + tid;

    // 每个线程加载 [gid] 和 [gid + blockDim.x]；尾部越界元素按 0 处理。
    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + blockDim.x < n) val += input[gid + blockDim.x];
    smem[tid] = val;
    __syncthreads();

    // 对每个线程的局部和执行与 v2 相同的反向步长共享内存归约。
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        output[blockIdx.x] = smem[0];
    }
}


//============================================================================
// warpReduce: 在一个 warp 内部（32 个线程）完成树形归约
//
// block 级共享内存归约停止时还剩 64 个部分和，分散在 shmem[0..63]；
// 只需让一个 warp 的 32 个线程先执行 +32，就能把它们折叠到 shmem[0..31]，
// 再继续完成 32 -> 16 -> 8 -> 4 -> 2 -> 1 的树形归约。
//
// 参数 volatile 至关重要：它阻止编译器把多次 shmem[tid] 读改写重排成寄存器
// 缓存（读写共享内存必须按顺序发生），否则归约会得到错误结果。
//
// 步长从 32 开始逐次减半。经典隐式同步写法依赖同一 warp 的锁步执行，
// 每一轮使用上一轮已经写好的值；最终结果收敛在 lane 0，即 shmem[0]。
//============================================================================
__device__ void warpReduce(volatile float* shmem, int tid)
{
    // 调用方只让 tid=0..31（一个 warp）进入这里，但开始时待归约的数据有
    // 64 个，即 shmem[0..63]。第一句由 32 个 lane 把后半区折叠到前半区：
    //   lane i: B[i] + B[i+32] -> C[i]，得到 C[0..31]。
    //
    // 后续每句仍由全部 32 个 lane 实际执行；“有效 lane”才会逐轮减半。
    // 例如执行 +16 时：
    //   lane 0 : C[0]  + C[16] -> shmem[0]   （有效）
    //   lane 16: C[16] + B[32] -> shmem[16]  （无效，不是 C[16]+C[32]）
    // 在经典 warp 锁步模型中，同一条指令的各 lane 先读取本轮旧值，再写回
    // 新值。因此 lane 0 读取的是 lane 16 本轮写回前的 C[16]；而 lane 16
    // 写出的无效结果之后不会再进入 shmem[0] 的依赖链，不会污染最终结果。
    // 各轮“实际执行 lane / 结果有效 lane”分别为：
    //   +32: 0..31 / 0..31    +16: 0..31 / 0..15
    //   +8 : 0..31 / 0..7     +4 : 0..31 / 0..3
    //   +2 : 0..31 / 0..1     +1 : 0..31 / 0
    // 最终只有 shmem[0] 会被调用方使用。
    //
    // volatile 禁止编译器缓存或重排这些共享内存访问。该写法依赖经典的
    // 隐式 warp 同步；现代 CUDA 更推荐用 __syncwarp() 明确同步，或改用
    // __shfl_down_sync() 在寄存器之间完成 warp 规约。
    shmem[tid] += shmem[tid + 32];   // 64 -> 32 个有效部分和   lane 0..31
    shmem[tid] += shmem[tid + 16];   // 32 -> 16               lane 0..15
    shmem[tid] += shmem[tid + 8];    // 16 -> 8                lane 0..7
    shmem[tid] += shmem[tid + 4];    // 8  -> 4                lane 0..3
    shmem[tid] += shmem[tid + 2];    // 4  -> 2                lane 0..1
    shmem[tid] += shmem[tid + 1];    // 2  -> 1，收敛到 shmem[0]lane 0
}

//============================================================================
// reduce_v4: 共享内存归约 + warp 内归约
//
// 基于 v3 的每线程双元素预归约，再把 v3 的最后几步共享内存归约替换为
// warpReduce。这是 CUDA C Programming Guide 的 v4 优化版本。
//
// 相比 v3 的改进：v3 的循环会一直降到 s=1，最后 5 轮参与线程不足一个 warp，
// 绝大多数 lane 空转且每轮都要一次 __syncthreads()（还伴随明显的 warp 分歧）。
// v4 让循环在 s>32 时提前退出，剩下 64 个部分和（shmem[0..63]）交给
// warpReduce：其第一步 (+32) 把后 32 个折进前 32 个，随后在单个 warp 内
// 做 5 步树形归约收敛到 shmem[0]——全程无需任何 __syncthreads()。
//
// 启动配置与 v3 一致：gridSize 需按 ceil(n / (2 * blockSize)) 计算。
//============================================================================
__global__ void reduce_v4(float* input, float* output, int n)
{
    extern __shared__ float shmem[];  // 每个 block 私有的动态共享内存

    int tid = threadIdx.x;                          // 块内线程编号
    int gid = blockIdx.x * (blockDim.x * 2) + tid;  // 每个 block 处理 2*blockDim.x 个元素

    // 每线程预归约两个相距 blockDim.x 的元素，越界位置按 0 处理；
    // 之后进入共享内存归约的数据量从 2*blockDim.x 压缩到 blockDim.x。
    float val = 0.0f;
    if (gid < n)               val += input[gid];
    if (gid + blockDim.x < n)  val += input[gid + blockDim.x];
    shmem[tid] = val;
    __syncthreads();            // 保证共享内存归约前所有数据均已写入

    // 与 v3 相同的反向步长共享内存归约，但循环在 s>32 时即停止：
    // 当活跃线程数降到 32 以内时，剩下的归约交给 warpReduce，避免
    // 小规模线程下的 __syncthreads() 与 warp 分歧开销。
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            shmem[tid] += shmem[tid + s];
        }
        __syncthreads();        // 仅当线程数 > 32 时才需要同步
    }

    // 最后 32 个部分和（shmem[0..31]）在一个 warp 内归约，无需同步。
    if (tid < 32)
    {
        warpReduce(shmem, tid);
    }

    // 归约结果收敛在 shmem[0]，由 tid 0 写出该 block 的部分和。
    if (tid == 0)
    {
        output[blockIdx.x] = shmem[0];
    }
}

//============================================================================
// reduce_v5: 用模板参数完全展开共享内存归约
//
// v3/v4 的 blockSize 是运行时变量，因此共享内存归约需要循环和步长判断。
// v5 把 BLOCK_SIZE 变为编译期常量，使编译器可以删除不适用的分支并把各轮
// 归约直接展开。每个线程仍先读取两个元素，所以一个 block 覆盖
// 2 * BLOCK_SIZE 个输入元素。
//
// 前三轮把共享内存中的有效部分和缩减到 64 个，最后一个 warp 再完成
// 64 -> 32 -> ... -> 1 的展开式归约。当前实现要求 BLOCK_SIZE 是
// 64、128、256 或 512；启动时的 blockDim.x 必须与模板参数完全一致。
//============================================================================
template <int BLOCK_SIZE>
__global__ void reduce_v5(float* input, float* output, int n)
{
    extern __shared__ float smem[];

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * (BLOCK_SIZE * 2) + tid;

    // 在寄存器中先合并两个输入，尾部越界位置按 0 处理。
    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + BLOCK_SIZE < n) val += input[gid + BLOCK_SIZE];
    smem[tid] = val;
    __syncthreads();

    // BLOCK_SIZE 是编译期常量，不满足的 if 会被编译器直接消除。
    // 每一轮均依赖上一轮产生的部分和，因此轮次之间仍需 block 级同步。
    if (BLOCK_SIZE >= 512) { if (tid < 256) smem[tid] += smem[tid + 256]; __syncthreads(); }
    if (BLOCK_SIZE >= 256) { if (tid < 128) smem[tid] += smem[tid + 128]; __syncthreads(); }
    if (BLOCK_SIZE >= 128) { if (tid <  64) smem[tid] += smem[tid +  64]; __syncthreads(); }

    // 剩余 64 个部分和由第一个 warp 完成；volatile 保证共享内存访问不会
    // 被编译器缓存或重排。只有 lane 0 的最终结果有效。
    if (tid < 32)
    {
        volatile float* vsmem = smem;
        if (BLOCK_SIZE >= 64) vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    if (tid == 0)
    {
        output[blockIdx.x] = smem[0];
    }
}

//============================================================================
// warpReduceSum: 使用 shuffle 指令在一个 warp 的寄存器之间求和
//
// __shfl_down_sync 直接读取同一 warp 中更高 lane 的寄存器，无需先写共享
// 内存。offset 依次为 16、8、4、2、1；最终 lane 0 持有整个 warp 的和，
// 其他 lane 的返回值只是中间结果，调用方不应使用。
//============================================================================
__device__ float warpReduceSum(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

//============================================================================
// reduce_v6: Warp Shuffle + 两级规约
//
// v5 的最后阶段仍通过共享内存交换数据；v6 改用 shuffle 在寄存器间通信：
//   1. 每个 warp 独立求和，lane 0 把结果写入 warp_results；
//   2. warp 0 读取这些结果，再执行一次 warp 内求和。
// 这样共享内存只需保存“每个 warp 一个值”，并且整个 block 只需要一次
// __syncthreads()。blockDim.x 必须是 32 的倍数且不超过 1024。
//============================================================================
__global__ void reduce_v6(float* input, float* output, int n)
{
    const int tid  = threadIdx.x;
    const int gid  = blockIdx.x * (blockDim.x * 2) + tid;
    const int lane = tid % 32;      // 线程在 warp 内的编号 [0, 31]
    const int wid  = tid / 32;      // warp 在当前 block 内的编号

    // 与 v3~v5 相同，每个线程先在寄存器中预归约两个输入元素。
    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + blockDim.x < n) val += input[gid + blockDim.x];

    // 第一级：每个 warp 在寄存器中独立规约。
    val = warpReduceSum(val);

    // 一个 block 最多有 1024 / 32 = 32 个 warp，故静态数组长度为 32。
    // 只有各 warp 的 lane 0 写入，槽位 wid 互不冲突。
    __shared__ float warp_results[32];
    if (lane == 0)
    {
        warp_results[wid] = val;
    }
    __syncthreads();                    // 保证 warp 0 读取前所有部分和已写入

    // 第二级：warp 0 的前 num_warps 个 lane 各读取一个 warp 的结果，
    // 其余 lane 补 0，随后复用相同的 shuffle 规约。
    const int num_warps = blockDim.x / 32;
    if (wid == 0)
    {
        val = (lane < num_warps) ? warp_results[lane] : 0.0f;
        val = warpReduceSum(val);
    }

    if (tid == 0)
    {
        output[blockIdx.x] = val;
    }
}

//============================================================================
// reduce_v7: float4 向量化加载 + Grid Stride Loop + Warp Shuffle
//
// v6 每线程只读取两个 float；v7 把输入视作 float4，每条向量加载指令读取
// 四个连续 float，并通过 grid-stride loop 允许固定数量的线程处理任意长度
// 输入。向量区间之后不足 4 个元素的尾部仍用标量加载，因而 n 无需是 4
// 的倍数。cudaMalloc 返回的地址满足 float4 所需的 16 字节对齐要求。
//
// block 内归约与 v6 相同：先做 warp 内 shuffle，再由 warp 0 汇总各 warp。
// blockDim.x 必须是 32 的倍数且不超过 1024。
//============================================================================
__global__ void reduce_v7(float* input, float* output, int n)
{
    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int wid  = tid / 32;

    // n4 表示可安全按 float4 访问的完整向量数量。
    const float4* input4 = reinterpret_cast<const float4*>(input);
    const int n4 = n / 4;

    float val = 0.0f;

    // 所有 block 的线程组成一个逻辑网格；每轮跨过整个网格，保证每个
    // float4 恰好由一个线程处理，同时保持相邻线程访问相邻地址。
    for (int idx = blockIdx.x * blockDim.x + tid;
         idx < n4;
         idx += gridDim.x * blockDim.x)
    {
        float4 data = input4[idx];
        val += data.x + data.y + data.z + data.w;
    }

    // 标量处理最后 n % 4 个元素；沿用相同的 grid-stride 分工，避免重复。
    const int tail_start = n4 * 4;
    for (int idx = tail_start + blockIdx.x * blockDim.x + tid;
         idx < n;
         idx += gridDim.x * blockDim.x)
    {
        val += input[idx];
    }

    // 第一级：各 warp 独立归约线程局部和。
    val = warpReduceSum(val);

    __shared__ float warp_results[32];
    if (lane == 0)
    {
        warp_results[wid] = val;
    }
    __syncthreads();

    // 第二级：warp 0 汇总所有 warp 的部分和。
    const int num_warps = blockDim.x / 32;
    if (wid == 0)
    {
        val = (lane < num_warps) ? warp_results[lane] : 0.0f;
        val = warpReduceSum(val);
    }

    if (tid == 0)
    {
        output[blockIdx.x] = val;
    }
}


// 归约内核的统一签名。传入此类型即可被 testReduceKernel 复用。
using ReduceKernel = void (*)(float* d_input, float* d_output, int n);

//----------------------------------------------------------------------------
// testReduceKernel: 验证一个归约内核的正确性，并测量其执行性能
//
// kernel     : 待测试的 __global__ 内核函数（需遵循 ReduceKernel 的参数顺序）
// kernelName : 输出时显示的内核名称
//
// 时间仅统计 GPU 内核执行：不包括 cudaMalloc、H2D/D2H 拷贝和 CPU 最终求和。
// strictBenchmark=false：一次预热 + 100 次运行，适合快速查看结果。
// strictBenchmark=true ：充分预热后进行 21 轮长时间采样，报告中位数和波动范围。
//----------------------------------------------------------------------------
void testReduceKernel(ReduceKernel kernel, const char* kernelName,
                      int n, int gridSize, int blockSize,
                      bool strictBenchmark = false)
{
    const int warmupIterations = strictBenchmark ? 1000 : 1;
    const int iterations = strictBenchmark ? 10000 : 100;
    const int benchmarkSamples = strictBenchmark ? 21 : 1;

    //--- 生成测试数据，并计算 CPU 参考值（double 累加，减小舍入误差）---
    vector<float> h_in(n);
    double h_ref = 0.0;
    for (int i = 0; i < n; ++i)
    {
        h_in[i] = (float)(i % 1000);
        h_ref += h_in[i];
    }

    //--- 分配设备内存并拷贝输入 -------------------------------------------
    float *d_in = nullptr, *d_out = nullptr;
    CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CHECK(cudaMalloc(&d_out, gridSize * sizeof(float)));
    CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    //--- 预热与性能计时 -----------------------------------------------------
    // cudaLaunchKernel 允许把内核函数作为参数传入；args 中每一项都是对应
    // 内核参数的地址。动态共享内存大小由 shmemBytes 指定。
    const dim3 grid(gridSize);
    const dim3 block(blockSize);
    const size_t shmemBytes = blockSize * sizeof(float);
    void* args[] = {&d_in, &d_out, &n};

    for (int i = 0; i < warmupIterations; ++i)
    {
        CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), grid, block,
                               args, shmemBytes));
    }
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());                       // 确保预热执行完毕

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    vector<float> sampleTimesMs;
    sampleTimesMs.reserve(benchmarkSamples);
    for (int sample = 0; sample < benchmarkSamples; ++sample)
    {
        CHECK(cudaEventRecord(start));
        for (int i = 0; i < iterations; ++i)
        {
            CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), grid, block,
                                   args, shmemBytes));
        }
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));                 // 等待本轮所有内核完成

        float totalMs = 0.0f;
        CHECK(cudaEventElapsedTime(&totalMs, start, stop));
        sampleTimesMs.push_back(totalMs / iterations);
    }
    CHECK(cudaGetLastError());

    // 非严格模式只有一个样本，故与原先的单次平均计时完全相同。
    // 严格模式使用中位数，避免偶发的频率切换或系统调度拉高平均值。
    vector<float> sortedTimesMs = sampleTimesMs;
    sort(sortedTimesMs.begin(), sortedTimesMs.end());
    const float averageMs = sortedTimesMs[benchmarkSamples / 2];
    const float totalMs = averageMs * iterations;
    const float p5Ms = sortedTimesMs[(benchmarkSamples - 1) * 5 / 100];
    const float p95Ms = sortedTimesMs[(benchmarkSamples - 1) * 95 / 100];

    // 按一次读取全部输入及写出每个 block 的部分和估算有效全局内存带宽。
    const double bytesPerRun = static_cast<double>(n + gridSize) * sizeof(float);
    const double bandwidthGBs = bytesPerRun / (averageMs * 1.0e6); // GB/s，1 GB = 10^9 bytes
    const double throughputGElements = n / (averageMs * 1.0e6);   // GElements/s

    //--- 拷回各 block 的部分和，在主机上做最后一次归约 -------------------
    // 所有版本仅产出“每个 block 一个部分和”，总和需要由主机再累加一次。
    vector<float> h_part(gridSize);
    CHECK(cudaMemcpy(h_part.data(), d_out, gridSize * sizeof(float), cudaMemcpyDeviceToHost));

    double h_gpu = 0.0;
    for (int i = 0; i < gridSize; ++i)
        h_gpu += h_part[i];

    //--- 对比并输出结果 ------------------------------------------------------
    double err = fabs(h_gpu - h_ref);
    double rel = err / (fabs(h_ref) + 1e-30);
    printf("[%s] n=%d, grid=%d, block=%d\n", kernelName, n, gridSize, blockSize);
    printf("    CPU sum = %.6f\n", h_ref);
    printf("    GPU sum = %.6f\n", h_gpu);
    printf("    error   = %.6g (relative %.2e)  %s\n", err, rel,
           rel < 1e-3 ? "PASS" : "FAIL");
    if (strictBenchmark)
    {
        printf("    benchmark = strict (%d warmups, %d samples x %d iterations)\n",
               warmupIterations, benchmarkSamples, iterations);
        printf("    time    = median %.4f ms/kernel (P5 %.4f, P95 %.4f)\n",
               averageMs, p5Ms, p95Ms);
    }
    else
    {
        printf("    time    = %.4f ms/kernel (%.4f ms / %d iterations)\n",
               averageMs, totalMs, iterations);
    }
    printf("    performance = %.2f GB/s effective bandwidth, %.2f GElements/s\n",
           bandwidthGBs, throughputGElements);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
}


int main()
{
    constexpr int n = 1 << 20;             // 测试数据量：1,048,576 个 float
    constexpr int blockSize = 256;         // 每个 block 使用 256 个线程（8 个 warp）
    constexpr bool strictBenchmark = false; // 改为 true 可启用严格性能测试

    // v0~v2 每线程读取一个元素；v3~v6 每线程读取两个元素；v7 每次向量化
    // 读取四个元素。分别计算网格大小，避免启动不会处理数据的多余 block。
    constexpr int oneElementGrid = (n + blockSize - 1) / blockSize;
    constexpr int twoElementGrid =
        (n + blockSize * 2 - 1) / (blockSize * 2);
    constexpr int fourElementGrid =
        (n + blockSize * 4 - 1) / (blockSize * 4);

    // 依次运行所有版本。统一测试函数会完成预热、计时、CPU 参考值校验，
    // 因此输出既能检查每次优化后的正确性，也便于横向比较执行时间。
    testReduceKernel(reduce_v0, "reduce_v0", n, oneElementGrid, blockSize, strictBenchmark);
    testReduceKernel(reduce_v1, "reduce_v1", n, oneElementGrid, blockSize, strictBenchmark);
    testReduceKernel(reduce_v2, "reduce_v2", n, oneElementGrid, blockSize, strictBenchmark);
    testReduceKernel(reduce_v3, "reduce_v3", n, twoElementGrid, blockSize, strictBenchmark);
    testReduceKernel(reduce_v4, "reduce_v4", n, twoElementGrid, blockSize, strictBenchmark);
    testReduceKernel(reduce_v5<blockSize>, "reduce_v5", n, twoElementGrid,
                     blockSize, strictBenchmark);
    testReduceKernel(reduce_v6, "reduce_v6", n, twoElementGrid, blockSize, strictBenchmark);
    testReduceKernel(reduce_v7, "reduce_v7", n, fourElementGrid, blockSize, strictBenchmark);
    return 0;
}

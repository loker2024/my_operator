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
    constexpr int n = 1 << 20;                            // 测试数据量 1048576
    constexpr int blockSize = 256;                         // 每 block 线程数
    constexpr int gridSize = (n + blockSize - 1) / blockSize;
    constexpr int v3GridSize = (n + blockSize * 2 - 1) / (blockSize * 2);
    constexpr bool strictBenchmark = true; // 改为 true 可启用严格性能测试
    testReduceKernel(reduce_v0, "reduce_v0", n, gridSize, blockSize, strictBenchmark);
    testReduceKernel(reduce_v1, "reduce_v1", n, gridSize, blockSize, strictBenchmark);
    testReduceKernel(reduce_v2, "reduce_v2", n, gridSize, blockSize, strictBenchmark);
    testReduceKernel(reduce_v3, "reduce_v3", n, v3GridSize, blockSize, strictBenchmark);
    return 0;
}

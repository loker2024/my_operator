# CUDA Reduction 学习笔记：从基础实现到向量化加载

本文以项目中的 `Reduce/src/reduce.cu` 为主线，面向刚开始学习 CUDA 的读者，讲解如何把一个浮点数组的求和操作从 CPU 思维改写为 GPU 并行归约，并逐步理解 `reduce_v0` 到 `reduce_v7` 的优化过程。

## 1. 这段程序要解决什么问题

假设输入数组为：

```text
[1, 2, 3, 4, 5, 6, 7, 8]
```

目标是计算：

```text
1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 = 36
```

CPU 最直接的写法是一个循环：

```cpp
float sum = 0.0f;
for (int i = 0; i < n; ++i)
{
    sum += input[i];
}
```

这段代码有明显的前后依赖：下一次加法必须等待上一次加法得到新的 `sum`。如果直接让许多 GPU 线程同时修改同一个 `sum`，就会发生数据竞争。因此，GPU 归约通常采用树形结构：

```text
第 1 轮：1+2  3+4  5+6  7+8  -> 3  7  11  15
第 2 轮：3+7  11+15          -> 10 26
第 3 轮：10+26               -> 36
```

每一轮中的多组加法互不依赖，可以并行执行。输入规模每轮减半，因此归约一个 block 的数据大约需要 `log2(blockSize)` 轮。

## 2. 程序的整体结构

`reduce.cu` 可以分成四部分：

1. `CHECK` 宏：检查 CUDA Runtime API 是否执行成功。
2. `reduce_v0`～`reduce_v7`：八个逐步优化的 GPU 内核。
3. `testReduceKernel`：准备数据、启动内核、计时并校验结果。
4. `main`：设置问题规模和启动配置，依次测试全部版本。

需要特别注意：这些 kernel 每个 block 只输出一个部分和，并没有在 GPU 上直接得到整个数组的最终和。

```text
输入数组
   │
   ├─ block 0 ─> output[0]
   ├─ block 1 ─> output[1]
   ├─ block 2 ─> output[2]
   └─ ...
                  │
                  └─ 拷回 CPU，再累加所有部分和
```

这是因为 CUDA block 之间不能使用 `__syncthreads()` 相互同步。当前示例选择在 CPU 上完成最后一步，便于把学习重点放在单个 block 内的归约。实际库或大型程序通常会继续启动一个或多个归约 kernel，直到设备端只剩一个结果。

## 3. 阅读代码前需要掌握的 CUDA 概念

### 3.1 Host 与 Device

- Host 指 CPU 及其内存。
- Device 指 GPU 及其显存。
- 普通 C++ 函数在 CPU 上运行。
- 带有 `__global__` 的 kernel 由 CPU 发起，在 GPU 上由大量线程执行。
- 带有 `__device__` 的函数只能从 GPU 代码中调用。

本程序中的 `main` 和 `testReduceKernel` 在 CPU 上运行；`reduce_v0`～`reduce_v7`、`warpReduce` 和 `warpReduceSum` 在 GPU 上运行。

### 3.2 Grid、Block、Thread 与 Warp

一次 kernel 启动会创建一个 grid，grid 中包含许多 block，每个 block 又包含许多 thread。

代码中常见的内建变量含义如下：

| 变量 | 含义 |
| --- | --- |
| `threadIdx.x` | 当前线程在 block 内的编号 |
| `blockIdx.x` | 当前 block 在 grid 内的编号 |
| `blockDim.x` | 每个 block 的线程数 |
| `gridDim.x` | grid 中的 block 数 |

对于“每线程处理一个元素”的版本，全局下标通常写成：

```cpp
int gid = blockIdx.x * blockDim.x + threadIdx.x;
```

GPU 硬件以 warp 为基本调度单位。一个 warp 在当前 NVIDIA GPU 上固定包含 32 个线程，也称 32 个 lane。程序通过下面两行得到线程在 warp 中的位置：

```cpp
int lane = threadIdx.x % 32;
int wid  = threadIdx.x / 32;
```

当 `blockSize = 256` 时，一个 block 有 `256 / 32 = 8` 个 warp。

### 3.3 常见存储层次

| 存储位置 | 本例中的对象 | 特点 |
| --- | --- | --- |
| Host 内存 | `h_in`、`h_part` | CPU 可直接访问 |
| Global Memory | `d_in`、`d_out` | 所有 GPU 线程可访问，容量大但延迟较高 |
| Shared Memory | `shmem`、`warp_results` | 同一个 block 的线程共享，速度较快 |
| Register | `val`、`tid`、`gid` | 线程私有，通常最快 |

归约优化的核心之一，就是尽量减少对 global memory 的访问，并尽量在寄存器和 shared memory 中完成中间计算。

### 3.4 动态共享内存

v0～v5 使用下面的声明：

```cpp
extern __shared__ float shmem[];
```

数组长度没有写在代码中，而是在启动 kernel 时通过共享内存字节数指定：

```cpp
const size_t shmemBytes = blockSize * sizeof(float);
```

每个 block 都会得到一份独立的 `shmem`。不同 block 不能访问彼此的共享内存。

### 3.5 为什么需要 `__syncthreads()`

`__syncthreads()` 是 block 级屏障：同一个 block 中的所有线程到达屏障后，才会继续向下执行。

例如：

```cpp
shmem[tid] = input[gid];
__syncthreads();
shmem[tid] += shmem[tid + step];
```

如果缺少同步，某个线程可能已经开始读取 `shmem[tid + step]`，而负责写入该位置的线程还没有完成，结果就会变得不确定。

不能把 `__syncthreads()` 放进只有部分线程会进入的普通分支中，否则其他线程无法到达屏障，可能导致 block 永久等待。本项目各轮归约中的同步都位于参与条件之外，所有线程都会执行。

## 4. v0：交错寻址归约

v0 是最基础的共享内存归约，可以分为三个阶段。

### 4.1 加载数据

```cpp
shmem[tid] = (gid < n) ? d_input[gid] : 0;
__syncthreads();
```

每个线程把一个输入元素从 global memory 搬到 shared memory。如果 `gid` 越界就写入 0，因为 0 不会改变求和结果。这样最后一个不完整 block 也可以使用相同的归约逻辑。

### 4.2 交错归约

```cpp
for (int step = 1; step < blockDim.x; step *= 2)
{
    if (tid % (2 * step) == 0)
    {
        shmem[tid] += shmem[tid + step];
    }
    __syncthreads();
}
```

假设 block 中有 8 个线程：

```text
step = 1：线程 0、2、4、6 工作
step = 2：线程 0、4 工作
step = 4：线程 0 工作
```

活跃线程在 warp 中交错分布。GPU 以 warp 为单位执行指令，即使一条分支中只有少数 lane 工作，整个 warp 仍要执行该分支，未参与的 lane 只能空转。这叫做 warp divergence，即 warp 分歧。

此外，取模运算 `tid % (2 * step)` 也会带来额外指令。因此 v0 适合作为正确性基线，但不是高效实现。

### 4.3 写出部分和

```cpp
if (tid == 0)
{
    d_output[blockIdx.x] = shmem[0];
}
```

每个 block 的最终结果都收敛到 `shmem[0]`，由线程 0 写入对应的输出位置。

### 4.4 项目中的完整代码

```cpp
__global__ void reduce_v0(float* d_input, float* d_output, int n)
{
    extern __shared__ float shmem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    shmem[tid] = (gid < n) ? d_input[gid] : 0;
    __syncthreads();

    for (int step = 1; step < blockDim.x; step *= 2)
    {
        if (tid % (2 * step) == 0)
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
```

这里展示的是项目中完整的 v0 kernel；启动时还要传入 `blockSize * sizeof(float)` 字节的动态共享内存。

### 4.5 优缺点与改进方向

**优点：**

- 归约过程直接对应二叉树合并，代码短，适合作为正确性基线和理解共享内存、线程同步的起点。
- 初始 global memory 读取由相邻线程访问相邻元素，可以形成合并访存；越界线程补 0，也能正确处理不完整的最后一个 block。
- block 之间彼此独立，只需各自输出一个部分和，便于使用同一测试框架验证结果。

**缺点：**

- `tid % (2 * step) == 0` 让活跃 lane 在 warp 中交错分布，分支效率低；取模和乘法也增加了地址、条件计算开销。
- 每轮只有一半线程继续工作，却仍由整个 warp 执行指令，线程利用率随着归约推进迅速下降。
- 每轮都通过 shared memory 交换中间值并执行一次 `__syncthreads()`，同步和共享内存访问开销较高。
- 每线程只读取一个元素，需要较多 block 和部分和输出；归约循环还假定 block size 是 2 的幂。

**与上一代的关系：** v0 是本项目的第一版，没有可比较的上一代。它暴露出的首要问题是“交错活跃线程 + 取模判断”，因此 v1 的改进目标是只改变线程到数据对的映射，在保持归约树不变的情况下让活跃线程连续排列。

## 5. v1：连续活跃线程

v1 没有改变归约的数学结构，主要改变“由哪个线程负责哪一对数据”。

```cpp
int index = 2 * step * tid;
if (index < blockDim.x)
{
    shmem[index] += shmem[index + step];
}
```

当 `blockDim.x = 8` 时：

```text
step = 1：tid 0、1、2、3 分别处理 [0,1]、[2,3]、[4,5]、[6,7]
step = 2：tid 0、1 分别处理 [0,2]、[4,6]
step = 4：tid 0 处理 [0,4]
```

参与工作的线程变成连续的低编号线程，避免了 v0 中交错 lane 带来的严重分歧，也删除了取模判断。

不过，真正访问的 shared memory 下标仍然是逐渐变大的跨距，并且每一轮仍需同步。

### 5.1 项目中的完整代码

```cpp
__global__ void reduce_v1(float* d_input, float* d_output, int n)
{
    extern __shared__ float shmem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    shmem[tid] = (gid < n) ? d_input[gid] : 0.0f;
    __syncthreads();

    for (int step = 1; step < blockDim.x; step *= 2)
    {
        int index = 2 * step * tid;
        if (index < blockDim.x)
        {
            shmem[index] += shmem[index + step];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_output[blockIdx.x] = shmem[0];
    }
}
```

### 5.2 优缺点与相对 v0 的改进

**优点：**

- 活跃线程集中在连续的低编号 lane 中，同一 warp 内不再出现 v0 那种奇偶交错的执行模式。
- 用一次范围判断代替取模判断，控制逻辑更简单，同时保留 v0 的合并加载、越界补 0 和通用测试方式。
- 数学归约树没有改变，便于把性能差异归因到线程组织方式，而不是算法结果。

**缺点：**

- `index = 2 * step * tid` 使一个 warp 访问带跨距的 shared memory 地址，早期轮次可能产生典型的 shared-memory bank conflict。
- 每轮仍需计算乘法和索引，仍有 `log2(blockSize)` 次 block 级同步。
- 活跃线程仍然每轮减半，后半段大量线程空转；每线程也仍然只处理一个输入。

**针对 v0 缺点的改进思路：** v1 不改变“哪些元素成对相加”，而是把“由哪个线程执行加法”重新编号。原来由 `tid = 0, 2, 4, ...` 执行的任务，改由 `tid = 0, 1, 2, ...` 执行，再通过 `index` 找到对应数据。这样解决了 v0 的交错分歧和取模开销，但把问题转移成了跨距 shared memory 访问；v2 将进一步改数据布局和步长方向。

## 6. v2：反向步长归约

v2 从数组的中间开始，把后半部分加到前半部分：

```cpp
for (int step = blockDim.x / 2; step > 0; step /= 2)
{
    if (tid < step)
    {
        shmem[tid] += shmem[tid + step];
    }
    __syncthreads();
}
```

当 `blockDim.x = 8` 时：

```text
step = 4：线程 0~3 处理 [0,4]、[1,5]、[2,6]、[3,7]
step = 2：线程 0~1 处理 [0,2]、[1,3]
step = 1：线程 0 处理 [0,1]
```

这种写法称为 sequential addressing。它有几个优点：

- 活跃线程总是连续的低编号线程。
- 判断条件简单，只需要 `tid < step`。
- shared memory 访问模式更直接。
- 循环的含义接近标准二叉树归约，更容易继续优化。

需要理解的是，v2 并没有减少输入读取量，也没有减少同步轮数；它主要改善线程组织和寻址方式。

### 6.1 项目中的完整代码

```cpp
__global__ void reduce_v2(float* d_input, float* d_output, int n)
{
    extern __shared__ float shmem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    shmem[tid] = (gid < n) ? d_input[gid] : 0.0f;
    __syncthreads();

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
```

### 6.2 优缺点与相对 v1 的改进

**优点：**

- 活跃线程直接访问 `shmem[tid]` 和 `shmem[tid + step]`，低编号线程和低地址连续对应，寻址比 v1 更直观。
- 避免 v1 中 `2 * step * tid` 形成的典型倍增跨距和 bank-conflict 模式，也省去了每轮的乘法索引计算。
- 判断条件始终是简单的 `tid < step`，代码接近标准的二叉树归约，便于继续做加载合并和循环展开。

**缺点：**

- 与 v1 一样，每线程只读一个 global memory 元素，block 数、输入加载指令数和部分和数量都没有减少。
- 全部中间结果仍放在 shared memory 中，每轮仍有一次 `__syncthreads()`；后几轮仅少数 lane 工作。
- 正确性依赖 block size 为 2 的幂，且 kernel 只生成 block 级部分和，最终结果仍要再次归约。

**针对 v1 缺点的改进思路：** v2 放弃“步长从小到大、数据下标不断拉开”的组织方式，先把数组后半段加到前半段，再逐轮缩小 `step`。线程编号和 shared memory 下标因此保持连续，解决了 v1 的复杂索引和跨距访问问题。不过同步轮数和输入工作量没有变化，所以 v3 会把一部分归约提前到寄存器和加载阶段。

## 7. v3：每个线程先处理两个元素

v0～v2 中，每个线程只从 global memory 读取一个元素。v3 让每个线程先在寄存器中累加两个元素：

```cpp
int gid = blockIdx.x * (blockDim.x * 2) + tid;

float val = 0.0f;
if (gid < n)              val += input[gid];
if (gid + blockDim.x < n) val += input[gid + blockDim.x];
smem[tid] = val;
```

当 `blockSize = 256` 时：

- v0～v2 的一个 block 覆盖 256 个输入。
- v3 的一个 block 覆盖 512 个输入。
- grid 中所需的 block 数大约减少一半。
- 两个输入先在寄存器中相加，写入 shared memory 的中间值数量也减半。

因此 v3 的网格大小必须按下式计算：

```cpp
gridSize = ceil(n / (2 * blockSize));
```

整数运算中常用下面的写法完成向上取整：

```cpp
(n + 2 * blockSize - 1) / (2 * blockSize)
```

这一版本体现了 GPU 优化中的常见思想：让每个线程做稍多的工作，用较少的 block、线程调度和同步完成相同的总任务。

### 7.1 项目中的完整代码

```cpp
__global__ void reduce_v3(float* input, float* output, int n)
{
    extern __shared__ float smem[];

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * (blockDim.x * 2) + tid;

    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + blockDim.x < n) val += input[gid + blockDim.x];
    smem[tid] = val;
    __syncthreads();

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
```

v3 的 kernel 启动网格必须使用 `ceil(n / (2 * blockSize))`，否则不能保证每个输入只被处理一次且全部覆盖。

### 7.2 优缺点与相对 v2 的改进

**优点：**

- 每线程先在寄存器中合并两个输入，一个 block 的覆盖范围翻倍，grid size 和部分和输出数量约减半。
- 两批 global memory 读取各自仍由相邻线程访问相邻地址，可以保持合并访存。
- 进入 shared memory 前就把 `2 × blockSize` 个输入压缩为 `blockSize` 个局部和，相同输入规模下减少了 block 调度和跨 block 后处理开销。
- 相比每线程只做一次加载，增加了单线程有效工作量，有助于摊薄索引、线程调度和同步成本。

**缺点：**

- block 内仍完整沿用 v2 的 shared-memory 归约，仍需多轮 `__syncthreads()`，最后一个 warp 中的同步尤其浪费。
- 每线程固定只处理两个元素，缺少 grid-stride loop，网格配置仍与输入规模直接绑定。
- 需要两次边界判断；block size 仍需为 2 的幂。若问题很小或 block 数过少，grid 减半也可能降低可用并行度。

**针对 v2 缺点的改进思路：** v3 将 v2 原本交给两个线程、两个 shared-memory 槽位的输入，先交给一个线程在寄存器中相加，再只写一个局部和。这样直接缓解 v2 的线程数多、block 数多和 shared memory 输入量大的问题，但没有改变 block 内归约阶段；v4 因而把目标转向最后一个 warp 的同步开销。

## 8. v4：最后一个 warp 内手工展开

v3 的归约循环一直执行到 `step = 1`。当 `step` 小于 32 后，只有一个 warp 中的部分 lane 还在工作，但每轮仍执行一次 `__syncthreads()`。

v4 只让普通共享内存循环执行到 `s > 32`：

```cpp
for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
{
    if (tid < s)
    {
        shmem[tid] += shmem[tid + s];
    }
    __syncthreads();
}
```

剩余 64 个部分和交给第一个 warp：

```cpp
if (tid < 32)
{
    warpReduce(shmem, tid);
}
```

`warpReduce` 把循环手工展开为固定的六次加法：

```text
+32 -> +16 -> +8 -> +4 -> +2 -> +1
```

参数使用 `volatile float*`，目的是强制编译器按代码顺序访问 shared memory，避免把中间值长期保存在寄存器里。

这是经典 CUDA 归约教程中的优化方法。需要知道的是，较新的 NVIDIA 架构支持 independent thread scheduling，不能在所有场景中仅依赖“一个 warp 天然锁步”作为线程间内存同步保证。生产代码更适合使用 `__syncwarp()` 明确同步，或直接采用 v6 的 shuffle 方案。当前代码把 v4 保留为理解经典优化路径的教学版本。

### 8.1 项目中的完整代码

v4 依赖设备辅助函数 `warpReduce`，因此完整展示必须同时包含辅助函数和 kernel：

```cpp
__device__ void warpReduce(volatile float* shmem, int tid)
{
    shmem[tid] += shmem[tid + 32];
    shmem[tid] += shmem[tid + 16];
    shmem[tid] += shmem[tid + 8];
    shmem[tid] += shmem[tid + 4];
    shmem[tid] += shmem[tid + 2];
    shmem[tid] += shmem[tid + 1];
}

__global__ void reduce_v4(float* input, float* output, int n)
{
    extern __shared__ float shmem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * 2) + tid;

    float val = 0.0f;
    if (gid < n)               val += input[gid];
    if (gid + blockDim.x < n)  val += input[gid + blockDim.x];
    shmem[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            shmem[tid] += shmem[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32)
    {
        warpReduce(shmem, tid);
    }

    if (tid == 0)
    {
        output[blockIdx.x] = shmem[0];
    }
}
```

### 8.2 优缺点与相对 v3 的改进

**优点：**

- 归约到 64 个部分和后，固定的 `+32、+16、...、+1` 代替循环，减少循环控制和地址更新指令。
- 最后六轮不再执行 block 级 `__syncthreads()`，显著削减只剩一个 warp 工作时的高成本同步。
- 保留 v3 的双元素预归约，因此同时具备较少 block 和较低末段同步开销。

**缺点：**

- 依赖 `volatile` shared memory 和经典的隐式 warp 锁步假设；在支持 independent thread scheduling 的现代架构上，这不是稳健的通用同步写法。
- `warpReduce` 把 warp size 32 和各个偏移硬编码进函数，可移植性和可维护性较弱，并要求 block 至少有 64 个线程。
- 大于一个 warp 的阶段仍使用 shared memory 和 `__syncthreads()`；block size 也仍需为受支持的 2 的幂。

**针对 v3 缺点的改进思路：** v4 观察到当 `step <= 32` 时，参与者已经局限在第一个 warp，却仍在支付整个 block 的屏障成本，因此让普通循环在 `s > 32` 时停止，再手工完成余下步骤。它解决的是 v3 的“末段同步和循环开销”，但以架构相关的 warp 同步假设为代价；v5 先继续消除前半段的运行时循环，v6 再从根本上替换这种通信方式。

## 9. v5：模板参数与完全展开

v5 把 block 大小变成模板参数：

```cpp
template <int BLOCK_SIZE>
__global__ void reduce_v5(...)
```

`main` 中实例化的是：

```cpp
reduce_v5<blockSize>
```

此时 `blockSize` 是编译期常量 256。编译器在生成机器代码时已经知道每个 block 有多少线程，因此可以：

- 删除永远不会执行的分支。
- 直接展开各轮归约。
- 减少循环控制和运行时判断。

例如 `BLOCK_SIZE = 256` 时：

```cpp
if (BLOCK_SIZE >= 512) { ... } // 编译时删除
if (BLOCK_SIZE >= 256) { ... } // 保留
if (BLOCK_SIZE >= 128) { ... } // 保留
```

v5 的代价是通用性下降。当前实现只适合 `BLOCK_SIZE` 为 64、128、256 或 512，并且实际启动时的 `blockDim.x` 必须和模板参数一致。如果模板写成 256，却用 128 个线程启动，索引和同步逻辑都会失去原本含义。

v5 的最后一个 warp 仍使用基于 shared memory 和 `volatile` 的手工展开，因此也具有 v4 所述的现代架构同步注意事项。

### 9.1 项目中的完整代码

```cpp
template <int BLOCK_SIZE>
__global__ void reduce_v5(float* input, float* output, int n)
{
    extern __shared__ float smem[];

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * (BLOCK_SIZE * 2) + tid;

    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + BLOCK_SIZE < n) val += input[gid + BLOCK_SIZE];
    smem[tid] = val;
    __syncthreads();

    if (BLOCK_SIZE >= 512) { if (tid < 256) smem[tid] += smem[tid + 256]; __syncthreads(); }
    if (BLOCK_SIZE >= 256) { if (tid < 128) smem[tid] += smem[tid + 128]; __syncthreads(); }
    if (BLOCK_SIZE >= 128) { if (tid <  64) smem[tid] += smem[tid +  64]; __syncthreads(); }

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
```

项目在 `main` 中以 `reduce_v5<blockSize>` 取得具体 kernel；模板值、实际 block size 和动态共享内存容量必须一致。

### 9.2 优缺点与相对 v4 的改进

**优点：**

- `BLOCK_SIZE` 在编译期已知，编译器可以删除不适用分支、常量折叠偏移，并将大于一个 warp 的归约轮次完全展开。
- 消除了 v4 前半段循环的步长更新、循环比较和回跳指令，控制流更固定，也更利于编译器调度指令。
- 保留双元素加载与末段展开，形成“加载预归约 + 编译期 block 归约”的完整静态路径。

**缺点：**

- 通用性下降：只支持代码列出的 block size，模板参数必须与实际 `blockDim.x` 一致，否则可能得到错误结果。
- 每种 block size 都要生成一个 kernel 实例，会增加编译时间和二进制代码体积；运行时不能任意选择未经实例化的大小。
- 轮次之间仍需 block 屏障，末段仍依赖 v4 的 `volatile` shared memory 和隐式 warp 同步假设。

**针对 v4 缺点的改进思路：** v4 只展开了最后一个 warp，前半段仍用运行时循环。v5 把 block size 提升为模板常量，用显式的编译期条件列出各轮，使编译器消除循环管理开销。这个办法换来了更高的静态优化空间，但没有解决 v4 的 shared-memory 通信与现代 warp 同步风险；这正是 v6 使用 shuffle 的原因。

## 10. `warpReduceSum`：使用 Shuffle 指令

v6 和 v7 不再让一个 warp 通过 shared memory 交换局部和，而是使用：

```cpp
__shfl_down_sync(mask, val, offset)
```

它可以让当前 lane 直接取得同一个 warp 中编号更高的 lane 的寄存器值。

```cpp
for (int offset = 16; offset > 0; offset >>= 1)
{
    val += __shfl_down_sync(0xffffffff, val, offset);
}
```

以 32 个 lane 为例：

```text
offset 16：lane 0 读取 lane 16，lane 1 读取 lane 17，...
offset  8：lane 0 读取 lane 8，...
offset  4
offset  2
offset  1：lane 0 读取 lane 1
```

最后只有 lane 0 保存整个 warp 的完整结果。其他 lane 中只是不同阶段的部分和。

掩码 `0xffffffff` 表示 32 个 lane 全部参与。v6/v7 会让整个 warp 都调用 `warpReduceSum`，即使某个线程没有有效输入，它也会带着 `0.0f` 参与，所以这里使用完整掩码是成立的。如果以后把调用放进只有部分 lane 进入的分支，就必须根据实际活跃线程构造掩码，不能盲目沿用 `0xffffffff`。

## 11. v6：Warp 内与 Warp 间两级归约

v6 的 block 内归约分成两级。

### 11.1 第一级：每个 warp 独立归约

```cpp
val = warpReduceSum(val);
```

执行后，每个 warp 的 lane 0 保存该 warp 的局部和。

### 11.2 保存每个 warp 的结果

```cpp
__shared__ float warp_results[32];
if (lane == 0)
{
    warp_results[wid] = val;
}
__syncthreads();
```

一个 block 最多有 1024 个线程，也就是最多 32 个 warp，因此数组长度设置为 32。shared memory 不再保存每个线程的值，只保存每个 warp 的一个值。

### 11.3 第二级：由 warp 0 汇总

```cpp
int num_warps = blockDim.x / 32;
if (wid == 0)
{
    val = (lane < num_warps) ? warp_results[lane] : 0.0f;
    val = warpReduceSum(val);
}
```

以 256 个线程为例，block 中有 8 个 warp。warp 0 的 lane 0～7 读取 8 个部分和，lane 8～31 读取 0，然后再执行一次 shuffle 归约。

### 11.4 项目中的完整代码

v6 依赖 `warpReduceSum`，下面将辅助函数和 kernel 一并完整列出：

```cpp
__device__ float warpReduceSum(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduce_v6(float* input, float* output, int n)
{
    const int tid  = threadIdx.x;
    const int gid  = blockIdx.x * (blockDim.x * 2) + tid;
    const int lane = tid % 32;
    const int wid  = tid / 32;

    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + blockDim.x < n) val += input[gid + blockDim.x];

    val = warpReduceSum(val);

    __shared__ float warp_results[32];
    if (lane == 0)
    {
        warp_results[wid] = val;
    }
    __syncthreads();

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
```

与早期版本相比，v6 的特点是：

- 大部分中间结果保留在寄存器中。
- shared memory 只保存每个 warp 的结果。
- 整个 block 只需要一次 `__syncthreads()`。
- 不再依赖 shared memory 中的 warp 隐式锁步归约。

当前实现要求 `blockDim.x` 是 32 的倍数且不超过 1024。否则 `num_warps = blockDim.x / 32` 会忽略不完整 warp，完整掩码的 shuffle 假设也不再成立。

### 11.5 优缺点与相对 v5 的改进

**优点：**

- warp 内通过 shuffle 直接交换寄存器值，不再为每一级中间结果反复读写 shared memory。
- shared memory 从每线程一个槽位降为每 warp 一个槽位；整个 block 归约只保留一次跨 warp 的 `__syncthreads()`。
- 不再依赖 v4/v5 的 `volatile` shared memory 和隐式 warp 锁步技巧，代码更贴近现代 CUDA 的 warp 原语。
- block size 回到运行时参数，不再需要为每个大小实例化模板 kernel。

**缺点：**

- 当前完整掩码 `0xffffffff` 和 `num_warps = blockDim.x / 32` 要求 block 是完整 warp 的整数倍，且最多 1024 个线程。
- 每线程仍固定读取两个输入，没有解决网格必须随输入规模变化的问题，也没有继续提高加载阶段的单线程工作量。
- block 之间仍不能直接交换结果，仍会写出多个部分和并由 CPU 完成最终求和。
- shuffle 降低的是归约通信开销；若 kernel 已主要受 global memory 带宽限制，继续优化归约部分的收益会变小。

**针对 v5 缺点的改进思路：** v6 不再继续堆叠模板和 shared-memory 展开，而是改变线程间通信介质：warp 内数据留在寄存器中，用 shuffle 汇总；只有各 warp 的 lane 0 把一个结果写入 shared memory。这样同时解决 v5 的模板特化限制、shared memory 访问量大、同步多以及末段隐式同步风险。剩余瓶颈主要转向输入加载与固定的每线程工作量，交给 v7 处理。

## 12. v7：`float4` 向量化加载与 Grid-Stride Loop

v7 继续使用 v6 的两级 shuffle 归约，但改变了读取输入的方式。

### 12.1 使用 `float4` 一次读取四个 float

```cpp
const float4* input4 = reinterpret_cast<const float4*>(input);
const int n4 = n / 4;
```

`float4` 是 CUDA 提供的四分量向量类型，包含 `x`、`y`、`z`、`w` 四个 `float`：

```cpp
float4 data = input4[idx];
val += data.x + data.y + data.z + data.w;
```

这里把原来的 `float*` 重新解释为 `float4*`。`cudaMalloc` 返回的设备地址具有足够的对齐，因此数组起始地址满足 `float4` 的 16 字节对齐要求。

向量化的主要意义不是减少需要读取的总字节数，而是让每个线程用更宽的内存操作获取连续数据，并减少地址计算、指令和线程调度开销。实际性能提升大小取决于 GPU 架构、编译器生成的指令和内存访问模式。

### 12.2 Grid-Stride Loop

```cpp
for (int idx = blockIdx.x * blockDim.x + tid;
     idx < n4;
     idx += gridDim.x * blockDim.x)
{
    ...
}
```

第一次迭代中，每个线程处理自己的全局下标。下一次迭代时，线程向前跨越整个 grid 的线程总数。

假设 grid 中一共有 1024 个线程：

```text
线程 0 处理 0、1024、2048、...
线程 1 处理 1、1025、2049、...
...
```

这样即使 grid 中线程数少于输入向量数，也能覆盖全部输入。相邻线程在同一轮访问相邻的 `float4`，有利于形成合并的 global memory 访问。

### 12.3 处理不足四个元素的尾部

当 `n` 不是 4 的倍数时，最后 1～3 个 float 不能按 `float4` 读取：

```cpp
int tail_start = (n / 4) * 4;
```

代码使用另一个标量 grid-stride loop 从 `tail_start` 开始处理这些元素，因此不会越界，也不会遗漏。

### 12.4 v7 的网格大小

`main` 使用：

```cpp
fourElementGrid = ceil(n / (4 * blockSize));
```

在当前 `n = 1 << 20`、`blockSize = 256` 时：

```text
fourElementGrid = 1048576 / (4 × 256) = 1024
```

由于 v7 有 grid-stride loop，网格也可以设置得更小，让每个线程多迭代几次。实际项目中常根据 SM 数量、占用率和测试结果选择 grid 大小，而不一定严格做到“一轮覆盖全部数据”。

### 12.5 项目中的完整代码

v7 复用 v6 的 `warpReduceSum`。为了让本节代码可以独立阅读，下面再次完整列出辅助函数和 kernel：

```cpp
__device__ float warpReduceSum(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduce_v7(float* input, float* output, int n)
{
    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int wid  = tid / 32;

    const float4* input4 = reinterpret_cast<const float4*>(input);
    const int n4 = n / 4;

    float val = 0.0f;

    for (int idx = blockIdx.x * blockDim.x + tid;
         idx < n4;
         idx += gridDim.x * blockDim.x)
    {
        float4 data = input4[idx];
        val += data.x + data.y + data.z + data.w;
    }

    const int tail_start = n4 * 4;
    for (int idx = tail_start + blockIdx.x * blockDim.x + tid;
         idx < n;
         idx += gridDim.x * blockDim.x)
    {
        val += input[idx];
    }

    val = warpReduceSum(val);

    __shared__ float warp_results[32];
    if (lane == 0)
    {
        warp_results[wid] = val;
    }
    __syncthreads();

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
```

### 12.6 优缺点与相对 v6 的改进

**优点：**

- 每次处理一个 `float4`，把每线程基础工作量从两个 float 提高到四个，并减少标量加载、地址计算和线程调度指令。
- 相邻线程读取相邻 `float4`，保持连续、合并的 global memory 访问；尾部标量循环保证任意 `n` 都不遗漏、不越界。
- grid-stride loop 将“输入长度”和“必须启动多少线程”解耦，可以按 GPU 的 SM 数、占用率和实测结果复用较小网格。
- 继续使用 v6 的两级 shuffle，加载和 block 内归约两端都采用了更低开销的实现。

**缺点：**

- `float4` 指针需要 16 字节对齐；当前 `cudaMalloc` 起始地址满足要求，但若以后传入带偏移的子数组指针就必须重新检查对齐。
- 向量类型不保证在所有架构和编译结果中都带来同等加速；总读取字节数没有减少，kernel 仍可能受显存带宽限制。
- 每线程串行累加更多元素可能提高寄存器压力和指令依赖，网格过小还可能降低并行度，因此 grid size 需要实测调优。
- 仍继承 v6 对完整 warp 的要求，并只输出 block 部分和；没有完成全 GPU 多级归约或处理浮点加法顺序造成的数值差异。

**针对 v6 缺点的改进思路：** v6 已大幅降低 block 内通信成本，但每线程只加载两个标量，网格仍按固定覆盖量配置。v7 将优化重点移到 global memory 输入阶段：用宽加载减少指令，用 grid-stride loop 让线程反复处理多个向量，从而提高单线程工作量并允许更灵活的网格。它改善的是加载和调度效率，而不是减少必需的内存字节数，因此性能结论仍要依靠基准测试。

## 13. 八个版本的演进总结

| 版本 | 一个 block 覆盖的输入数 | 针对上一代的主要改进 | 核心优点 | 仍未解决的主要问题 |
| --- | ---: | --- | --- | --- |
| v0 | `blockSize` | 第一版，建立基线 | 结构直接、合并加载、边界安全 | 交错分歧、取模、每轮同步、线程工作量低 |
| v1 | `blockSize` | 连续线程承接 v0 的交错任务 | 去掉取模，显著减轻分歧 | 跨距 shared memory 访问、同步轮数不变 |
| v2 | `blockSize` | 反向步长替代 v1 的倍增索引 | 连续寻址、控制简单、避免典型 bank conflict | 每线程只读一个元素，shared memory 和同步仍多 |
| v3 | `2 × blockSize` | 在寄存器中预先合并 v2 的两个输入 | block 和部分和约减半，摊薄调度开销 | block 内归约未变，末段同步浪费 |
| v4 | `2 × blockSize` | 展开 v3 的最后一个 warp | 删除末六轮 block 屏障和循环控制 | 依赖 `volatile` shared memory 与隐式 warp 同步 |
| v5 | `2 × BLOCK_SIZE` | 编译期展开 v4 的前半段循环 | 常量折叠、删除循环和无效分支 | 模板规格受限，仍有 shared memory 和同步风险 |
| v6 | `2 × blockSize` | shuffle 替代 v5 的 warp 内 shared-memory 通信 | 每 warp 一个共享值、全 block 仅一次屏障 | 要求完整 warp，每线程固定两个输入 |
| v7 | 至少 `4 × blockSize`，可循环处理更多 | 宽加载和 grid-stride 改善 v6 的输入阶段 | 加载指令更少、网格灵活、工作量更高 | 对齐与调参要求，仍受显存带宽和最终归约限制 |

优化不是简单地“版本号越大就必然越快”。最终性能还受到以下因素影响：

- GPU 型号和架构。
- block 大小与 occupancy（占用率）。
- kernel 启动开销。
- shared memory 和寄存器使用量。
- GPU 当前频率、温度以及系统中其他任务。
- 编译模式和编译器优化结果。

因此必须在保证结果正确的前提下进行实际测量。

## 14. `testReduceKernel` 测试框架详解

### 14.1 函数指针统一不同 kernel

所有归约 kernel 都使用相同参数：

```cpp
float* d_input, float* d_output, int n
```

因此代码定义统一函数指针类型：

```cpp
using ReduceKernel = void (*)(float*, float*, int);
```

`testReduceKernel` 可以接收不同 kernel，而不需要为每个版本复制一整套测试代码。模板 kernel v5 在传入前通过 `reduce_v5<blockSize>` 得到一个具体实例。

### 14.2 准备 CPU 参考结果

```cpp
vector<float> h_in(n);
double h_ref = 0.0;
```

输入元素是 `float`，但 CPU 使用 `double` 累加，目的是降低参考结果自身的舍入误差。测试数据 `i % 1000` 都是可由 float 精确表示的小整数，也便于复现实验。

### 14.3 分配和传输设备内存

```cpp
cudaMalloc(&d_in, n * sizeof(float));
cudaMalloc(&d_out, gridSize * sizeof(float));
cudaMemcpy(d_in, h_in.data(), ..., cudaMemcpyHostToDevice);
```

输出数组需要 `gridSize` 个元素，因为每个 block 会写出一个部分和。

### 14.4 `cudaLaunchKernel` 的参数形式

通常教材使用三尖括号启动语法：

```cpp
kernel<<<grid, block, shmemBytes>>>(d_in, d_out, n);
```

本项目为了通过函数指针复用测试逻辑，使用 Runtime API：

```cpp
void* args[] = {&d_in, &d_out, &n};
cudaLaunchKernel(kernel, grid, block, args, shmemBytes);
```

`args` 中保存的是每个 kernel 参数自身的地址，而不是直接把 `d_in`、`d_out` 和 `n` 转换为 `void*`。

测试函数统一传入 `blockSize * sizeof(float)` 字节的动态共享内存。v0～v5 会使用它；v6/v7 使用静态的 `warp_results`，因而这部分动态共享内存虽然被分配但没有被访问。

### 14.5 为什么需要预热

第一次 kernel 执行可能包含 CUDA 上下文初始化、缓存尚未建立、GPU 频率尚未稳定等额外成本。预热之后再计时，能让结果更接近稳定执行状态。

普通模式执行一次预热并计时 100 次；严格模式使用更多预热和 21 组长时间采样。

### 14.6 CUDA Event 计时

GPU 工作相对 CPU 是异步的，普通 CPU 时钟不能简单包围一次 kernel 启动得到可靠的 GPU 执行时间。程序使用 CUDA Event：

```cpp
cudaEventRecord(start);
// 多次启动 kernel
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&totalMs, start, stop);
```

计时范围只包含 kernel 执行，不包含：

- `cudaMalloc`。
- Host 到 Device 的输入拷贝。
- Device 到 Host 的结果拷贝。
- CPU 上对部分和的最终累加。

因此输出时间适合比较各个 kernel，本身不等于整个应用端到端耗时。

### 14.7 中位数、P5 与 P95

严格模式把 21 个样本排序：

- 中位数表示处于中间位置的样本，对偶发抖动不敏感。
- P5 表示偏快一侧的时间水平。
- P95 表示偏慢一侧的时间水平。
- P5 与 P95 相差越大，说明测量波动越明显。

### 14.8 有效带宽与吞吐率

代码使用下面的近似字节数：

```cpp
bytesPerRun = (n + gridSize) * sizeof(float);
```

它表示读取 `n` 个输入并写出 `gridSize` 个部分和。有效带宽计算为：

```text
有效带宽 = 每次处理的字节数 / 每次 kernel 时间
```

这里的 GB 按十进制 `10^9` 字节计算。这个指标是按照算法的必要数据量估算的“有效带宽”，不等于硬件层面所有实际内存事务，也不应直接当作显卡标称物理带宽。

元素吞吐率表示每秒处理多少十亿个输入元素：

```text
GElements/s = n / 时间 / 10^9
```

### 14.9 正确性判断

GPU 的所有 block 部分和被拷回 CPU 后，使用 `double` 再次累加：

```cpp
double err = fabs(h_gpu - h_ref);
double rel = err / (fabs(h_ref) + 1e-30);
```

程序用相对误差 `< 1e-3` 判断通过。分母中的 `1e-30` 用于避免参考值恰好为 0 时发生除零。

浮点加法不满足严格的结合律：

```text
(a + b) + c 不一定逐位等于 a + (b + c)
```

不同归约树会改变加法顺序，所以实际数据上 GPU 结果和串行 CPU 结果可能存在小误差。不能简单要求所有情况都逐位完全相等。

## 15. `main` 中的启动配置

当前配置为：

```cpp
constexpr int n = 1 << 20;
constexpr int blockSize = 256;
constexpr bool strictBenchmark = false;
```

`1 << 20` 等于 1,048,576。根据每个线程首次处理的元素数量，代码分别计算三种 grid：

```cpp
oneElementGrid  = ceil(n / blockSize);       // v0~v2
twoElementGrid  = ceil(n / (2*blockSize));   // v3~v6
fourElementGrid = ceil(n / (4*blockSize));   // v7
```

在当前参数下：

| 版本 | Grid Size | Block Size | 总线程数 |
| --- | ---: | ---: | ---: |
| v0～v2 | 4096 | 256 | 1,048,576 |
| v3～v6 | 2048 | 256 | 524,288 |
| v7 | 1024 | 256 | 262,144 |

后续版本使用更少线程，但每个线程承担更多输入元素。减少线程数并不意味着并行度一定不足，因为当前规模仍远大于 GPU 的并行执行能力。

## 16. 代码中的重要假设与常见陷阱

### 16.1 Block Size 应为 2 的幂

v0～v5 的二分归约逻辑假定 `blockSize` 是 2 的幂，例如 64、128、256 或 512。若使用 300 之类的大小，简单的步长翻倍或减半可能遗漏部分元素。

### 16.2 v4/v5 至少需要 64 个线程

`warpReduce` 的第一条语句会读取 `shmem[tid + 32]`。如果 block 只有 32 个线程，只分配了 32 个 float 的 shared memory，这次访问就会越界。

### 16.3 v5 的模板参数必须与实际 Block Size 一致

下面两者必须相同：

```text
reduce_v5<256>  <->  blockDim.x = 256
```

这是使用编译期特化换取性能时必须承担的约束。

### 16.4 v6/v7 要求完整 Warp

当前写法要求 block 大小是 32 的倍数，并让参与 shuffle 的 warp 中所有 32 个 lane 执行相同的 shuffle 调用。

### 16.5 `__syncthreads()` 不能同步不同 Block

它只对当前 block 有效。这也是 kernel 只能生成 block 部分和、无法在一次普通 kernel 中直接等待其他 block 再完成全局最终归约的原因。

### 16.6 输出数组大小必须与 Grid Size 匹配

每个 block 都会写 `output[blockIdx.x]`，因此至少要分配 `gridSize * sizeof(float)` 字节。

### 16.7 空输入需要单独设计

当前程序假定 `n > 0`。如果要支持 `n = 0`，需要明确定义结果、避免零字节 `cudaMalloc`，并保证 grid size 不为 0。

### 16.8 不要只看一次性能结果

GPU 性能会受预热、动态频率、温度和后台负载影响。先检查 `PASS`，再进行多次采样。需要正式比较时应启用严格模式，并尽量保持测试环境一致。

## 17. 构建与运行

在配置好 CUDA Toolkit 和 C++ 编译器的环境中，从项目根目录执行：

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target Reduce
.\build\Reduce.exe
```

如果 CMake 无法自动识别 GPU 架构，可以根据显卡计算能力显式指定，例如：

```powershell
cmake -S . -B build -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_CUDA_ARCHITECTURES=89
```

Windows 上 CUDA 的宿主编译器通常是 MSVC。应从 Visual Studio 的 x64 Developer PowerShell 或 x64 Native Tools Command Prompt 中构建，避免混用 MinGW 链接器和 MSVC CUDA 工具链。

## 18. 建议的学习与实验顺序

1. 把 `n` 改小，用 8 或 16 个元素在纸上画出 v0、v1、v2 的每轮下标。
2. 在 kernel 中暂时使用小规模输入和 `printf`，观察 `tid`、`gid` 与 `step`，但不要用这种方式测性能。
3. 比较 v2 与 v3，确认 grid 数量为什么减半。
4. 对照 v4/v5，理解“运行时循环”和“编译期展开”的区别。
5. 单独推演一个 warp 的 `__shfl_down_sync` 数据流。
6. 把 `n` 改为不能被 4 整除的值，例如 `1 << 20` 再加 3，验证 v7 尾部处理。
7. 分别测试 block size 128、256、512，但要同步修改 v5 的模板实例，并遵守各版本约束。
8. 启用严格模式，对比中位数和 P5/P95，而不是只比较单次运行。
9. 使用 NVIDIA Nsight Compute 观察 global load、shared memory、warp execution efficiency 和 occupancy，把测量结果与源码中的优化意图对应起来。

## 19. 可以继续扩展的方向

当前项目为了教学清晰，把最终部分和放到 CPU 上归约。掌握这些版本后，可以继续尝试：

- 递归启动归约 kernel，在 GPU 上完成最终求和。
- 使用 Cooperative Groups 表达 warp 和 block 级协作。
- 根据实际活跃 lane 使用 `__activemask()` 或 `__ballot_sync()` 构造 shuffle 掩码。
- 使用 `__syncwarp()` 改写经典 shared-memory warp 归约。
- 比较 `float` 累加与 `double` 累加的精度和性能。
- 使用 Kahan summation 等方法改善数值精度。
- 根据 SM 数量限制 grid，让 grid-stride loop 中的线程处理更多数据。
- 与 CUB 的 `DeviceReduce` 或 `BlockReduce` 做正确性和性能对照。
- 用 Nsight Compute 判断瓶颈究竟来自带宽、指令、同步还是 occupancy。

## 20. 最后总结

这份 `reduce.cu` 展示的并不只是“八种求和写法”，而是一条典型的 CUDA 优化路线：

```text
建立正确基线
  -> 改善线程分支和寻址
  -> 提高每线程工作量
  -> 减少 block 和同步
  -> 用编译期信息展开控制流
  -> 用 warp shuffle 代替 shared-memory 通信
  -> 用向量化加载和 grid-stride loop 改善全局访存与任务分配
```

学习 GPU 优化时应始终遵循三个原则：先保证正确，再解释瓶颈，最后用测量验证优化。仅凭代码形式猜测性能并不可靠；正确性测试、稳定计时和性能分析工具缺一不可。

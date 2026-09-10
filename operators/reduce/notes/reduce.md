# Reduce：从树形归约到向量化加载 —— v0→v7 优化讲解

> 本文是 `operators/reduce/README.md` 的「讲解篇」。README 维护**状态与结论**（版本总表、
> 严格基准、测试场景、构建方式）；本文回答**为什么这么设计**——从“一维整体求和”这个最小
> 问题出发，把 v0→v7 每一步的瓶颈、动机、设计取舍与实测逐层推导出来，顺带沉淀一批在
> 访存受限算子中通用的优化套路与易错点。
>
> 事实以仓库 HEAD 的源码注释、`README.md` 结论记录表与 `docs/benchmark-methodology.md`
> 为准；正文里的代码片段均摘录自 `src/reduce.cu` / `src/reduce.cuh`，并附行号区间。

## 0. 配合源码的阅读路径

推荐顺序：

1. 先读 `README.md` 的「状态」表与「结论记录」表，拿到 v0→v7 的版本画像与实测数字；
2. 再快速过 `src/main.cu`（被测内核表、A/B/C 场景表、`GridFor`）与 `src/test.cu` 的
   `test_reduce_kernel`，理解“被测口径”到底是什么；
3. 最后回到本文逐版本推导，配合 `src/reduce.cu`（v0…v4、v6、v7）与 `src/reduce.cuh`
   （接口声明 + v5 模板）的实现读。

分工对照：

| 话题 | 位置 |
| --- | --- |
| 版本状态、启动约束、一句话版本规划、构建/运行、严格基准 | `README.md` |
| 算法推导、每步动机与代价、瓶颈迁移主线、易错点原理 | 本文（`notes/reduce.md`） |
| 正确性 / 性能的统一度量口径 | `docs/benchmark-methodology.md` |

## 1. 问题与性能模型

### 1.1 要解决的问题

本阶段聚焦一维 fp32 数组的**整体求和**（一维全归约）：

```text
out = Σ_j x[j]        // 输入 x[0, n)，输出标量
```

整体求和是所有归约形态（行 / 列 / 全局）的公共原语，也是 Softmax、GEMM 内部归约的基础
——先在一个最简单的归约上把优化空间摸清，后续扩展行 / 列归约时思路可直接复用。

### 1.2 为什么它是“访存受限”算子

数一下每个输入元素要付出多少计算与多少访存：

- 计算：整体求和只需每元素 1 次加法（n−1 次 ≈ n 次 FLOP）；
- 访存：读入 4·n 字节，另写回 4·grid 字节的部分和。

粗略的算术强度约为 1 FLOP / 4 B ≈ 0.25 FLOP/B，比 GPU 的「算力 / 带宽」比低若干个量级。
因此优化方向几乎不可能是“减少 FLOP”，而只能是：

1. **减少同步与归约本身的固定开销**（共享内存访问、`__syncthreads`、循环）；
2. **提升访存效率**（每线程多元素、向量化加载、减少冗余 block 与空转）。

这正好对应 v0→v7 的演进主线（见 §4、§5）。

### 1.3 有效带宽口径与 L2 警示

归约类算子报告**有效带宽**（见 `docs/benchmark-methodology.md`）：

```text
有效带宽 = (输入读 n 个 float + 输出写 grid 个部分和) / 中位耗时
```

具体在 `test_reduce_kernel`（`src/test.cu:83-85`）中实现为
`bytes = (n + grid) * sizeof(float)`。注意它只计入 grid 个部分和的写回，不计最后一次
主机侧汇总（该步数据量相对 n 可忽略）。

**必须记住的警告**（`README.md` 结论记录表）：
默认 `n = 2^20` 的工作集仅 ~4 MiB，远小于本机 32 MB L2（RTX 4060 Laptop / AD107）。
预热后输入完全驻留 L2，测得的是**片上 L2 命中带宽**而非显存物理带宽——v5 之后数值超过
GDDR6 理论峰值 256 GB/s 即由此而来，并非违反物理上限。因此这些数字**只能用于同规模下
各版本的横向对比**；要验证真正的显存带宽，需要把工作集加大到远超 L2 的量级。

### 1.4 测试口径速览（先跑对，再跑快）

- 参考实现 `reduce_cpu` 在主机端用 **double** 累加，抑制长序列 fp32 顺序累加的舍入误差
  （`src/reduce.cu:16-22`）；
- 正确性判据：GPU 结果 vs double 参考的相对误差 ≤ **1e-3**（归约顺序随粒度而变，参照
  GEMM 放宽，见 `docs/benchmark-methodology.md`）；
- 测试输入取 `i % 1000`（确定性、可复现，和为较大的正数，避免正负抵消放大相对误差）；
- 计时：开发档 1 次预热 + 100 次迭代；严格档 100 次预热 + 21 组 × 1000 次（面向本机
  RTX 4060 Laptop 下调采样量），报 **中位数**与 P5/P95。

## 2. 两阶段归约约定与统一签名

### 2.1 为什么是“两阶段”而不是一个内核出标量

让每个 block 只负责把一块连续输入收敛成 **1 个部分和**，写入
`output[blockIdx.x]`（所以 `output` 至少要有 `grid` 个元素）；最终标量由调用方对
`output[0, grid)` 再做一次轻量汇总。

这样做的收益：

- **各版本签名完全一致**，可用同一套测试驱动以函数指针方式驱动（见 §2.2）；
- 每 block 的内部归约是**单 block、可复现**的子树，容易推理与调优；
- 二次汇总可以走主机端 double 累加（`src/test.cu:92-95`），既简单又避免部分和再次舍入；
- 也为未来“多 block 各自归约 → 最终一次汇总”的并行扩张留了空间。

### 2.2 统一签名 ReduceKernel 与函数指针测试驱动

```cpp
// src/reduce.cuh:134
using ReduceKernel = void (*)(const float* input, float* output, int n);
```

v0…v7 输出约定一致，因此 `test_reduce_kernel` 只认这一个函数指针类型
（`src/test.cuh:10`、`src/test.cu:13`），通过 `cudaLaunchKernel` + 运行期函数指针启动
（`src/test.cu:48-58`），一份驱动覆盖所有版本。`main.cu` 用一张“名字 + 函数指针 +
每线程元素数”的表（`kKernels`，`src/main.cu:38-47`）注册被测内核，接入新版本 = 往表里
追加一行。

### 2.3 “超配安全”与空输入

- 覆盖口径按“每 block 覆盖固定 span”走时（v0…v6），`GridFor` 计算恰好覆盖 n 所需的
  block 数：`base = ceil(n / (block * elems_per_thread))`，且 `n == 0` 时 `base` 至少为 1
  （`src/main.cu:68-73`）。越界元素一律**补 0**，所以：
  - 多余配的 block 全部越界、部分和为 0，不影响最终结果 —— **超配安全**；
  - `n == 0` 时所有线程都走越界分支、不读输入，输出恒为 0。
- v7 换成 grid-stride 扫描后更宽松：任意 `grid >= 1` 都完整覆盖输入，冗余 block 只是
  空转并写 0（见 §4.8）。

## 3. 一切开始之前：把“正确”钉死

先有正确性基线，优化才有意义：

- `reduce_cpu`（double 累加，`src/reduce.cu:16-22`）作为主机参考；
- v0 作为 GPU 端第一个实现，结果必须与参考一致（容差 1e-3 内）。

共享内存树形归约的通用框架是：每线程搬 1 个（或多个）元素进 smem → 若干轮两两合并、
把部分和数量每轮减半 → 收敛到 `smem[0]` → 由 tid 0 写 `output[blockIdx.x]`。从 v0 到
v7 的优化，本质是对“搬入 + 合并”两个阶段在**活跃线程形态、共享内存访问形态、同步
次数、归约载体（smem / 寄存器）、加载指令宽度**五个维度上逐项打磨：

```mermaid
flowchart LR
    v0["v0 交错寻址<br/>正确性基线 · 有 warp 分歧"] -->|"活跃线程连续化"| v1["v1 连续寻址"]
    v1 -->|"步长折半 · 连续槽寻址"| v2["v2 折半步长"]
    v2 -->|"每线程 2 元素预加和"| v3["v3 每线程多元素"]
    v3 -->|"归约尾部交给 warp 0 展开"| v4["v4 末 warp 展开"]
    v4 -->|"block 尺寸编译期常量"| v5["v5 模板展开"]
    v5 -->|"寄存器 shuffle 替代 smem 树"| v6["v6 两级 warp shuffle"]
    v6 -->|"float4 向量化 + grid-stride"| v7["v7 向量化加载"]
```

下表先给结论，§4 逐个展开：

| 版本 | 主要瓶颈 | 关键改动 | 对齐形状有效带宽（n=2^20） |
| --- | --- | --- | --- |
| v0 | warp 内分歧 + 每线程 1 元素 | 交错寻址树形归约 | 55 GB/s（基线） |
| v1 | smem 访问形态有 bank 冲突 | 连续寻址 | 99 GB/s |
| v2 | 全局读 + 固定开销为主 | 折半步长，读写连续槽 | 103 GB/s |
| v3 | 每线程加载太少 | 每线程 2 元素、grid 减半 | 181 GB/s |
| v4 | 树形归约轮次与同步多 | 归约停 32，末 warp 展开 | 252 GB/s |
| v5 | 运行时归约循环 | block 尺寸模板常量化 | 272 GB/s |
| v6 | 归约仍走共享内存 | 两级 warp shuffle | 340 GB/s |
| v7 | 标量加载指令/事务多 | float4 + grid-stride | 412 GB/s |

## 4. 逐步优化：v0 → v7

> 以下每节的“实测”均取 `README.md` 结论记录表 n=2^20（对齐）严格基准数字；设备 /
> 构建 / 采样口径见该表，不再重复。所有版本的约束共同点：block 需为 2 的幂（默认
> 256），动态 smem = `blockDim.x * sizeof(float)`（v5 模板同理，v6/v7 用静态
> `warp_results[32]`）。

### 4.1 v0 —— 交错寻址树形归约（正确性基线）

**上一版瓶颈 / 动机**：没有上一版。v0 的首要目标是**简单、正确、可读**，作为后续所有
版本的对照基线。

**设计**：每线程搬 1 个元素到 `smem[tid]`（越界补 0）；随后 step 从 1 倍增，满足
`tid % (2*step) == 0` 的线程把 `smem[tid+step]` 并入 `smem[tid]`，`log2(block)` 轮后
收敛到 `smem[0]`。

```cpp
// src/reduce.cu:30-49（核心循环）
smem[tid] = (gid < n) ? input[gid] : 0.0f;
__syncthreads();                          // 槽位全部就绪后才能开始归约
for (size_t step = 1; step < blockDim.x; step *= 2) {
  if (tid % (2 * step) == 0) {
    smem[tid] += smem[tid + step];
  }
  __syncthreads();                        // 下一轮要读本轮刚写入的局部和
}
if (tid == 0) output[blockIdx.x] = smem[0];
```

**正确性与约束**：每轮合并前必须 `__syncthreads`，因为下轮线程读的是其他线程本轮写入
的槽；block 为 2 的幂保证各轮配对均匀。

**代价**：step 较小时活跃线程在 warp 内**交错分布**（`tid % (2*step) == 0`），同一 warp
里只有一部分 lane 干活、其余空转，每轮都有 warp 分歧；多数线程的算力被浪费。

**实测**：55 GB/s，作为基线。

### 4.2 v1 —— 连续寻址（消除 warp 内分歧）

**上一版瓶颈**：v0 的活跃线程是交错的，warp 内每轮都有分歧。

**设计**：把活跃线程从“交错间隔”改成**连续前缀**：每轮 `index = tid * 2 * step`，
`index < blockDim.x` 时合并 `smem[index] += smem[index+step]`。于是要么整条 warp 活跃、
要么整条 warp 空闲，warp 内不再有分歧。

```cpp
// src/reduce.cu:57-77（核心循环）
for (size_t step = 1; step < blockDim.x; step *= 2) {
  const int index = 2 * static_cast<int>(step) * tid;
  if (index < blockDim.x) {
    smem[index] += smem[index + static_cast<int>(step)];
  }
  __syncthreads();
}
```

**正确性 / 约束**：与 v0 完全同构——网格/块模型、启动约束、签名一致，`reduce_v1` 只是
换了个活跃线程筛选方式。

**实测**：99 GB/s，相对 v0 约 +79%。分歧消除立竿见影。

### 4.3 v2 —— 折半步长（消除 smem bank 冲突）

**上一版瓶颈**：v1 按 `index = tid*2*step` 寻址，多数轮次下访问 stride 是 2·step 的
间隔，同一 warp 的相邻线程会打在同一（几个）bank 上，存在 2 路及以上的共享内存 bank
冲突。

**设计**：步长方向反过来——stride 自 `blockDim.x/2` 每轮折半到 1；线程 tid（`tid <
stride`）把 `smem[tid]` 与 `smem[tid+stride]` 合并后**就地写回 `smem[tid]`**。部分和始终
落在数组最前端的连续槽上，读写下标在活跃段内连续 → **无 bank 冲突**；活跃线程仍是连续
前缀，无 warp 内分歧。

```cpp
// src/reduce.cu:86-105（核心循环）
for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
  if (static_cast<size_t>(tid) < stride) {
    smem[tid] += smem[tid + static_cast<int>(stride)];
  }
  __syncthreads();
}
```

**实测**：103 GB/s，相对 v1 约 +4%，且差量落在正常流程噪声带内。这说明 v1→v2 读入的
全局数据量与同步轮次完全相同，**归约耗时大头在全局读与固定开销**，单纯消除 bank 冲突
带不来数量级提升——下一档收益必须来自算法层（v3 的每线程多元素），而不是共享内存访问
形态的微调。

### 4.4 v3 —— 每线程 2 元素（标量加载，grid 减半）

**上一版瓶颈**：v0…v2 每线程只搬 1 个元素，每 block 覆盖 `block` 个连续元素；要覆盖
n=2^20 需要 4096 个 block，每个线程只发 1 条全局加载就进入归约，加载/归约的固定开销
摊不开。

**设计**：每 block 覆盖 `2*blockDim.x` 个连续元素。线程 tid 的段内下标
`gid = blockIdx.x * (2*blockDim.x) + tid`，先把 `gid` 与 `gid + blockDim.x` 两个元素在
**寄存器里预加和**（越界跳过，等价补 0），再走 v2 的折半步长归约。两次加载在 warp 内
各自连续且互不依赖，还能提升内存级并行。

```cpp
// src/reduce.cu:114-136（加载 + 预加和）
float val = 0.0f;
if (gid < n) val += input[gid];
if (gid + blockDim.x < n) val += input[gid + blockDim.x];
smem[tid] = val;                          // smem 存“每线程 2 元素预加和”而非原始元素
__syncthreads();
for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) { /* v2 折半 */ }
```

**正确性 / 约束**：签名同 v0/v1/v2，唯一差别是覆盖 n 所需 grid 减半为
`ceil(n / (2*blockDim.x))` —— `GridFor` 已按“每线程元素数”（`elems_per_thread`）计算，
`main.cu` 把 v3 的该字段记为 2（`src/main.cu:41`、`:68-73`）。测试复用同一驱动，越界
补 0 逻辑等价。

**实测**：181 GB/s，相对 v2 约 +76%；尾部非对齐形状约 -2%（末 block 分摊不均）。每线程
预加和把一次归约摊到 2 次全局读上，收益明显。

### 4.5 v4 —— 归约尾部交给 warp 0 展开（省 5 轮同步）

**上一版瓶颈**：v3 的折半树仍要跑满 `log2(block)` = 8 轮，每轮一次 `__syncthreads`；
block=256 时最后 5 轮（stride 16→1）只涉及前 32 个槽，却要让全 block 同步 5 次。

**设计**：折半归约**只做到 stride = 32 就停**，剩余 5 轮改由 **warp 0** 对残留在 smem
前端的 32 个部分和做**逐位展开**合并：

```cpp
// src/reduce.cu:158-184（归约尾部）与 :144-151（warpReduce）
for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) { /* ... __syncthreads() */ }
if (static_cast<unsigned int>(tid) < 32) {
  warpReduce(smem, tid);   // smem[tid] += smem[tid+32/+16/+8/+4/+2/+1]
}
if (tid == 0) output[blockIdx.x] = smem[0];
```

warp 内指令按 SIMT 同步推进，展开写不必再加 `__syncthreads`；但 smem 必须声明为
`volatile`，强制每次读写真实落内存——否则编译器可能把中间结果缓存在寄存器里，错过其他
lane 刚写入的值（这也是为什么 `warpReduce` 的形参是 `volatile float*`）。

**正确性 / 约束**：要求 block 为 2 的幂且 ≥ 64（展开第一步要读 `smem[tid+32]`）。

**实测**：252 GB/s，相对 v3 约 +39% —— 同步确实是树形归约的主要开销之一。

### 4.6 v5 —— block 尺寸编译期常量化（模板展开）

**上一版瓶颈**：v4 的折半循环仍在运行时逐轮判断 stride；stride 序列在 block=256 时是
固定的（128→64→32…），本可以在编译期就确定要执行哪几轮。

**设计**：block 尺寸改由模板参数 `BLOCK_SIZE` 固定，归约各级写成
`if (BLOCK_SIZE >= 512) {...}` 这样的**编译期常量条件**。编译器只保留当前 `BLOCK_SIZE`
真正需要的折半级并整体展开 —— 没有运行时归约循环，指令序列更短、更易优化：

```cpp
// src/reduce.cuh:71-115（reduce_v5 模板，编译期归约级示意）
if (BLOCK_SIZE >= 512) { if (tid < 256) smem[tid] += smem[tid + 256]; __syncthreads(); }
if (BLOCK_SIZE >= 256) { if (tid < 128) smem[tid] += smem[tid + 128]; __syncthreads(); }
if (BLOCK_SIZE >= 128) { if (tid < 64)  smem[tid] += smem[tid + 64];  __syncthreads(); }
// 收尾同 v4：<= 64 个部分和由 warp 0 以 volatile smem 展开合并
```

**工程约束（值得单独记一笔）**：在 `-rdc=false`（默认）下，跨翻译单元引用
`__global__` 模板特化已被 nvcc 弃用，因此 `reduce_v5` 的**模板定义整体内联在
`reduce.cuh`**，由各实例化点自行生成设备代码；`main.cu` 以 `reduce_v5<kBlock>` 注册
（`src/main.cu:44`），且显式实例化须与 `kBlock = 256` 保持同步。

**实测**：272 GB/s；同场 v4 复测 260 GB/s → 约 +4.6%。收益有限——如 v2 处所料，瓶颈已
从“归约循环”转移到“全局访存”，编译期展开只是顺手把归约侧的固定开销榨干。

### 4.7 v6 —— 两级 warp shuffle（块内只剩一次同步）

**上一版瓶颈**：即使展开，树形归约的数据流仍然每轮都要经过共享内存；v4/v5 里归约的
实际内容（合并 32/64 个部分和）相比全局读占比已经很小，剩余开销是 smem 往返与那几次
同步。

**设计**：块内归约整体换成**寄存器 shuffle**：

1. `warpReduceSum` 用 5 轮 `__shfl_down_sync`（offset 16→1）把每 warp 归成 1 个部分和
   （全程在寄存器、不碰 smem、无需同步）；
2. lane 0 把结果写入 `warp_results[wid]`；
3. 一次 `__syncthreads` 后，warp 0 对 `numWarps` 个部分和再做一次 shuffle 归约
   （`lane < numWarps` 取对应部分和，否则视为 0，但**整个 warp 0 都参与**归约）。

```cpp
// src/reduce.cu:213-243（两级归约骨架）与 :198-203（warpReduceSum）
val = warpReduceSum(val);                 // ① warp 内归约
if (lane == 0) warp_results[wid] = val;
__syncthreads();
const int numWarps = blockDim.x / 32;
if (wid == 0) {
  val = (lane < numWarps) ? warp_results[lane] : 0.0f;
  val = warpReduceSum(val);               // ② warp 0 归约 numWarps 个部分和
}
if (tid == 0) output[blockIdx.x] = val;
```

**正确性要点**：`__shfl_down_sync` 的 mask 为 `0xffffffff`，要求**整个 warp 收敛**参与，
任何 lane 提前退出都是未定义行为 —— 所以第 ② 级即使只有 `numWarps` 个有效部分和，也要
让 lane ≥ numWarps 的线程以 0 参与、不能跳过。

**约束**：block 为 2 的幂且 `32 <= block <= 1024`；`numWarps` 个部分和须装得进
`warp_results[32]`。加载与覆盖口径同 v3/v4/v5（每 block 覆盖 `2*block` 个元素）。

**实测**：340 GB/s；同场 v4/v5 复测约 311/331 GB/s → 相对 v5 约 +2.5%。收益进一步收窄：
此时块内归约已是“无关紧要”的部分，**瓶颈彻底转移到全局访存**——这也预告了 v7 的方向。

### 4.8 v7 —— float4 向量化加载 + grid-stride 扫描

**上一版瓶颈**：v6 每线程仍只有 2 条**标量**加载；覆盖 n=2^20 需 2048 个 block，加载
指令数多、每条指令只搬 4 B，全局访存指令与事务的开销成了主角。

**设计**：加载阶段与 v0…v6“每 block 分块连续覆盖”的思路完全不同，改成
**grid-stride 扫描 + float4 向量化**：

- 全体线程以 `gridDim.x * blockDim.x` 为总步长联合遍历 float4 序列，每轮读入 1 个
  `float4`（16 B）做 4 分量寄存器累加 —— 全局加载指令数与访存事务约为标量加载的 1/4；
- input 需 **16 字节对齐**（`cudaMalloc` 分配天然满足）；
- `n % 4` 的尾部（至多 3 个元素）由第二个 grid-stride 循环以**标量**方式补齐；
- 块内归约沿用 v6 的两级 warp shuffle（约束同 v6）。

```cpp
// src/reduce.cu:259-302（加载骨架）
const float4* input4 = reinterpret_cast<const float4*>(input);
const int n4 = n / 4;
float val = 0.0f;
for (int idx = blockIdx.x * blockDim.x + tid; idx < n4;
     idx += gridDim.x * blockDim.x) {
  const float4 data = input4[idx];
  val += data.x + data.y + data.z + data.w;
}
// [n4*4, n) 尾部标量补齐（同 stride 的 grid-stride 循环）
// ... 块内两级 warp shuffle（同 v6）
```

**与之前版本不同的覆盖语义**：grid-stride 对任意 `grid >= 1` 都完整覆盖输入，启动网格
不再要求严格配比；冗余 block 只是不读数据、写 0，仍满足“超配安全”。`main.cu` 里 v7 的
`elems_per_thread = 4` 只用于给出“多数线程单轮读完”的**推荐网格** `ceil(n/(4*block))`
（n=2^20 → 1024），而非精确覆盖所需（`src/main.cu:46`、`:62-73`）。

**正确性 / 约束**：float4 主循环要求 16 字节对齐；`n % 4` 余 1/2/3 的尾部路径由边界
场景表专门覆盖（`main.cu` 的 `n=block-2`、`kBlock±1` 组合出余 2/1/3，见
`src/main.cu:83-92`）。

**实测**：412 GB/s，相对 v6 约 +20.7%（对齐）；尾部非对齐变体 381 GB/s、约 +12.8%。
向量化加载是最后一档明显收益 —— 它直接压低了“发指令”本身的开销。

## 5. 演进小结：瓶颈迁移主线

### 5.1 一张表看全演进（n=2^20 对齐，数字取自 README 结论记录表）

| 版本 | 中位数 ms | 有效带宽 GB/s | 相对上一版 | 这一步解决的瓶颈 |
| --- | --- | --- | --- | --- |
| v0 | 0.0765 | 55 | — | 基线（正确性优先） |
| v1 | 0.0426 | 99 | ~+79% | warp 内分歧 |
| v2 | 0.0410 | 103 | ~+4% | smem bank 冲突 |
| v3 | 0.0233 | 181 | ~+76% | 每线程加载太少（算法层） |
| v4 | 0.0167 | 252 | ~+39% | 树形归约轮次 / 同步 |
| v5 | 0.0154 | 272 | ~+4.6% | 运行时归约循环 |
| v6 | 0.0124 | 340 | ~+2.5% | smem 往返 / 同步 |
| v7 | 0.0102 | 412 | ~+20.7% | 标量加载指令与事务 |

> 相对增益按同场复测口径折算（v5/v6/v7 三档含复测数字，细节见 README）；全部
> `max_err = 0`，正确性全程达标。数字受 §1.3 的 L2 驻留效应影响，只做同规模横向对比。

### 5.2 主线一句话

优化永远在追当前**占比最大的瓶颈**，瓶颈随优化逐步转移：

```text
分歧（v0）→ smem bank 冲突（v1）→ [算法层：加载太少]（v2→v3）
→ 同步 / 归约轮次（v4）→ 归约循环 / smem 往返（v5/v6）
→ 全局访存指令与事务（v7）→ 理论上限 = 访存（L2 / DRAM）带宽
```

观察：

- v1→v2 与 v4→v5、v5→v6 几档收益都很小，因为它们只是在**已被挤压的维度**上继续微调；
  大跳跃全部来自**换算法/换访存结构**（v0→v1、v2→v3、v3→v4、v6→v7）；
- 越往后收益越接近访存上限并趋于递减——这是访存受限算子的典型规律，下一步要么扩展
  归约形态（行/列），要么换实现语言/换更大工作集验证显存带宽（见 §7）。

### 5.3 一个反直觉点：为什么要为“正确性基线”保留 v0

v0 又慢又有分歧，但它是最容易被人工核对、最适合当差分对象的实现。每一版优化都在
“改变活跃线程形态 / 寻址 / 归约载体 / 加载宽度”，而 v0 提供了一个语义最简单的最小
公共参照，保证后续改动没有悄悄改变“数学上求的是同一个和”。

## 6. 陷阱清单（踩过 / 容易踩的坑）

1. **`__syncthreads` 必须全 block 配对收敛**：树形归约每轮都有部分线程“读别人刚写的
   局部和”，漏一轮 barrier 会读到旧值；条件分支里夹 barrier 更危险。
2. **`volatile` 只在“同一 warp、SIMT 推进”的前提下够用**：v4 的 `warpReduce` 展开能省
   同步，前提是 0..31 lane 在 lockstep 下写同一段 smem；volatile 只是防止编译器缓存到
   寄存器，跨 warp 的写仍需要真正的 barrier。
3. **shuffle 掩码与全 warp 收敛**：`__shfl_down_sync(0xffffffff, ...)` 要求整条 warp
   参与；只让“有意义的 lane”参与而让其他 lane 提前退出是未定义行为。v6 第 ② 级用
   “无效 lane 补 0”来满足收敛，而不是跳过。
4. **`float4` 需要 16 字节对齐**：`reinterpret_cast` 不负责对齐，`cudaMalloc` 天然满足，
   但如果未来把输入换成用户缓冲或做了偏移就要显式保证；`n%4` 尾部必须标量补齐，
   否则越界读。
5. **越界补 0 是“超配安全”的地基**：固定 span 覆盖的版本里多余 block 全部越界、
   部分和为 0；若改成不补 0 直接读，超配 block 会读非法地址。
6. **空输入 `n == 0`**：`cudaMalloc(d, 0)` 行为未定义，测试侧按 `max(n,1)` 申请
   （`src/test.cu:37-40`）；启动侧 `GridFor` 保证 `grid >= 1`；内核则全走越界补 0 分支、
   输出恒 0。
7. **参数契约防御**：`n < 0 || grid < 1` 判 FAIL 而非崩溃（`src/test.cu:21-25`）。
8. **fp32 长序列累加误差**：误差随项数增长（4096 项顺序累加可达 ~1e-4 量级），且归约
   顺序随粒度而变，所以参考实现用 double 累加、容差放宽到 1e-3；测试输入选非负的
   `i % 1000` 避免正负抵消把相对误差放大成假 FAIL。
9. **计时口径**：中位数为主指标，P5/P95 反映波动；严格档（100 预热 + 21×1000，面向本机
   RTX 4060 Laptop 下调）出的数字才进结论表，开发档只做快速冒烟。注意工作集小于 L2 时“带宽”是 L2 命中带宽。
10. **`-rdc=false` 下跨翻译单元引用 `__global__` 模板特化已被弃用**：需要模板化的内核
    定义内联进 `.cuh`（v5 的教训），避免 nvcc 弃用告警与链接错误。

## 7. 扩展方向

- **vx —— 行 / 列 / 全局归约**：整体求和是公共原语，默认扩展形态是行求和
  `out[i] = Σ_j x[i,j]`。一个 block 处理多行可以摊薄调度与边界开销；两阶段约定里的
  “每 block 1 个部分和”也可推广为“每 block 负责若干行、每行产出 1 个结果”。
- **更大工作集 / 显存带宽验证**：把 n 加大到远超 L2（≥ 数百 MiB），重新测量以逼近
  显存物理带宽，验证向量化加载在真实访存压力下的收益。
- **Triton t0/t1**：同一算子用 Triton 实现并做正确性对齐与性能对照（规划中）——树形
  归约 / shuffle / 向量化这一整套推导，恰好是理解 Triton 底层会替你做什么、不替你
  做什么的绝佳素材。

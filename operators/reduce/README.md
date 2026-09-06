# Reduce

对一维 fp32 数组做归约求和。**当前阶段聚焦“整体求和”（一维全归约）**：

```text
out = Σ_j x[j]        // 输入 x[0, n)，输出标量
```

整体求和是所有归约形态的公共原语，也是 Softmax / GEMM 等算子内部归约的基础。
后续在此原语之上扩展**行 / 列 / 全局归约**（行求和即为默认目标形态）：

```text
out[i] = Σ_j x[i, j]  // 行求和（row-sum），输入 rows×cols，输出长度 rows
```

该算子属典型的**访存受限**算子；核心是 block 级归约原语（共享内存树形归约，
后续可升级为 warp shuffle + 跨 warp 汇总）。

## 状态

| 阶段 | 版本 | 实现要点 | 状态 |
| --- | --- | --- | --- |
| CUDA | v0 | 交错寻址共享内存树形归约（interleaved addressing）：一段连续元素交给一个 block，块内 log2(block) 轮收敛出 1 个部分和；活跃线程在 warp 内交错分布，存在分歧 | 完成（正确性基线） |
| CUDA | v1 | 连续寻址共享内存树形归约（consecutive addressing）：活跃线程改为连续前缀，消除 v0 的 warp 内分歧；网格/块模型、启动约束与签名和 v0 完全一致 | 完成 |
| CUDA | v2 | v1 + 每线程多元素展开 + `float4` 向量化（需对齐约束与尾列处理） | 规划中 |
| CUDA | vx | 扩展形态：行/列/全局归约、一个 block 处理多行以摊薄调度 | 规划中 |
| Triton | t0/t1 | 与 CUDA 同规格的 Triton 实现并做性能对照 | 规划中 |

- 正确性判据：相对误差 ≤ 1e-3（见 `docs/benchmark-methodology.md`）。长序列 fp32 累加误差随项数增长（4096 项顺序累加已可达 ~1e-4 量级），且归约顺序随实现粒度变化，故参照 GEMM 放宽容差。
- 指标：有效带宽 =（输入读 + 输出写）/ 中位耗时。

## 版本规划说明

- **v0 交错寻址**：每个 block 处理 `blockDim.x` 个连续元素，越界补 0；块内按
  `tid % (2*step) == 0` 让交错间隔的线程逐轮合并，结果收敛到 `smem[0]`。
  直观、适合做正确性基线；缺点是每轮活跃线程在 warp 内交错分布、造成分歧。
- **v1 连续寻址**：把 v0 的活跃线程从“交错间隔”改为“连续前缀”
  （`index = tid * 2 * step < blockDim.x`），整条 warp 全活跃或全空闲，无 warp
  内分歧。本机 Release 实测快于 v0（严格基准数字记入下方“结论记录”表），
  可作为后续向量化版本（v2）的起点。
- **v2 向量化**：在 v1 基础上用 `float4` 连续加载 + 每线程多元素展开（需行
  对齐约束与尾列处理），减少指令与访存事务。
- 两阶段归约约定：每个 block 只产出 1 个部分和到 `output[blockIdx.x]`，最终
  标量由调用方对 `output[0, grid)` 做一次轻量求和。该约定保证了各版本签名
  完全一致（`ReduceKernel`），从而可被同一套测试驱动复用。

## 参考规模

- 默认：`n = 2^20`（fp32，约 4 MiB 输入，`grid = n/block = 4096`）。
- 可选：`n = 2^20 + 1000`（尾部非对齐，覆盖越界补 0 路径）等边界形状见
  `src/main.cu` 的场景表。
- block 大小默认 256。

## 目录布局与测试

```
reduce/
├── README.md   # 本文档：规划 + 结论总表
├── CMakeLists.txt  # 构建脚本（src/main.cu 存在即自动启用）
└── src/        # CUDA 实现（v0/v1 已完成）
    ├── reduce.cuh / reduce.cu   # 算子接口与实现（被测试对象）
    ├── test.cuh  / test.cu      # 可复用测试驱动：正确性(容差1e-3)+性能
    └── main.cu                  # 执行入口：注册 v0/v1，运行测试
```

> 注：`triton/`（第二阶段 Triton 实现）与 `notes/`（学习笔记）属规划目录，尚未创建。

测试入口 `src/main.cu` 将 `reduce_v0` / `reduce_v1` 注册给同一测试驱动，覆盖
三类场景（每组场景两个内核各跑一遍）：

- **正常流程**：大规模对齐（`n=2^20`）、尾部非对齐（`n=2^20+1000`）；
- **边界条件**：`n=1` 单元素、`n=block` 恰一个 block 满载、`n=block-1`、
  `n=block+1`、`n=2*block-1`（末 block 仅 1 个有效元素）；
- **异常 / 健壮性**：`n=0` 空输入（期望和 = 0）、grid 超配 +3 个冗余 block
  （多余 block 全补 0、不影响结果）、非法参数（`n<0` / `grid<1` 判 FAIL）。

构建与运行（仓库根目录）：

```bash
cmake --preset release
cmake --build build --target reduce
./build/operators/reduce/reduce
```

## 结论记录

（严格基准结果按 `docs/benchmark-methodology.md` 口径运行后填写）

| 版本 | 形状 | 中位数 ms | 有效带宽 GB/s | max_err | 备注 |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

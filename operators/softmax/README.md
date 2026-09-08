# Softmax

对行主序 fp32 矩阵逐行做数值稳定的 softmax：

```text
m_i = max_j x_ij
y_ij = exp(x_ij - m_i) / Σ_j exp(x_ij - m_i)
```

规模关注 `rows × cols`，重点考察全局带宽上限的逼近程度（该算子算力需求低，是典型的**访存受限**算子）。

## 状态

| 阶段 | 版本 | 实现要点 | 状态 |
| --- | --- | --- | --- |
| CUDA | v0 | 每行一个 block，行内列维由 `blockDim.x` 个线程协同遍历（stride = block）；共享内存树形规约两次：先求行最大、再求 Σexp（max-shift 数值稳定），写回时第三次读行重算 exp | 完成（正确性基线，已接入测试） |
| CUDA | v1 | online（单遍）：一次遍历逐段更新 running max/sum | 未开始 |
| CUDA | v2 | v1 + 向量化访问（`float4` 读写，按行对齐约束） | 未开始 |
| Triton | t0 | 与 CUDA 同规格的 Triton 实现 | 规划中 |
| Triton | t1 | 与 CUDA 性能对照并记录 | 规划中 |

- 正确性判据：相对误差 ≤ 1e-5（逐元素，跳过 `|ref| < 1e-30` 的元素，见 `docs/benchmark-methodology.md`）。
- 指标：有效带宽 =（输入读 + 输出写）/ 中位耗时，其中输入输出按**逻辑数据量**各计 1 次（每元素 4 B 读 + 4 B 写）。内核实际可能多次读行（如 v0 每元素读 3 次），低效会直接反映为更低的有效带宽，便于横向比较。

## 版本规划说明

- **v0 朴素两遍规约（本轮完成）**：`grid = rows`、每 block 处理一行；行内列维由
  blockDim 个线程以 `stride = blockDim` 协同遍历（warp 内列下标连续 → 全局读合并，
  行宽 `cols` 任意、不要求整除 block）。共享内存只存规约中间量（`block` 个 float）：
  第①遍树形 `fmaxf` 归约出行最大 m；第②遍以 `expf(x - m)` 树形归约出 Σexp；第③遍
  再读一次行、写 `y = expf(x - m) / Σexp`。Σexp 经块内分段 + 树形归约，fp32 舍入误差
  在 ~(每线程元素数 + log2 block)·ulp 量级，可满足 1e-5 容差（单线程串行累加整行
  会超差，故不用）。每元素实际读行 3 次、写 1 次，是直观但访存低效的正确性基线。
- **v1 online（单遍）**：一次遍历同时维护 `m` 与 `Σexp`，是 FlashAttention 的前置知识，故优先做。
- **v2 向量化**：在 v1 基础上用 `float4` 提升访存效率；若行宽不能整除 4，则做尾列处理。

## 参考规模

- 默认：`rows=4096, cols=4096`（fp32，约 64 MiB 输入）。
- 可选：`rows=16384, cols=1024`（宽行场景）。
- block 大小默认 256。

## 目录布局与测试

```
softmax/
├── README.md   # 本文档：规划 + 结论总表
├── CMakeLists.txt  # 构建脚本（src/main.cu 存在即自动启用）
└── src/        # CUDA 实现（当前 v0，后续版本追加进 softmax.cuh/.cu）
    ├── softmax.cuh / softmax.cu   # 算子接口与实现（被测试对象）
    ├── test.cuh  / test.cu        # 可复用测试驱动：正确性(容差1e-5)+性能
    └── main.cu                    # 执行入口：注册 v0，运行测试
```

> 注：`triton/` 与 `notes/` 属规划目录，尚未创建。

测试入口 `src/main.cu` 将 `softmax_v0` 注册给同一测试驱动，覆盖三类场景
（每场景启动网格均为 `grid = rows`）：

- **正常流程**：`4096×4096`、`16384×1024`；
- **边界条件**：`1×1`、行宽在 block 边界附近的 `1×(block±1)`、恰满载的 `1×block`、
  第 2 列块仅 1 列的 `1×(2*block-1)`，以及多行多列块 `3×(2*block)`；
- **异常 / 健壮性**：`0×1024` 空矩阵（内核越界空转）、`3×0` 空行（不读不写）。

开关（均为 `RunScenarios` 的函数参数，**默认关闭**，在 `main()` 中集中设置）：

| 开关 | 默认 | 作用 |
| --- | --- | --- |
| `enable_boundary` | `false` | 是否执行“边界条件”与“异常 / 健壮性”场景；关闭时只跑正常流程（2 项），开启后为全量回归（10 项） |
| `strict_benchmark` | `false` | 是否按严格口径采样（透传给 `test_softmax_kernel` 的同名参数）：关闭时 1 次预热 + 100 次迭代；开启时 1000 次预热 + 21 组 × 2000 次并输出 P5/P95（softmax 单次内核开销远大于 reduce，严格档迭代数相对 reduce 的 21×10000 折半） |

日常开发保持默认即可（只跑有性能意义的大规模形状）；出严格基准数字或做全量
回归时，把 `main()` 中对应常量改为 `true`。

构建与运行（仓库根目录）：

```bash
cmake --preset release
cmake --build build --target softmax
./build/operators/softmax/softmax
```

## 结论记录

严格基准按 `docs/benchmark-methodology.md` 口径实测（开关 `strict_benchmark=true`）。

| 设备 / 构建 / 采样 | 值 |
| --- | --- |
| 设备 | NVIDIA GeForce RTX 4060 Laptop (sm_89)，CUDA 12.9 |
| 构建 | Release, block=256, arch=89 |
| 预热 / 迭代 | 1000 次预热 / 21 组 × 2000 次 |

口径：中位数为主指标（P5/P95 见运行日志）；有效带宽按**逻辑数据量**（每元素
读 1 次 + 写 1 次）计；`max_err` 为 GPU vs 主机 double 参考的相对误差（容差 1e-5）。

> 数字说明：v0 每元素实际读行 3 次（两遍规约 + 写回重算 exp），实际访存流量约为
> 逻辑数据量的 2 倍，故下表有效带宽低于显存理论峰值 256 GB/s 属预期；该口径便于
> 与后续 v1/v2（读行次数减少）做横向对比。输入工作集 16–64 MiB，与 32 MB L2 同量级，
> 预热后部分数据驻留 L2，数字仍只用于同规模下各版本对比。

| 版本 | 形状 | 中位数 ms | 有效带宽 GB/s | max_err | 备注 |
| --- | --- | --- | --- | --- | --- |
| v0 | 4096x4096 | 0.7462 | 179.87 | 1.231e-06 | 正确性基线 |
| v0 | 16384x1024 | 0.8267 | 162.36 | 1.234e-06 | 宽行场景 |

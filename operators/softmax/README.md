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
| CUDA | v0 | 每行一个 block，朴素两次归约（先求 max，再求和） | 未开始 |
| CUDA | v1 | 单次遍历（online softmax）：逐段更新 running max/sum | 未开始 |
| CUDA | v2 | v1 + 向量化访问（`float4` 读写，按行对齐约束） | 未开始 |
| Triton | t0 | 与 CUDA 同规格的 Triton 实现 | 规划中 |
| Triton | t1 | 与 CUDA 性能对照并记录 | 规划中 |

- 正确性判据：相对误差 ≤ 1e-5（见 `docs/benchmark-methodology.md`）。
- 指标：有效带宽 =（输入读 + 输出写）/ 中位耗时。

## 版本规划说明

- **v0 朴素两次遍历**：直观但每行读两遍全局内存；用于建立正确性基线。
- **v1 online（单遍）**：一次遍历同时维护 `m` 与 `Σexp`，是 FlashAttention 的前置知识，故优先做。
- **v2 向量化**：在 v1 基础上用 `float4` 提升访存效率；若行宽不能整除 4，则做尾列处理。

## 参考规模

- 默认：`rows=4096, cols=4096`（fp32，约 64 MiB 输入）。
- 可选：`rows=16384, cols=1024`（宽行场景）。
- block 大小默认 256。

## 目录布局

```
softmax/
├── README.md   # 本文档：规划 + 结论总表
├── src/        # CUDA 内核与宿主入口（待实现）
├── triton/     # Triton 版本（第二阶段）
└── notes/      # 学习笔记（可选）
```

## 结论记录

（内核实现并基准后填写）

| 版本 | 形状 | 中位数 ms | 有效带宽 GB/s | max_err | 备注 |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

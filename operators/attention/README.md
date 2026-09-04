# Attention

单头、无 mask 的自注意力核心计算（缩放点积注意力）：

```text
S = Q·Kᵀ / sqrt(d)
P = softmax(S, 按 key 维)
O = P·V
```

Q/K/V 与输出 O 均为行主序 fp32，形状：`Q(M×d)`、`K(N×d)`、`V(N×d)`、`O(M×d)`。d 默认 64（后续可试 128）。

## 状态

| 阶段 | 版本 | 实现要点 | 状态 |
| --- | --- | --- | --- |
| CUDA | v0 | 朴素：每 query 行两次/三次读 K/V，全局内存主导，正确性基线 | 未开始 |
| CUDA | v1 | FlashAttention-1 风格：按 N 维分块 + online softmax，单遍产出 O | 未开始 |
| CUDA | v2 | v1 + 分块维数/block 规模调优 | 未开始 |
| Triton | t0 | FlashAttention-1 风格的 Triton 实现 | 规划中 |
| Triton | t1 | 与 CUDA 性能对照并记录 | 规划中 |

- 正确性判据：相对误差 ≤ 1e-5（参考实现同样使用 online 手法或朴素两次遍历均可，判据不变）。
- 指标：有效带宽 =（Q/K/V 读 + O 写）/ 中位耗时；同时记录 `ms`。

## 版本规划说明

- **v0 朴素**：直接计算完整 `M×N` 打分矩阵再 softmax，理解标准流程、暴露中间矩阵的显存开销，作为对照。
- **v1 Flash 风格**：不落中间矩阵，按 K/V 的 N 维分块迭代，块内用 online softmax 累积 running max/sum 与部分输出。复用 softmax 算子的单遍经验。
- **v2**：调整分块大小、是否双缓冲，给出完整的分块参数选择记录。

## 参考规模

- 默认：`M=N=4096, d=64`（fp32，单头）。
- block 默认 256；Flash 分块默认 `br=64, bc=64`（待调）。

## 目录布局

```
attention/
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

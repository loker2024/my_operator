# Reduce

对行主序 fp32 矩阵做归约求和。默认形态为**逐行求和**（row-sum），并规划扩展到列 / 全局归约：

```text
out[i] = Σ_j x[i, j]
```

规模关注 `rows × cols`，输出大小为 `rows`。与 Softmax 同属典型的**访存受限**算子；核心是 block 级归约原语（warp shuffle 树状归约 + shared memory 跨 warp 汇总），也是 FlashAttention / GEMM 内部归约的基础。

## 状态

| 阶段 | 版本 | 实现要点 | 状态 |
| --- | --- | --- | --- |
| CUDA | v0 | 每行一个 block，block 内归约（warp shuffle + shared memory），建立正确性基线 | 未开始 |
| CUDA | v1 | 每行多 block 协作（split 行）：各行写出 partial → 轻量二次归约汇总，覆盖长行 | 未开始 |
| CUDA | v2 | v1 + `float4` 向量化 + per-thread 多元素展开，提升访存吞吐 | 未开始 |
| CUDA | vx | 扩展变体：列归约 / 全归约，或一个 block 处理多行以摊薄调度 | 规划中 |
| Triton | t0 | 与 CUDA 同规格的 Triton 实现 | 规划中 |
| Triton | t1 | 与 CUDA 性能对照并记录 | 规划中 |

- 正确性判据：相对误差 ≤ 1e-3（见 `docs/benchmark-methodology.md`）。长序列 fp32 累加误差随项数增长（4096 项顺序累加已可达 ~1e-4 量级），且归约顺序随实现粒度变化，故参照 GEMM 放宽容差。
- 指标：有效带宽 =（输入读 + 输出写）/ 中位耗时。

## 版本规划说明

- **v0 每行一个 block**：行内用 grid-stride 方式由线程均摊读取，warp 内 shuffle 归约后经 shared memory 汇总各 warp 的部分和，写出 1 个浮点。直观、适合做正确性基线。
- **v1 split 行**：行宽很大、单 block 覆盖不足时，把一行拆给多个 block 各自产出 partial，再由轻量汇总阶段相加。提升并行度，是长行场景的必经优化。
- **v2 向量化**：在 v1 基础上用 `float4` 连续加载 + 展开循环（需行对齐约束与尾列处理），减少指令与访存事务；视情况让每个 block 处理多行。

## 参考规模

- 默认：`rows=4096, cols=4096`（fp32，约 64 MiB 输入）。
- 可选：`rows=16384, cols=1024`（长行 / 输出远小于输入的场景）。
- block 大小默认 256。

## 目录布局

```
reduce/
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

---
name: reduce-notes-doc
overview: 为 reduce 算子撰写一份新的讲解型学习文档（置于 operators/reduce/notes/），内容侧重算法推导与 v0→v7 逐步优化动机/取舍/陷阱的讲解，与现有 operators/reduce/README.md（规划+状态+结论表）互补；并按仓库惯例同步 CHANGELOG.md。
todos:
  - id: grounding-outline
    content: 对照 reduce 源码注释与 README、benchmark-methodology 梳理各版本事实、数字与约束，确定 notes/reduce.md 章节大纲
    status: completed
  - id: write-part1
    content: 撰写 operators/reduce/notes/reduce.md 前半：导读、问题与性能模型、两阶段归约与统一签名设计
    status: completed
    dependencies:
      - grounding-outline
  - id: write-part2
    content: 续写 v0–v7 逐版本推导、演进小结与瓶颈主线、陷阱清单、扩展方向章节
    status: completed
    dependencies:
      - write-part1
  - id: sync-docs
    content: 更新 reduce/README.md 目录布局注记，并在 CHANGELOG.md [Unreleased]/Added 追加带时间戳条目
    status: completed
    dependencies:
      - write-part1
      - write-part2
---

## 需求概述

新建一份与 `operators/reduce/README.md` **分工互补**的 reduce 算子原理讲解文档，交付到 `operators/reduce/notes/`（遵循顶层 README 推荐的算子目录布局）。用户明确要求该文档侧重「算法推导 + 逐步优化」的讲解形态，而非复述 README 已有的状态表与结论。

## 核心内容

- **导读**：说明与 README.md 的分工、建议配合源码（reduce.cu / reduce.cuh / main.cu / test.*）的阅读顺序。
- **问题与性能模型**：一维 fp32 整体求和的定义、访存受限分析与有效带宽口径（含算术强度为什么低、为什么带宽是主指标）。
- **设计约定**：两阶段归约（每 block 1 个部分和 → 调用方汇总）与统一签名 `ReduceKernel` 背后的设计动因（可复用同一测试驱动）。
- **v0→v7 逐版本推导**：每版按「上一版瓶颈 → 动机 → 设计（分歧 / bank 冲突 / 同步 / 归约结构 / 访存模式）→ 关键实现引用 → 正确性要点与约束 → 实测对比」展开。
- **演进小结**：瓶颈迁移主线（分歧 → bank 冲突 → 同步开销 → 归约循环 → shuffle → 全局访存 → 向量化加载）与收益递减规律，并保留 L2 带宽可比性警示的解读。
- **陷阱清单与扩展方向**：同步配对、volatile、shuffle mask 全 warp 收敛、float4 对齐、越界补 0、超配安全、n==0、fp32 容差等；以及 vx 行/列/全局归约与 Triton t0/t1 的规划衔接。

## 边界与约束

- 语言为简体中文 Markdown，文件路径 `operators/reduce/notes/reduce.md`（[NEW]）。
- 所有算法描述、代码引用、约束条件与实测数字**必须以当前源码注释、README 状态表与结论记录表、docs/benchmark-methodology.md 为准**，不得虚构或从旧版本照搬；引用代码片段须与 HEAD 一致。
- 不重写 / 不复制 operators/reduce/README.md 内容；文档形态为讲解与推导。
- 同步一致性：`operators/reduce/README.md` 目录布局小节补充 notes/ 条目并更正「notes 尚未创建」的注记；`CHANGELOG.md` 的 [Unreleased] > Added 追加一条带日期时间戳（`- YYYY-MM-DD HH:MM 描述`）的新条目。

## 实现方式

本任务为文档撰写，不改动任何 CUDA 代码与构建。写作以「源码注释 + 既有文档」为唯一事实源，先产出章节大纲并核对事实，再分两段撰写正文，最后做一致性同步。

### 事实源与引用规则

- `operators/reduce/src/reduce.cu` / `reduce.cuh`：各版本实现与注释（v5 模板内联于 cuh）；引用代码片段时标注文件与行号区间，且须与 HEAD 一致。
- `operators/reduce/src/main.cu`：kBlock=256、kKernels 表、A/B/C 场景表、GridFor、开关语义。
- `operators/reduce/README.md`：状态表、版本规划说明、目录布局、测试场景与结论记录表（实测数字只取此表，保留设备/构建/口径上下文与 L2 带宽警示）。
- `docs/benchmark-methodology.md`：容差 1e-3、中位数 + P5/P95、有效带宽公式等口径。
- 讲解主线按版本演进组织，避免与 README 的「表格式摘要」重复表述。

### 目录结构

```
my_operator/
├── CHANGELOG.md                          # [MODIFY] [Unreleased]/Added 追加带时间戳条目（- 2026-09-08 HH:MM 描述）
└── operators/reduce/
    ├── README.md                         # [MODIFY] 目录布局小节补 notes/reduce.md，并更正「notes/ 尚未创建」注记
    ├── notes/
    │   └── reduce.md                     # [NEW] 独立原理讲解文档（算法推导 + 逐步优化），本任务主交付物
    └── src/                              # 只读引用，不改动
```

### 文档大纲（notes/reduce.md）

1. 导读：与 README 的分工、配合源码的阅读路径。
2. 问题定义与性能模型：访存受限、有效带宽口径、算术强度分析。
3. 两阶段归约约定与统一签名 `ReduceKernel` 的设计考量。
4. v0–v7 逐版本推导：每节固定结构（上一版瓶颈 → 动机 → 设计 → 关键代码引用 → 正确性/约束 → 实测对比）。
5. 演进小结：瓶颈迁移主线与收益趋势；用少量 ```mermaid 流程图（如版本演进/瓶颈主线）辅助，不宜过多。
6. 陷阱清单：同步配对、volatile、shuffle 掩码与全 warp 收敛、float4 对齐、越界补 0、超配安全、n==0 时 grid≥1、fp32 长序列累加误差与容差 1e-3。
7. 扩展方向预告：vx（行/列/全局归约）、Triton t0/t1 对照。
# my_operator

面向 **CUDA 基础算子** 的学习与实验仓库。目标算子：Softmax、GEMM、Attention、Reduce（后续按需扩展）。

实现路线：

1. **第一阶段：CUDA 核心版本**。先用 CUDA 写出正确、可运行的内核，逐个做优化变体，并建立统一的正确性验证与性能基准口径。
2. **第二阶段：Triton 版本**。为同一算子用 Triton 实现，与 CUDA 版本做正确性对齐和性能对照。

各算子的详细状态、内核版本规划以其目录内 `README.md` 为准。

## 算子与实现状态

| 算子 | 说明 | CUDA 核心版 | Triton 版 | 目录 |
| --- | --- | --- | --- | --- |
| Softmax | fp32，行主序、逐行归一化 | 进行中（v0/v1/v2/v3/v4/v5 与 online-v0 完成） | 规划中 | `operators/softmax` |
| GEMM | fp32 SGEMM，`C = A(M×K) · B(K×N)` | 规划中 | 规划中 | `operators/gemm` |
| Attention | 单头、fp32、无 mask | 规划中 | 规划中 | `operators/attention` |
| Reduce | fp32 一维整体求和（标量），将扩展行/列/全局归约 | 完成（v0…v7） | 规划中 | `operators/reduce` |

状态说明：

- `规划中`：骨架与规划已建立，内核尚未实现。
- `进行中`：CUDA/Triton 核心版本已可运行，处于优化阶段。
- `完成`：核心版本通过正确性验证，并记录了基准结果。

## 目录结构

```
my_operator/
├── common/                     # 各算子共享的宿主工具（头文件库）
│   └── include/operator_common/
│       ├── cuda_check.h        # CUDA 错误检查 / 设备信息
│       ├── CpuTimer.h          # CPU 计时（std::chrono）
│       └── GpuTimer.h          # CUDA 事件计时
├── operators/
│   ├── softmax/                # Softmax：CUDA → Triton（src/ 已创建，v0/v1/v2/v3/v4/v5 与 online-v0 完成）
│   │   ├── README.md           # 规划 + 状态 + 结论记录
│   │   ├── CMakeLists.txt      # 出现 src/main.cu 后自动启用（各算子同一约定）
│   │   └── src/                # CUDA 实现：softmax.cuh/.cu、online_softmax.cuh/.cu、test.cuh/.cu、main.cu
│   ├── gemm/                   # SGEMM：CUDA → Triton（骨架，结构同 softmax）
│   ├── attention/              # Attention：CUDA → Triton（骨架，结构同 softmax）
│   └── reduce/                 # Reduce：CUDA 核心版 v0…v7 已实现
│       ├── README.md           # 规划 + 状态 + 结论记录
│       ├── notes/
│       │   └── reduce.md       # v0→v7 算法推导与优化讲解（学习文档）
│       ├── CMakeLists.txt
│       └── src/                # CUDA 实现：reduce.cuh/.cu、test.cuh/.cu、main.cu
├── docs/
│   └── benchmark-methodology.md # 正确性验证与性能基准的统一口径
├── demo/                       # 独立 CUDA 学习示例（不接入顶层 CMake）
├── CMakeLists.txt
├── CMakePresets.json           # 一条命令完成 Release/Debug 配置
├── CHANGELOG.md
└── README.md
```

单个算子的推荐目录布局（按实现进度逐步生成）：

```
operators/<name>/
├── README.md      # 算子说明、内核版本规划、状态与结论
├── src/           # 第一阶段：CUDA 内核与宿主入口（main.cu）
├── triton/        # 第二阶段：Triton 实现（Python）
└── notes/         # 学习笔记、优化记录（可选，也可统一放 docs/）
```

## 环境要求

- NVIDIA GPU + CUDA Toolkit（开发机为 RTX 4060 Laptop，Ada 架构 sm_89，CUDA 12.9）
- CMake ≥ 3.24
- Ninja（推荐）与支持 C++17 的编译器（gcc / clang / MSVC）

## 构建与运行

每个算子目录都自带 `CMakeLists.txt` 且已被顶层注册，但**只有目录里出现 `src/main.cu` 才会真正启用**（否则 configure 时自动跳过，不影响其他算子）。实现一个算子的流程：

```bash
# 1. 创建源码，例如 operators/softmax/src/main.cu（及其它 .cu/.cuh）
# 2. 重新 configure（新增/删除文件后必须重新执行）
cmake --preset release
# 3. 构建并运行（当前 reduce / softmax 已创建 src/main.cu，其余算子目标自动跳过）
cmake --build build --target softmax reduce   # 目标名 = 算子目录名
./build/operators/softmax/softmax
./build/operators/reduce/reduce
```

运行该可执行文件即执行该算子的「正确性验证 + 性能基准」，打印通过/失败与耗时统计（失败时返回非零退出码，便于脚本化）。

`demo/` 保留三个可单独用 `nvcc` 编译的学习示例：`helloWorld.cu` 用于最小 CUDA
启动验证，`demo_utils.cu` 演示向量加法、页锁定内存与统一计时工具，
`demo_stream.cu` 演示 grid-stride 循环与多 stream 传输/计算流水。它们不属于
顶层 CMake 的算子目标。

其他 GPU 上构建时，用 `native` 覆盖默认架构即可：

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
```

## 质量与基准口径

正确性验证与性能计时遵循统一口径，详见 `docs/benchmark-methodology.md`。核心约定：

- **先验证正确性，再计时**；fp32 默认容差按算子类型分别声明。
- 性能统计使用多次迭代的**中位数**，并给出 P5/P95 区间；带宽类算子报告**有效带宽**。
- 所有数值规模、块大小、预热/迭代次数在算子入口处集中配置，便于复现。

## 路线图

- [x] 仓库骨架与文档
- [ ] Softmax：CUDA 核心版本（v0/v1/v2/v3/v4/v5 与 online-v0 完成 → Triton 对照）
- [ ] GEMM：CUDA 核心版本（v0 → 优化变体）
- [ ] Attention：CUDA 核心版本（v0 → Flash 风格）
- [x] Reduce：CUDA 核心版本（v0 → v7 优化变体）与验证 / 基准记录   <!-- 全量回归为 8 个内核 × 10 个场景 = 80 项；默认仅跑 2 个正常场景，即 16 项；严格基准结论见 operators/reduce/README.md「结论记录」。 -->
- [ ] Softmax / GEMM / Attention：CUDA 核心版本 → 正确性验证与基准记录
- [ ] 逐个补充 Triton 版本，与 CUDA 对齐并对照性能

## 许可证

[MIT](LICENSE)

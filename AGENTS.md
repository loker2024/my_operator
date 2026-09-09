# AGENTS.md — my_operator 项目指南（供 agent 阅读）

面向在 `my_operator` 仓库内工作的 AI 代理/协作者。**先读本文件，再读 `docs/benchmark-methodology.md` 与对应算子目录的 `README.md`。** 项目对话与文档语言：简体中文。

## 0. 项目是什么

CUDA 基础算子学习与实验仓库（MIT）。目标算子：**Softmax、GEMM、Attention、Reduce**。两阶段路线：① CUDA 核心版（正确性基线 → 逐版本优化）→ ② Triton 版（与 CUDA 对齐、性能对照）。

**现状速览**（权威来源：根 `README.md`，改动后须同步）：

| 算子 | 状态 | 版本 |
|---|---|---|
| Reduce | 完成 | v0…v7，全量回归 80 项通过 |
| Softmax | 进行中 | v0/v1/v2 完成，后续 v3 等优化变体 |
| GEMM / Attention | 骨架规划中 | 无 `src/main.cu`，configure 自动跳过 |

## 1. 环境与构建

- 开发机：RTX 4060 Laptop（Ada，**sm_89**）、CUDA 12.9。其他 GPU：`-DCMAKE_CUDA_ARCHITECTURES=native` 覆盖。
- CMake ≥ 3.24 + Ninja + C++17。顶层 `CMakeLists.txt` 先以 CXX 初始化、找到 CUDAToolkit 再 `enable_language(CUDA)`；无 GPU/CUDA 时也能完成文档级 configure。

```bash
cmake --preset release          # 新增/删除 .cu 后必须重新 configure（GLOB CONFIGURE_DEPENDS）
cmake --build build --target <算子名>    # 目标名 = 算子目录名，如 reduce / softmax
./build/operators/<name>/<name> # 运行 = 正确性验证 + 性能基准；失败返回非零退出码
```

## 2. 目录与构建约定

- `common/` 共享宿主工具（头文件库）：`cuda_check.h`、`CpuTimer.h`、`GpuTimer.h`（GpuTimer 用 CUDA event，`StopMs` 前隐式同步）。
- 算子目录标准布局：`README.md`（规划+状态+结论记录）、`src/`（`.cuh`/`.cu` + `test.cuh/.cu` + `main.cu`）、`notes/`（学习笔记）、`triton/`（第二阶段，规划目录）。
- **CMake 门控**：算子目录 CMakeLists 判断 `src/main.cu` 是否存在——不存在则 configure 跳过。`src/*.cu` 以 `file(GLOB ... CONFIGURE_DEPENDS)` 自动纳入目标，**新增源文件不改 CMake**。
- `demo/` 独立示例，可单独用 nvcc 编译，不接入顶层 CMake、不属于算子目标。
- `docs/benchmark-methodology.md` 定义统一测法；`docs/` 不放置个算子结论。

## 3. 开发工作准则（硬性）

1. **先正确，后性能**。每个内核版本必须先通过正确性验证，再做计时与基准。声称"完成"必须有验证记录。
2. **版本签名保持一致**，使同一套测试驱动可复用（Reduce 统一为 `ReduceKernel`：每 block 只产出 1 个部分和到 `output[blockIdx.x]`，最终标量由调用方对 `output[0,grid)` 轻量求和）。
3. **新版本必须注册进被测内核表**（`src/main.cu`），并同步算子 README 状态表与规划说明。
4. 参考实现放**主机端**，朴素、易读的循环，与 GPU 内核同文件测试驱动。
5. 输入可复现：Reduce 用确定性 `i % 1000` 周期序列；其他算子可用固定种子伪随机，避免极端值掩盖错误。
6. **Surgical changes**：只改任务相关代码；既有死代码只报告不删；同步清理自己改动引入的无用 import/变量。

## 4. 正确性验证口径

逐元素相对误差判据（`docs/benchmark-methodology.md`）：

| 算子 | 容差 | 说明 |
|---|---|---|
| Softmax / Attention | ≤ 1e-5 | fp32 运算顺序简单，从严 |
| GEMM | ≤ 1e-3 | 归约顺序/累加粒度不同 |
| Reduce（行求和等） | ≤ 1e-3 | 长序列累加误差随项数增长 |

判据式：`err = |gpu-ref| / max(|ref|, 1e-30)`，`max_err ≤ tol` 且无 NaN/Inf 为通过。失败时：逐元素算子打印最大误差位置与前后元素；标量归约打印 CPU 参考值、GPU 结果、相对误差。

**测试场景分层**（`main.cu` 内开关，默认只跑正常流程 2 场景/16 项）：
- 正常流程：大规模对齐、尾部非对齐；
- 边界条件：n=1、n=block±1、n=block、n=2·block−1 等（覆盖整除/余 1/2/3 尾部路径）；
- 异常/健壮性：n=0 空输入（期望和=0）、grid 超配冗余 block（须补 0 不影响结果）、非法参数（`n<0`/`grid<1` 判 FAIL）。

## 5. 性能基准口径

- 仅 **Release** 下计时；GPU 计时用 CUDA event；内核轮间不加额外同步（体现真实吞吐）。
- **预热**消除冷启动；汇报**中位数**（主指标）+ **P5/P95** 区间，**不用平均值**。
- 指标口径：延时型报 ms+吞吐；归约/Softmax 报**有效带宽**（输入读+输出写）/时间；GEMM 报 TFLOPS（2MNK/t）。
- 记录模板必含：算子/版本、形状、设备、构建参数、预热/迭代次数、中位数+P5/P95、指标、正确性结果（见 benchmark-methodology.md §3.4）。
- 对比限**同机同构建**；跨机器只比数量级。
- 默认迭代建议：开发阶段预热 1 次测 100 轮；严格对比预热 1000 次测 21 组×10000 次。

## 6. 代码风格与注释

- `.clang-format`：基于 Google；IndentWidth 2；ColumnLimit 100；指针靠左；`BreakBeforeBraces: Attach`；C++17。**改代码后保持该风格**。
- 注释规范：模块/函数/参数/返回值/启动约束/注意事项齐全；**声明与实现注释去重**；推导细节与理由放入算子 README / notes，不堆在源码。
- 已知技术约束：`-rdc=false` 下跨翻译单元引用 `__global__` 模板特化已被 nvcc 弃用 → 模板内核（如 `reduce_v5<BLOCK_SIZE>`）**内联定义在 `.cuh`**，`main.cu` 显式实例化注册。

## 7. 文档与版本管理

- 算子 README 维护「版本 × 性能」总表，表头含形状/设备/构建，避免裸数字；notes 学习文档与 README 互补（README 记结论，notes 讲推导）。
- **每完成一个版本变更，同步更新根 README（如适用）、算子 README、CHANGELOG.md**。
- `CHANGELOG.md`：Keep a Changelog 格式 + SemVer；条目带时间戳与变更说明。
- git 提交：**Conventional Commits + 中文 scope 描述**，如 `feat(reduce): 新增 v6 两级 warp shuffle 归约内核`、`docs(reduce): update v2 ...`、`refactor(reduce): 测试场景开关化`。
- 遇到不明确的规范，以仓库内文件为准；发现文件间事实不一致（如 README 与 CHANGELOG），先报告，经确认后同步修正。

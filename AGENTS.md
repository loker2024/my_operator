# AGENTS.md — my_operator 项目指南（供 agent 阅读）

面向在 `my_operator` 仓库内工作的 AI 代理/协作者。**先读本文件，再读 `docs/benchmark-methodology.md` 与对应算子目录的 `README.md`。** 项目对话与文档语言：简体中文。

## 0. 项目是什么

CUDA 基础算子学习与实验仓库（MIT）。目标算子：**Softmax、GEMM、Attention、Reduce**。两阶段路线：① CUDA 核心版（正确性基线 → 逐版本优化）→ ② Triton 版（与 CUDA 对齐、性能对照）。

**现状速览**（权威来源：根 `README.md`，改动后须同步）：

| 算子 | 状态 | 版本 |
|---|---|---|
| Reduce | 完成 | v0…v7，全量回归 80 项通过 |
| Softmax | 进行中 | v0/v1/v2/v3/v4/v5 与 online-v0/v1/v2/v3/v3_false/v4 完成，全量回归 240 项通过 |
| GEMM | 进行中 | v0、v1、v2 完成（入口固定 `512×512×512`，4/4 通过） |
| Attention | 骨架规划中 | 无 `src/main.cu`，configure 自动跳过 |

## 1. 环境与构建

- 开发机：RTX 4060 Laptop（Ada，**sm_89**）、CUDA 12.9。其他 GPU：`-DCMAKE_CUDA_ARCHITECTURES=native` 覆盖。
- **构建只在 WSL（Ubuntu-24.04）内做**：CUDA / gcc 与构建缓存均为 Linux 路径，Windows 侧 CMake 被顶层 `CMakeLists.txt` 的 `CMAKE_HOST_WIN32` 守卫拒绝（此前 Windows CMake 读到缓存里的 `/mnt/d/.../ninja` 会报 `no such file or directory`）。
- Ninja 用 WSL 原生版（`apt install ninja-build`）；缺装时 CMake 会经 interop 抓到 Windows 的 `ninja.exe`，把 `/mnt/d/...` 写进各构建目录缓存的 `CMAKE_MAKE_PROGRAM`。
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
- `demo/` 独立示例，可单独用 nvcc 编译，不接入顶层 CMake、不属于算子目标；**编译产物不入库**（`.gitignore` 忽略 `/demo/*`，仅放行 `.cu` / `.cuh`）。
- `scripts/` 仓库级 Python 工具（跨算子复用）：`plot_kernel_perf.py` 把 bench CSV 画成性能曲线，只要求 CSV 含 `label,size,gflops` 列（GEMM `--bench` 直接产出该格式）；依赖见 `scripts/requirements.txt`。扫描产物（CSV + 图）落在各算子目录的 `bench/<时间戳>/` 下、每次扫描一个目录，并入库作结论证据；README 配图直接引用对应时间戳目录里的 PNG。
- `docs/benchmark-methodology.md` 定义统一测法；`docs/` 不放置个算子结论。

## 3. 开发工作准则（硬性）

1. **先正确，后性能**。每个内核版本必须先通过正确性验证，再做计时与基准。声称"完成"必须有验证记录。
2. **版本签名保持一致**，使同一套测试驱动可复用（Reduce 统一为 `ReduceKernel`：每 block 只产出 1 个部分和到 `output[blockIdx.x]`，最终标量由调用方对 `output[0,grid)` 轻量求和）。
3. **新版本必须注册进被测内核表**（`src/main.cu`）：注册项的 `RowMap` / `smem_bytes` 必须与内核**实际**行映射一致，且源码自带符合 §6 的注释（含启动约束），再同步算子 README 状态表与规划说明。
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
- 默认迭代建议：开发阶段预热 1 次测 100 轮；严格对比预热 100 次测 21 组×1000 次（面向本机 RTX 4060 Laptop 下调采样量，见 `docs/benchmark-methodology.md` §3.2）。

## 6. 代码风格与注释

- `.clang-format`：基于 Google；`IndentWidth 4` + `UseTab: ForIndentation`（缩进是真正的 Tab）；ColumnLimit 100；指针靠左；`BreakBeforeBraces: Attach`；C++17。**改代码后保持该风格**。
- 完整规则见 [`.codebuddy/rules/注释规则.mdc`](.codebuddy/rules/注释规则.mdc)。以下要求是所有改动的合并门槛：
  1. **只写“做什么”**：新增 `.cu` / `.cuh` 一律用一两句文件头说明模块职责；在函数体的非直观操作阶段标注数据装载、坐标映射、归约、边界处理或写回等实际行为。声明处默认不写大段模板；仅当名称和类型无法表达输入、输出或副作用时补一条简短契约。
  2. **配置写在入口**：在 `main.cu` 或等价启动 / 注册位置，用一行说明当前实际采用的 block、grid、动态共享内存或测试配置；不要把同一信息重复到每个声明。
  3. **“为什么”留在文档**：算法推导、性能原理与取舍、数值误差依据、硬件背景、历史数据和版本对比统一写入算子 README、`notes/` 或 `docs/`，不写进源码注释。
  4. **注释与行为同步**：代码行为变更时同步修改相关注释，特别是边界处理、数据布局、同步位置和启动配置；与当前任务无关的旧注释不顺带重写。
- 已知技术约束：`-rdc=false` 下跨翻译单元引用 `__global__` 模板特化已被 nvcc 弃用 → 模板内核（如 `reduce_v5<BLOCK_SIZE>`）**内联定义在 `.cuh`**，`main.cu` 显式实例化注册。

## 7. 文档与版本管理

- 算子 README 维护「版本 × 性能」总表，表头含形状/设备/构建，避免裸数字；notes 学习文档与 README 互补（README 记结论，notes 讲推导）。
- **每完成一个版本变更，同步更新根 README（如适用）、算子 README、CHANGELOG.md**。
- `CHANGELOG.md`：Keep a Changelog 格式 + SemVer；条目带时间戳与变更说明。
- git 提交：**Conventional Commits + 中文 scope 描述**，如 `feat(reduce): 新增 v6 两级 warp shuffle 归约内核`、`docs(reduce): update v2 ...`、`refactor(reduce): 测试场景开关化`。
- 遇到不明确的规范，以仓库内文件为准；发现文件间事实不一致（如 README 与 CHANGELOG），先报告，经确认后同步修正。

# Changelog

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)；
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Fixed

- 2026-09-08 13:31 根 `README.md` 文档同步（事实校正）：Reduce 的 CUDA 核心版状态由「进行中（v0…v7）」改为「完成（v0…v7）」；目录结构树按仓库实际拆分 reduce（含 `notes/reduce.md`、`src/`），并标明 softmax / gemm / attention 为 src/ 尚未创建的骨架；构建示例目标由未启用的 softmax 改为当前唯一启用的 reduce；路线图勾选 Reduce 的验证 / 基准记录，收窄剩余项为 Softmax / GEMM / Attention。
- 2026-09-08 文档与注释同步：根 README 更新 Reduce 的 v0…v7 与 80 项全量回归状态，补充未接入 CMake 的 `demo/` 说明；基准方法改为反映 Reduce 实际使用的确定性 `i % 1000` 输入；修正 `main.cu` 中 reduce_v5 模板实例化位置的过时说明。

### Added

- 2026-09-04 14:18 仓库骨架与工程文档：`README.md`、`CHANGELOG.md`、`LICENSE`（MIT）、`.gitignore`、`.clang-format`。
- 2026-09-04 14:18 CMake 构建框架：顶层 `CMakeLists.txt`、`CMakePresets.json`（Release/Debug，Ninja），`common/` 头文件库（CUDA 错误检查与计时工具）。
- 2026-09-04 14:18 `docs/benchmark-methodology.md`：统一正确性验证与性能基准口径。
- 2026-09-04 14:18 算子目录骨架与规划：`operators/{softmax,gemm,attention}` 各自的 `README.md`。
- 2026-09-04 14:18 新增 `operators/reduce`（`README.md` + `CMakeLists.txt`），顶层统一注册，目标算子扩展为 Softmax / GEMM / Attention / Reduce。
- 2026-09-04 14:18 算子构建约定：每个算子目录自带 `CMakeLists.txt`，顶层统一注册；出现 `src/main.cu` 后自动参与构建，新增 `.cu` 无需改 CMake。
- 2026-09-06 12:58 `operators/reduce`：CUDA 核心版实现 —— `reduce_cpu` 主机参考、`reduce_v0`（交错寻址共享内存树形归约）与 `reduce_v1`（连续寻址，消除 warp 内分歧）内核；配套可复用测试驱动（`test.cuh/.cu`）与入口（`main.cu`），覆盖正常 / 边界 / 异常（空输入、grid 超配等）三类场景，两内核 18 项测试全部通过。
- 2026-09-06 12:58 `operators/reduce`：为 v0/v1 内核补齐模块、函数与关键逻辑注释（参数 / 返回值 / 启动约束 / 注意事项），并同步文档（`CHANGELOG.md`、顶层与算子 `README.md`）。
- 2026-09-06 13:09 `operators/reduce`：测试入口 `main.cu` 新增两个开关（函数参数形式，默认关闭）—— `enable_boundary` 控制边界条件与异常 / 健壮性场景（默认只跑正常流程 4 项，开启后 18 项全量回归）、`strict_benchmark` 控制严格性能口径（透传给 `test_reduce_kernel`）；重复场景循环收敛为 `RunScenarios`，场景表长度编译期推导。
- 2026-09-06 14:49 `operators/reduce`：`reduce_v3`（每线程 2 元素展开）由实验性实现正式接入——签名统一为 `ReduceKernel`（`const float*`）并声明加入 `reduce.cuh`，注册进 `main.cu` 被测内核表；`GridFor` 改为按内核“每线程元素数”计算覆盖 grid（v3 覆盖同一 `n` 所需 block 减半）；同步实现注释、`test.cuh` 注释与算子 `README.md`（状态表 / 版本规划）。
- 2026-09-07 10:58 `operators/reduce`：新增并接入 `reduce_v4`（每线程 2 元素 + 末 warp 展开归约，volatile 共享内存收尾省去 5 轮 `__syncthreads`）——修正草稿的段基址与越界判定，声明加入 `reduce.cuh`、注册进 `main.cu` 被测内核表，全量回归 5 内核 × 9 场景 45 项全部通过；`reduce_v4` 正常流程严格基准 251.5 GB/s（n=2^20）。同时精简算子源码注释：`cu`/`cuh` 声明与实现注释去重、删除冗余推导细节（并入算子 `README.md` / `docs`），并同步 `test.cuh`/`test.cu`/`main.cu` 注释。
- 2026-09-07 14:23 `operators/reduce`：新增并接入 `reduce_v5`（v4 + 编译期常量化 block 尺寸）——规整草稿实现并补齐模块/行内注释；因 `-rdc=false` 下跨翻译单元引用 `__global__` 模板特化已被 nvcc 弃用，`reduce_v5` 模板定义整体内联进 `reduce.cuh`（`main.cu` 以 `reduce_v5<kBlock>` 注册进被测内核表，同时同步 `test.cuh` 注释）；全量回归 6 内核 × 9 场景 54 项全部通过。算子 `README.md`：状态表 v5 标记完成、原 `float4` 向量化规划顺延为 v6，结论记录表回填 v5 严格基准（n=2^20 对齐 272.3 GB/s，同场相对 v4 约 +4.6%）；顶层 `README.md` Reduce 状态同步为 v0…v5。
- 2026-09-07 15:55 `operators/reduce`：新增并接入 `reduce_v6`（每线程 2 元素 + 两级 warp shuffle）——规整草稿实现并补齐精简注释（移除实验期遗留的 clang 内部头文件 include），声明加入 `reduce.cuh`、注册进 `main.cu` 被测内核表（同步 `test.cuh` 注释与 block 约束说明），全量回归 7 内核 × 9 场景 63 项全部通过。算子 `README.md`：状态表 v6 标记完成、原 `float4` 向量化规划顺延为 v7，结论记录表回填 v6 严格基准（n=2^20 对齐 339.68 GB/s，同场 v4/v5 复测 310.89/331.46 → 相对 v5 约 +2.5%）；顶层 `README.md` Reduce 状态同步为 v0…v6。
- 2026-09-07 16:22 `common/cuda_check.h`：`PrintDeviceInfo` 增加显存时钟（kHz）/ 位宽（bit）/ 理论峰值带宽输出（`cudaDeviceProp` 无带宽字段，按 `2 × memoryClockRate × busWidth/8` 估算并注明需跑 benchmark 才能测得实际带宽）。
- 2026-09-07 16:22 `operators/reduce/README.md`：结论记录表补带宽可比性警示——n=2^20 工作集仅 ~4 MiB、远小于本机 32 MB L2（预热后输入驻留 L2），有效带宽反映片上 L2 命中带宽而非显存物理带宽，v5/v6 超过 GDDR6 理论峰值 256 GB/s 系缓存命中所致，不违反物理上限；要逼近/验证显存带宽需改用远大于 L2 的规模。
- 2026-09-07 16:36 `operators/reduce`：新增并接入 `reduce_v7`（v6 两级 warp shuffle + `float4` 向量化加载与 grid-stride 扫描）——规整草稿实现（签名统一为 `ReduceKernel` 的 `const float*`，块内两次 shuffle 归约复用 `warpReduceSum`），补齐内核/接口注释，声明加入 `reduce.cuh`、注册进 `main.cu` 被测内核表（每线程元素数 4，推荐 grid = `n/(4*block)` = 1024），边界场景补 `n=block-2` 覆盖 `float4` 尾部余 2 路径；全量回归 8 内核 × 10 场景 80 项全部通过。算子 `README.md`：状态表 v7 标记完成，结论记录表回填 v7 严格基准（n=2^20 对齐 412.16 GB/s，同场 v5/v6 复测 329.99/341.39 → 相对 v6 约 +20.7%）；顶层 `README.md` Reduce 状态同步为 v0…v7。
- 2026-09-08 13:09 `operators/reduce`：新增讲解型学习文档 `notes/reduce.md`（算法推导 + 逐步优化讲解）——v0→v7 每版按「瓶颈 → 动机 → 设计 → 关键实现 → 约束 → 实测」展开，含问题性能模型、两阶段归约设计说明、演进小结与瓶颈迁移主线、陷阱清单、扩展方向，与算子 `README.md`（规划 + 状态 + 结论表）分工互补；同步 `README.md` 目录布局并更正「notes/ 尚未创建」注记。
- 2026-09-08 14:10 `operators/softmax`：实现并接入 `softmax_v0`（每行一个 block，行内列维由 blockDim 线程以 stride 协同遍历；共享内存树形归约两次——先求行最大、再求 Σexp，max-shift 数值稳定，写回时第三次读行重算 exp，作正确性基线）与 CPU 参考 `softmax_cpu`，补齐内核/接口/模块注释（移除草稿遗留的 clang 内部头文件 include）；新增仿 reduce 的可复用测试驱动 `test.cuh/.cu`（函数指针统一驱动、逐元素相对误差 ≤ 1e-5、跳过 `|ref|<1e-30`、开发 1+100 / 严格 1000+21×2000 两档采样）与入口 `main.cu`（A 正常 / B 边界 / C 异常——空矩阵与空行，默认 2 项、全量回归 10 项全部通过）。算子 `README.md`：状态表 v0 标记完成、目录布局与测试章节补齐，结论记录表回填 v0 严格基准（4096×4096 0.7462 ms / 179.87 GB/s，16384×1024 0.8267 ms / 162.36 GB/s，max_err≈1.2e-6）；顶层 `README.md` Softmax 状态同步为进行中（v0 完成）。
- 2026-09-08 15:10 `operators/softmax`：新增并接入 `softmax_v1`（每行一个 block，行内 `blockDim.x` 线程以 stride 协同遍历，行最大与 Σexp 各经一次共享内存折半树形归约，写回时第三次读行重算 exp）——规整草稿实现（补 `row >= M` 越界空转守卫、修正 smem 写入后的同步顺序），`softmax.cu` 注释精简为行内/短注释并移除草稿遗留的 clang 内部头文件 include，声明加入 `softmax.cuh`（含启动 grid 与动态共享内存契约）；测试驱动 `test.cuh/.cu` 增参 `grid`/`smem_bytes`（v0 每线程一行 `grid=ceil(rows/block)`、无共享内存；v1 每行一个 block `grid=rows`、smem=`blockDim.x*sizeof(float)`，由 `main.cu` 的 `RowMap` 推出），`main.cu` 注册 v0/v1 并补行宽维边界场景；全量回归 2 内核 × 13 场景 26 项全部通过。算子 `README.md`：状态表 v1 标记完成、目录/测试章节补双版本启动口径、结论表回填同场开发采样（v0 4096×4096 4.9560 ms/27.08 GB/s vs v1 0.6816 ms/196.93 GB/s，max_err 5.478e-06 → 1.231e-06）；顶层 `README.md` Softmax 状态同步为进行中（v0/v1 完成）。
- 2026-09-08 14:38 `operators/softmax`：纠正上一轮对 v0 的改动——v0 恢复为最初实现（每线程处理一行，行内串行三遍：求行最大 → 累加 Σexp → 归一化写回；无共享内存/同步，算法代码原样保留，仅移除 nvcc 无法解析的 clang 内部头文件 include 并补充注释），不再采用“每行一个 block + 共享内存树形归约”。测试驱动/入口随映射方式调整：grid = ceil(rows/block)、严格档迭代与 reduce 一致（1000 预热 + 21 组 × 10000 次）、去掉共享内存启动配置；边界场景改为围绕行数在 block 线程覆盖边界（block±1、恰满载、末 block 余 1 行）。全量回归 10/10 PASS；默认形状 max_err 5.478e-06（4096×4096）/ 2.537e-06（16384×1024），仍满足 1e-5 容差但已同量级（fp32 串行累加误差随行宽增长，记为 v1/v2 优化动机）。算子 `README.md` 同步 v0 描述与开发采样实测（4.87 ms / 27.56 GB/s、4.24 ms / 31.69 GB/s，max_err 同场实测）。

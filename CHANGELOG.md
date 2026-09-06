# Changelog

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)；
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

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

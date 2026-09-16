# MEMORY.md — 长期记忆（my_operator）

## 开发环境硬件规格（2026-09-15 实测，cudaDeviceGetAttribute）

- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，compute capability 8.9（sm_89），驱动 595.71，CUDA 12.9。
- 共享内存：每 block 默认上限 48 KB（49152 B）；动态 opt-in 上限 99 KB（101376 B，需 `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`）；每 SM 总量 100 KB（102400 B）。
- SM 数 24；每 block 最大线程 1024；每 SM 寄存器 65536。
- L2 32 MB；全局内存 8187 MB。
- maxGridSize = (2147483647, 65535, 65535)；maxThreadsDim = (1024, 1024, 64)。

## 代码注释规范（2026-09-16 起，权威口径）

- 权威文件是 `.codebuddy/rules/注释规则.mdc`（全仓代码注释规则）：源码注释只说明「当前做了什么、数据如何流动、边界行为是什么」。
- 源码中不得保留：设计原因、算法推导、性能原理（合并访问 / bank conflict / 占用率 / 寄存器溢出）、数值误差与容差依据、历史性能数据与版本对比、API 通用教程。
- 写法：文件头 1–2 句；声明处默认不写「作用 / 参数 / 返回值 / 启动约束 / 注意事项」模板，必要时一条简短契约；函数体只在不直观阶段前写短注释；启动配置（block / grid / 动态共享内存）在 `main.cu` 等入口处一行写明；「为什么」类信息归 README / notes / docs。
- 注：`AGENTS.md` §6 已于 2026-09-16 同步为该口径；`operators/gemm` 全部源码已按新规则整理，其他算子目录（reduce / softmax / attention）的老注释尚未整理。

## 工作方式经验

- **同一文件的多处编辑必须串行发起**：并行发起会随机丢失部分改动，甚至破坏代码（2026-09-16 在 `operators/gemm/src/main.cu` 上实测：末尾 `return RunValidation();` 被截断为 `\tr`、常量区注释替换未生效）。
- 批量改注释后必须复核落盘结果：`git diff --stat` 看范围、用关键短语 `grep -rqF` 逐条核对、必要时 `git diff | cat` 目检；不能只信工具的成功回执（其行数统计也不可靠）。

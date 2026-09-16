# Changelog

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)；
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added

- 2026-09-16 `operators/gemm`：新增并接入共享内存分块 + 寄存器行分块的 SGEMM v3
  - `include/sgemm_v3.cuh`：新增 `sgemm_v3<32,32,32,8>` 模板内核。一个 block 使用 128 个线程协作装载 A / B 的 `32×32` tile；每线程累加同一列连续 8 行的结果，M/N/K 尾块通过加载补 0 和写回保护覆盖。
  - `main.cu`：注册 v3（`grid=(ceil(N/32),ceil(M/32))`、`block=(128,1)`、静态共享内存 8 KiB、动态共享内存 0 B），并增加 `--v3-boundary`，用于只验证 `510×514×518` 的非对齐尾块场景。
  - 验证：Release 构建通过；默认入口 5/5 PASS；v3 边界专项 PASS（`max_err=1.288e-06`）。
  - 扫描产物：`--bench` 30 点全部成功，CSV / PNG 位于 `operators/gemm/bench/20260916201534/`；512³ v3 为 2170.1 GFLOP/s，约为同场 v2 的 1.93×。

- 2026-09-15 19:30 `operators/gemm`：新增共享内存分块版 SGEMM v2 并接入测试
  - `include/sgemm_v2.cuh`：新增 `sgemm_v2<BLOCKSIZE>` 模板内核（内联在头文件 —— `-rdc=false` 下跨翻译单元引用 `__global__` 模板特化已被 nvcc 弃用）。一个 block 以 `BLOCKSIZE²` 个线性线程覆盖 `BLOCKSIZE×BLOCKSIZE` 输出 tile，k 方向按 tile 把 A / B 搬进静态共享内存（BLOCKSIZE=32 时 8 KiB，动态共享内存仍为 0 B）；M / N 越界线程空转，K 非 BLOCKSIZE 倍数时尾块补 0，K == 0 时输出写 0。启动约束 `block=(BLOCKSIZE²,1)`、`grid=(ceil(M/BLOCKSIZE),ceil(N/BLOCKSIZE))`。
  - 修正写回位置：C 的写回原先放在 k 循环内，512³ / BLOCKSIZE=32 时每线程多写 15 次全量 C（约 15 MB 额外全局写）；移到循环外后只在末尾写一次。
  - 补齐注释：文件头说明（模板内联的原因、tile 复用带来的访存量变化）与函数注释（作用 / 参数 / 返回值 / 启动约束 / 注意事项），满足新增 `.cuh` 不得裸提交与不得缺启动约束的要求。
  - 清理：`src/sgemm_v2.cu` 原为 0 字节空文件，改为与 `sgemm_v1.cu` 一致的注释占位文件（写明实现内联在 `.cuh`、由 `main.cu` 显式实例化，以及为何不需要 CMake 排除）；`operators/gemm/CMakeLists.txt` 回滚为 `add_executable(gemm ${GEMM_SOURCES})`（去掉重复列出 `src/sgemm_v2.cu` 的改动 —— GLOB 已覆盖 `src/*.cu`）。
  - `main.cu`：包含 `sgemm_v2.cuh`，以 `BLOCKSIZE=32` 实例化 `sgemm_v2<32>` 并注册进 `kKernels`（`block=(1024,1)`、tile `32×32`、smem 0 B）。
  - 回归：`./build/operators/gemm/gemm` 由 3 项变 4 项、4/4 PASS；v2 在 `512×512×512` 为 `max_err=1.275e-06`、中位 `0.2660 ms` / `1.009 TFLOPS`（复测 0.2609 ms），同场 v1 `0.3750 ms` / `0.716 TFLOPS`、v0 `1.8980 ms` / `0.141 TFLOPS`、cuBLAS `0.0666 ms` / `4.028 TFLOPS`。
  - 形状复核：临时加入 `513×511×509`、`33×33×33`、`1×1×1` 三个场景，8 内核·场景组合 16/16 PASS（尾块与退化尺寸均通过），验证后恢复默认单场景 `512×512×512`。
  - 扫描产物：`--bench` 24 点全部成功，CSV / PNG 落在 `operators/gemm/bench/20260915192408/`；v2 相对 v1 的加速为 512³ 1.49×、1024³ 1.65×、2048³ 1.65×、4096³ 1.60×，约为 cuBLAS 的 1/5。
  - 文档同步：`operators/gemm/README.md`（状态表与规划说明改为共享内存分块、原「v2 向量化 / 每线程多元素」顺延为 v3、结论记录新增 v2 行与含 v2 的多尺寸曲线章节）、根 `README.md`（GEMM 进度改为 v0、v1、v2 完成）。

### Changed

- 2026-09-16 `scripts`：统一 Python 绘图环境为项目根目录 `.venv/`
  - `.venv/bin/python` 已安装 `scripts/requirements.txt` 声明的 matplotlib 与 numpy；根 README、GEMM 绘图命令与 `docs/python-environment.md` 记录初始化、更新和调用方式。

- 2026-09-16 10:55 `operators/gemm`：按 `.codebuddy/rules/注释规则.mdc` 整理 gemm 其余代码文件的注释
  - `include/sgemm_v0.cuh` / `src/sgemm_v0.cu`：声明处删去启动约束（block / grid / 0 B 动态共享内存）与 1e-3 容差说明，文件头收敛为「行主序 + 每线程一个输出元素」；函数体补坐标映射行尾注释与 K 方向累加说明。
  - `include/sgemm_v1.cuh`：文件头删去模板内联原因与 warp 访问模式叙述（warp 行为移入 README）；契约收敛为 `TILE_SIZE` 语义 + 「每 block 必须起满 `TILE_SIZE²` 个线程」。
  - `src/sgemm_v1.cu`：占位注释由 6 行收敛为 3 行（保留 CMake 排除现状与启用前提）。
  - `include/sgemm_reference.cuh` / `src/sgemm_reference.cu`：文件头精简为职责一句话，删去「不作为性能基线」等定位说明。
  - `include/test.cuh`：删去扫描模式的策略与背景长注（逐点校验的代价、固定迭代的耗时），保留 `strict_benchmark` 语义与返回契约；迭代自适应细节由实现承担。
  - `src/test.cu`：输入生成、cuBLAS 错误处理、扫描缓冲、自适应计时与 bench 契约注释改为「做什么」表述；补空输出、内核参数打包两处行为注释；原因类信息（确定性正数输入、0 填充、自适应迭代）移入 README。
  - `src/main.cu`：文件头删去构建 / 运行命令与两条路径详解；启动配置、CSV 落点、管道用法等注释收敛（写入失败退回标准输出的行为保留）。
  - `operators/gemm/README.md`：承接移出的文档类信息 —— 确定性正数输入的原因、v1 的 warp 访问模式、扫描缓冲 0 填充的原因。
  - 验证：`clang-format --dry-run --Werror` 对 11 个源码文件全部干净；`cmake --build build --target gemm` 通过；`./build/operators/gemm/gemm` 4/4 PASS（v0 2.1400 ms、v1 0.3896 ms、v2 0.2458 ms / 1.092 TFLOPS、cuBLAS 0.0624 ms）。

- 2026-09-16 10:48 `operators/gemm`：按 `.codebuddy/rules/注释规则.mdc` 重写 sgemm_v2 源码注释
  - `include/sgemm_v2.cuh`：文件头与函数声明处的「作用 / 参数 / 返回值 / 启动约束 / 注意事项」模板收敛为文件头两行说明 + 函数前两行契约（`TILE_SIZE` 语义与「每 block 必须起满 `TILE_SIZE²` 个线程」）；删去模板内联原因、共享内存字节数、`__syncthreads` 必要性、warp 访问模式与 1e-3 容差等推导 / 性能 / 背景叙述（启动配置由 `main.cu` 的记录承担）；函数体按「做什么」重写：拆分行列坐标、指针移到 tile 左上角、装载 tile 越界补 0、共享内存乘加、指针前进、合法范围写回，并删除 `A / B / C` 指针的行尾坐标注释（与统一注释重复）。
  - `src/sgemm_v2.cu`：占位注释由 6 行收敛为 2 行，`-rdc=false` 弃用原因与 CMake 排除说明移出源码。
  - `operators/gemm/README.md`：「版本规划说明」的 v2 条目补记模板定义内联在 `.cuh`、由 `main.cu` 显式实例化的原因（`-rdc=false` 下跨翻译单元引用 `__global__` 模板特化已被 nvcc 弃用），承接从源码移出的文档类信息。
  - 验证：`cmake --build build --target gemm` 通过、`git diff` 核对为纯注释 / 文档改动；`./build/operators/gemm/gemm` 4/4 PASS（v2 `max_err=1.275e-06`、中位 0.2667 ms / 1.007 TFLOPS）。

- 2026-09-15 19:47 `operators/gemm`：模板参数 `BLOCKSIZE` 重命名为 `TILE_SIZE`
  - `include/sgemm_v1.cuh` / `include/sgemm_v2.cuh`：模板参数与全部注释改用 `TILE_SIZE` —— 该常量是输出 tile 边长（`TILE_SIZE×TILE_SIZE`），同时决定静态共享内存大小（`2·TILE_SIZE²·4` B）、k 方向步长与 grid 划分；原名容易被读成「block 线程数」，而每 block 线程数其实是 `TILE_SIZE²`（v1 / v2 下为 1024）。纯重命名，语义与启动配置不变。
  - `main.cu`：`kSgemmV1BlockSize` / `kSgemmV2BlockSize` → `kSgemmV1TileSize` / `kSgemmV2TileSize`；`kSgemmV*Threads`（= `TILE_SIZE²`）与注册项的 `block=(1024,1)`、tile `32×32`、smem 0 B 均不变，内核描述字符串同步为 `TILE_SIZE=32`。
  - `src/sgemm_v1.cu` / `src/sgemm_v2.cu`：占位注释里的模板签名同步。
  - 文档同步：`operators/gemm/README.md`（3 处 `BLOCKSIZE` 改为 `TILE_SIZE`）。`CHANGELOG.md` 的历史条目保留原名，记录的是当时的命名。
  - 回归：`cmake --build build --target gemm` 通过，`./build/operators/gemm/gemm` 4/4 PASS（v2 `max_err=1.275e-06`、中位 `0.3034 ms` / `0.885 TFLOPS`；v1 `0.3966 ms` / `0.677 TFLOPS`；同属 1 预热 + 100 次迭代的快速采样，与重命名前同量级）。

- 2026-09-15 15:06 `operators/gemm`：新增 `--bench` 用法文档 `notes/bench-usage.md`
  - 新建 `operators/gemm/notes/bench-usage.md`：整理扫描模式的完整用法 —— 编译与入口区分（无参数 = 正确性 + 性能，`--bench` = 只计时）、`--sizes/--csv/--warmup/--budget` 参数表与约束、真实终端输出样例（`--sizes 128 --budget 50` 实测）、CSV 列定义与产物落点、绘图脚本命令、计时口径（自适应迭代的由来与严格对比口径的差别）、常见坑（相对路径落点、`--sizes` 非法值、退出码 0/1/2、样本量随尺寸缩水）。
  - `test.cu`：补齐扫描路径的函数级注释 —— 文件头说明 `bench_gemm_kernel` / `bench_cublas_sgemm` 只计时不校验；两个函数上方写明各参数用法（`warmup_iterations` 空转次数、`budget_ms` 只反推每组迭代数且采样组数固定 3 组、`iters_out` 回传样本量）与返回值语义（中位数 ms，非法/启动失败返回 -1.0）。
  - 文档同步：`operators/gemm/README.md`（「性能可视化」补指向用法文档的入口）。

- 2026-09-15 14:58 `operators/gemm`：bench 产物改为按时间戳分目录
  - `main.cu`：`--bench` 未指定 `--csv` 时，CSV 默认写到 `operators/gemm/bench/<YYYYmmddHHMMSS>/gemm_bench.csv`（`localtime_r` 取扫描时刻），每次扫描独占一个目录，同一台机器上多次运行不再互相覆盖；`--csv` 仍可覆盖路径，父目录按需创建、不可写时退回 stdout 的逻辑不变。
  - `scripts/plot_kernel_perf.py`：PNG 默认与输入 CSV 同目录同名，随 CSV 一起落入时间戳目录；新增重复点校验 —— 同一 `(label, size)` 出现两次即报错并提示只传单次扫描的 CSV（防止用通配符把多次扫描混成一张图时静默叠加数据点）。
  - 产物调整：删除旧路径 `bench/gemm_bench.csv|png`，改为 `bench/20260915145532/gemm_bench.csv|png`（表、图、CSV 三者同源）。
  - 文档同步：`operators/gemm/README.md`（性能可视化命令与产物落点说明、结论记录配图链接与采样波动说明）、根 `README.md`（命令示例与产物路径）、`AGENTS.md`（§2 约定 `bench/<时间戳>/`）。

- 2026-09-15 14:10 `operators/gemm`：新增 `--bench` 多尺寸性能扫描与仓库级绘图脚本
  - `test.cuh` / `test.cu`：新增 `bench_gemm_kernel` 与 `bench_cublas_sgemm` —— 只分配设备缓冲（输入以 0 填充）并做 CUDA event 计时，不生成主机端输入、不做 CPU 参考与结果回拷；迭代数按预算自适应（1 次调用估计单次耗时 → 每组迭代数 = clamp(200 ms/点预算内的调用数, 1, 100) → 连采 3 组取中位数）。固定 100 次迭代在朴素 v0 的 4096³（单次约 0.95 s）上会让单个采样点耗掉数分钟。内核启动失败返回负值，由调用方跳过该点，不终止整轮扫描。
  - `main.cu`：新增 `--bench [--sizes 128,256,...] [--csv <path>] [--warmup <n>] [--budget <ms>]`，默认扫 `128,256,512,1024,2048,4096`；`KernelEntry` 拆出短名 `plot_name`（图例/CSV）与 `description`（正确性报告），CSV 表头为 `label,size,median_ms,gflops,iters`；输出父目录按需创建，路径不可写时退回 stdout；部分采样点失败只打印 `SKIPPED` 并继续，全部失败才返回非 0。无参数时的「正确性 + 性能」流程与输出保持不变。
  - 新增 `scripts/plot_kernel_perf.py`（+ `scripts/requirements.txt`）：读 bench CSV 画「GFLOP/s vs 矩阵尺寸」多内核对比图，样式对齐参考图（灰底白网格、等宽字体、分类等距刻度 + 45° 旋转标签、线末端同色文字标注，默认配色 `#F8766D / #00BA38 / #619CFF`），并提供 `--palette / --title / --xlabel / --ylabel / --label-col / --x-col / --y-col` 等覆盖项；只依赖 CSV 的列名，其他算子可复用。
  - 回归与采样：默认入口 `3/3 PASS` 不变；`--bench` 18 个采样点全部成功（整轮约 9 s）。GFLOP/s 结果、采样说明与生成的性能曲线见 `operators/gemm/README.md`「结论记录」，CSV / PNG 复现产物入库到 `operators/gemm/bench/`。
  - 文档同步：`operators/gemm/README.md`（扫描口径、目录布局补 `bench/`、新增「性能可视化」小节与多尺寸曲线表）、根 `README.md`（目录树补 `scripts/`、环境要求补 Python 依赖、构建与运行补 bench + 绘图命令）、`AGENTS.md`（§2 补 `scripts/` 约定）。

- 2026-09-14 `operators/gemm`：接入 SGEMM v1 的原始模板实现测试
  - `sgemm_v1.cuh` 的函数体保持原样，不使用共享内存；入口以 `BLOCKSIZE=32`、`grid=(ceil(M/32),ceil(N/32))`、`block=(1024,1)` 启动。`sgemm_v1.cu` 保留但不编译进默认目标。
  - 默认入口仅运行 `512×512×512` 一个场景，并按 cuBLAS、v0、v1 的顺序输出。
  - 回归结果：cuBLAS 与 v0 通过；v1 记为 `max_err=1.000e+00`、错误元素数 253,952，未通过正确性验证，因此未记录性能结论。（2026-09-15 复核：该数值源于物理 block 只起 32 线程的启动配置错误，v1 索引映射本身正确，见当日条目。）
  - 文档同步：`operators/gemm/README.md`、根 `README.md`。

- 2026-09-14 `operators/gemm`：新增 cuBLAS SGEMM 厂商库对照并收敛默认测试口径
  - `operators/gemm/CMakeLists.txt`：GEMM 目标链接 `CUDA::cublas`；`test.cuh/.cu` 新增 `test_cublas_sgemm`，以行主序 `C=A×B` 与列主序 `C^T=B^T×A^T` 的等价映射调用 `cublasSgemm`，不产生额外转置或拷贝。
  - 入口运行 `512×512×512` 一个正常场景；自定义内核与 cuBLAS 对照统一使用 CPU double 参考、相对误差 `≤1e-3`、1 次预热和 100 次 CUDA event 计时迭代。GEMM 报告仅保留中位数与 TFLOPS，不再计算或输出 P5/P95。
  - cuBLAS 固定 `CUBLAS_PEDANTIC_MATH`，作为严格 FP32 对照；本机 RTX 4060 Laptop、CUDA 12.9、Release、sm_89 开发采样：v0 `1.8001 ms` / `0.149 TFLOPS` / `max_err=1.275e-06`，cuBLAS `0.0798 ms` / `3.366 TFLOPS` / `max_err=5.894e-07`，均通过。
  - 文档同步：`operators/gemm/README.md`。

- 2026-09-14 `operators/gemm`：接入 SGEMM v0 及可复用测试入口
  - 新增 `sgemm_v0.cuh`、`sgemm_reference.cuh/.cu`、`test.cuh/.cu` 与 `main.cu`：行主序 `C(M×N)=A(M×K)×B(K×N)`，CPU 参考以 double 累加；测试驱动用确定性正数输入、CUDA event 中位数/P5/P95 与逐元素相对误差 `≤1e-3` 统一验证并输出 TFLOPS。
  - `sgemm_v0`：每线程计算一个输出元素，固定 `block=(16,16)`、`grid=(ceil(M/16),ceil(N/16))`、无动态共享内存；补齐边界空转与 `C[row*N+col]` 写回。
  - 默认入口运行 `512×512×512` 与 `513×511×509` 两组快速场景（1 次预热 + 100 次迭代）；在本机 RTX 4060 Laptop、CUDA 12.9、Release、sm_89 下均通过，最大相对误差为 `1.275e-06` / `1.293e-06`。
  - 文档同步：`operators/gemm/README.md`、根 `README.md`。

- 2026-09-11 16:36 softmax：接入 cuDNN 厂商库对照参考（`-DSOFTMAX_WITH_CUDNN`，默认开启）
  - `operators/softmax/CMakeLists.txt`：新增 `SOFTMAX_WITH_CUDNN` 选项，按 `-DCUDNN_ROOT` > 环境变量 `CUDNN_ROOT` > `CONDA_PREFIX` > `/usr/local/cuda`、`/usr` > pip 版 `nvidia-cudnn-cu12`（`site-packages/nvidia/cudnn`）的顺序探测；pip / conda 版只提供带 SONAME 的 `libcudnn.so.9`（无 `libcudnn.so` 软链、不在 `ldconfig` 视界内），`find_library` 匹配不到时用 `file(GLOB)` 兜底，并写入 rpath 免设 `LD_LIBRARY_PATH`；选项默认开启（本机已装 cuDNN），未装时用 `-DSOFTMAX_WITH_CUDNN=OFF` 关闭，否则 configure 直接报错提示 `-DCUDNN_ROOT`
  - `operators/softmax/src/softmax.cuh` / `.cu`：新增 `SoftmaxHostKernel` 别名与 `softmax_cudnn`（`cudnnSoftmaxForward` + `CUDNN_SOFTMAX_ACCURATE` + `CUDNN_SOFTMAX_MODE_INSTANCE`），把行主序 `M×N` 映射为 `[n=M, c=1, h=1, w=N]` + `nStride=N`；句柄与张量描述符进程内复用、绑定默认流，`M <= 0 || N <= 0` 直接返回，错误经 `SOFTMAX_CUDNN_CHECK` 打印后终止
  - `operators/softmax/src/test.cuh` / `.cu`：`test_softmax_kernel` 新增可选参数 `host_kernel`，非空时改由主机 API 驱动（`kernel` / `grid` / `block` / `smem_bytes` 忽略），正确性判据与计时口径与内核路径完全一致
  - `operators/softmax/src/main.cu`：`SOFTMAX_WITH_CUDNN` 下在被测内核之后追加 cuDNN 对照段（经 `host_kernel` 通道、不注册进 `kKernels`），复用 A/B/C 场景组；另加运行开关 `kEnableCudnnReference`（默认 `true`）与 `enable_boundary` / `strict_benchmark` 并列放在 `main()` —— CMake 选项只决定“编不编、链不链”（链接期依赖，值由 CMake 缓存），本开关决定“跑不跑”，置 `false` 时打印「已跳过」并跳过该段
  - 回归验证：`-DSOFTMAX_WITH_CUDNN=OFF` 档 24/24 PASS；`=ON` 档 26/26 PASS，开启 `enable_boundary` 后 260/260 PASS（13 组 × 20 场景）；`kEnableCudnnReference=false` 时该段打印「已跳过」且回到 24/24。同场对照：cuDNN 0.6865 ms / 195.50 GB/s（4096²）、0.6889 ms / 194.83 GB/s（16384×1024），`max_err` 1.231e-06 / 1.234e-06 —— 与自研块内归约各版同档
  - 开关默认值由 `OFF` 改为 `ON`：本机已装 cuDNN，默认 `build/` 重新 configure 后回归 26/26 PASS；源码注释与文档一并改为「默认开启 + 未装时用 `=OFF` 关闭」，并补记 `option()` 默认值只在首次 configure 写入、缓存里是 `OFF` 时须显式传 `=ON` 的陷阱
  - 文档同步：`operators/softmax/README.md`（状态表新增「参考 / cuDNN」行、目录布局、开关表新增 `kEnableCudnnReference` 行并说明与 CMake 选项的分工、构建与运行新增 cuDNN 对照小节的开关用法与缓存说明、结论记录新增「cuDNN 厂商库对照」同场表）与根 `README.md`（构建与运行新增 `-DSOFTMAX_WITH_CUDNN` 构建开关说明）

### Fixed

- 2026-09-16 `operators/gemm`：统一 SGEMM v2 的 block 坐标与 host grid 轴语义
  - `sgemm_v2` 保持 `cRow = blockIdx.y`、`cCol = blockIdx.x`，即 `grid.y` 覆盖输出 tile 行、`grid.x` 覆盖输出 tile 列。
  - `main.cu` 的启动配置新增行轴归属；仅 v2 以 `grid=(ceil(N/32),ceil(M/32))` 启动，v0 / v1 继续使用原有 `grid=(ceil(M/tile_rows),ceil(N/tile_cols))` 映射。
  - `operators/gemm/README.md` 同步 v2 的二维坐标和启动约束；默认快速回归仍仅运行 `512×512×512`。
  - 非方阵复核：临时使用 `513×511×509`，修复前 v2 有 511 个错误元素；修复后 cuBLAS、v0、v1、v2 均通过，v2 最大相对误差 `1.293e-06`。

- 2026-09-15 13:50 `operators/gemm`：更正 v1 结论并补齐 v1 源码注释
  - 复核：`./build/operators/gemm/gemm` 当前 3/3 PASS，v1 为 `max_err=1.275e-06`、中位 `0.3630 ms` / `0.740 TFLOPS`（同日三次采样 0.3513–0.3811 ms；同场 v0 `1.8651 ms` / `0.144 TFLOPS`、cuBLAS `0.0657 ms` / `4.088 TFLOPS`）—— 2026-09-14 记录的「v1 未通过正确性验证」不再成立。
  - 根因定位：以 `block=(32,1)` 启动能复现与旧记录逐位一致的 `max_err=1.000e+00`、错误元素数 253,952（每 block 只写出 `32×32` tile 首行 → 8192 项正确、253,952 项未写入，未写入区为 0 时相对误差恰为 `1.000e+00`）；v1 的线性索引映射本身正确，故障来自物理 block 未给满 `BLOCKSIZE² = 1024` 个线程。
  - 形状复核：按现有注册方式（`grid=(ceil(M/32),ceil(N/32))`、`block=(1024,1)`）另测 `513×511×509`、`32×32×32`、`33×1×1`、`1×1×1`、`1024×1024×1024` 全部 PASS（`max_err ≤ 1.89e-06`）。
  - `sgemm_v1.cuh`：补 `#pragma once`、文件头模块说明与函数级注释（作用 / 参数 / 返回值 / 启动约束 / 注意事项，含「线程数不足 `BLOCKSIZE²` 时只覆盖 tile 首行且不报错」这一坑）；计算逻辑不变，仅按 `.clang-format` 规范化排版（Tab 缩进、运算符空格、超长签名换行）。
  - `sgemm_v1.cu`：原为 0 字节裸文件，补占位说明（v1 实现内联在 `.cuh`，本文件被 `CMakeLists.txt` 的 `list(REMOVE_ITEM)` 排除在 gemm 目标之外）。
  - 文档同步：`operators/gemm/README.md`（v1 状态改「完成（已接入测试）」、版本规划补启动约束、结论记录补 v1 行与同场采样噪声说明、更正旧失败叙述）、根 `README.md`（GEMM 状态、目录树与路线图）、`AGENTS.md`（§0 现状速览拆出 GEMM / Attention 两行）。

- 2026-09-14 仓库卫生：`.gitignore` 补充 `demo/` 就地编译产物的忽略规则
  - 现象：`demo/demo_stream`、`demo/demo_utils` 两个无扩展名的 ELF 可执行文件曾被入库（`2c6e477`）。原 `.gitignore` 已覆盖 `build*/`、`*.o`、`*.a`、`*.so` 等，但无法匹配**无扩展名**的 Linux 可执行文件，`demo/` 下用 nvcc 就地编译出的产物每次都会落到 `git status` 里。
  - 处理：`.gitignore` 新增 `/demo/*` 后放行 `!/demo/*.cu`、`!/demo/*.cuh` —— `demo/` 不接入 CMake、须 nvcc 就地编译（AGENTS.md §2），故整体忽略再白名单源码，新增示例无需再改 `.gitignore`。
  - 验证：`git check-ignore -v --no-index` 对 `demo/demo_stream`、`demo/demo_utils`、`demo/helloWorld`（含尚未存在的 `demo/demo_attention`）均命中 `/demo/*`；`demo/*.cu`、`demo/*.cuh` 未被忽略，仍可正常入库。
  - 遗留：两个二进制当前仍在 git 索引中，`.gitignore` 对已跟踪文件无效，本次提交需一并记录其删除（`git add -A demo/`）。
  - 文档同步：`AGENTS.md`（§2 目录与构建约定）。

- 2026-09-14 构建环境：消除构建缓存里的跨系统 Ninja 路径
  - 现象：Windows PowerShell 下 `cmake --preset release` 在 `project()` 处失败，报 `Running '/mnt/d/Development/Vivado/2025.2/Vivado/bin/ninja' '--version' failed with: no such file or directory`；改在 WSL 内复用旧缓存时又会因缓存记录的系统路径不一致而拒绝。
  - 根因：WSL 未安装原生 Ninja，CMake 经 WSL interop 抓到 Windows 的 `D:\Development\Vivado\2025.2\Vivado\bin\ninja.exe`，把 WSL 专有路径 `/mnt/d/...` 写进了 `build/`、`build-gemm/`、`build-gemm-cublas/`、`build-softmax-report/`、`build-softmax-report-cuda/` 五个构建目录的 `CMAKE_MAKE_PROGRAM`。
  - 处理：WSL 安装 `ninja-build 1.11.1-2`（`/usr/bin/ninja`，PATH 中优先于 interop 目录）；删除被污染的 `build/CMakeCache.txt` 后重新 configure（`CMAKE_MAKE_PROGRAM:FILEPATH=/usr/bin/ninja`、`CMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda/bin/nvcc`）；四个实验构建目录就地改写同一变量；顶层 `CMakeLists.txt` 新增 `CMAKE_HOST_WIN32` 守卫，明确 Windows 侧 CMake 不受支持。
  - 验证：WSL 内 `cmake --preset release` 通过（CUDA Toolkit 12.9.86），`cmake --build build --target gemm reduce softmax` 通过，`gemm` 2/2 PASS（max_err 1.275e-06 / 5.894e-07）、`reduce` 16/16 PASS；Windows 侧 `cmake --preset release` 命中守卫并打印提示。
  - 文档同步：根 `README.md`（环境要求）、`AGENTS.md`（§1 环境与构建）。

- 2026-09-08 13:16 文档与注释同步
  - 根 `README.md` 更新 Reduce 的 v0…v7 与 80 项全量回归状态
  - 补充未接入 CMake 的 `demo/` 说明
  - 基准方法改为反映 Reduce 实际使用的确定性 `i % 1000` 输入
  - 修正 `main.cu` 中 reduce_v5 模板实例化位置的过时说明
- 2026-09-08 13:31 根 `README.md` 文档同步（事实校正）
  - Reduce 的 CUDA 核心版状态由「进行中（v0…v7）」改为「完成（v0…v7）」
  - 目录结构树按仓库实际拆分 reduce（含 `notes/reduce.md`、`src/`），并标明 softmax / gemm / attention 为 src/ 尚未创建的骨架
  - 构建示例目标由未启用的 softmax 改为当前唯一启用的 reduce
  - 路线图勾选 Reduce 的验证 / 基准记录，收窄剩余项为 Softmax / GEMM / Attention
- 2026-09-09 17:35 按 AGENTS.md 注释规范去重收敛 reduce / softmax 源码注释
  - `reduce.cu`：各版本内核前的长段解说注释收敛为“实现结构 + 指向 `reduce.cuh` / `README.md`”，删除与声明注释重复的覆盖口径、启动约束与推导叙述
  - `softmax.cu`：`v1`/`v2`/`v3` 内核注释收敛为指向 `softmax.cuh` 的实现要点，删除与声明 / helper 注释重复的长段说明与误差量级推导（此类内容已在 README），并清理文件尾部杂散空行
  - 同步 `softmax` 的 `test.cuh` / `test.cu` 注释版本口径：`v0/v1/v2` → `v0/v1/v2/v3`（行映射与 `smem_bytes` 说明补 v3）
  - 将 v3 开发期临时调试文件 `softmax_v3_test.cu` 自 `src/` 移入 `scratch/`（其自带 `main()`，原会被 `file(GLOB)` 编入 softmax 目标导致链接失败）
- 2026-09-10 09:46 按 AGENTS.md 注释 `online_softmax` 并以实际代码校正启动配置与文档
  - `online_softmax.cuh` / `.cu`：补模块 / 函数 / 启动约束 / 注意事项注释（行映射 `row = blockIdx.x * blockDim.x + threadIdx.x`、`grid = ceil(M/blockDim.x)`、无共享内存、`N == 0` 空行不读不写），推导细节留在 README；缩进统一为 `.clang-format` 规定的 Tab
  - `main.cu`：`online_softmax_v0` 的行映射由 `kBlockPerRowShuffle` 改为 `kThreadPerRow`（实现实为每线程处理一行，原配置 `grid = rows` 会超配约 256 倍 block），同步打印名 / `RowMap` / 线程数 / `SmemFor` 注释版本口径
  - `operators/softmax/README.md`：状态表 / 版本规划 / 参考规模 / 目录布局与测试 / 场景说明按实际实现改为「每线程处理一行」，早期块内协作版描述（`warpMergeOnline` / `blockMergeOnline`）改为后续演进方向；online 开发采样替换为当前实现实测（4096² 4.1289 ms / 32.51 GB/s、16384×1024 3.3539 ms / 40.02 GB/s，max_err 5.488e-06 / 2.884e-06）
  - 回归验证：`cmake --build build --target softmax` 编译通过，softmax 14/14（默认档）PASS
- 2026-09-10 09:52 `AGENTS.md`：强化 §6 注释要求并校正风格描述
  - §6 注释规范改写为「不缺 / 不冗余 / 不失真」三条硬性要求：新增源文件禁止裸提交、必备启动约束、推导与性能结论只进 README / notes、代码与文档冲突以实际代码为准
  - §3 第 3 条补「注册项的 `RowMap` / `smem_bytes` 须与内核实际行映射一致 + 源码自带符合 §6 的注释」
  - 修正 §6 `.clang-format` 描述：`IndentWidth 2` → `IndentWidth 4` + `UseTab: ForIndentation`（与仓库 `.clang-format` 一致）

### Changed

- 2026-09-14 `operators/softmax`：测试入口固定为两组正常场景的快速正确性与性能测试
  - `src/main.cu`：删除边界/健壮性场景、`kEnableBoundary`、`kStrictBenchmark`、`kEnableCudnnReference` 及 `[A]` / `[B]` / `[C]` 分组输出；12 个自研内核固定运行两个正常形状，cuDNN 若由 CMake 编译进来则自动运行相同两组形状，均使用快速采样（1 次预热 + 100 次迭代）。
  - `operators/softmax/README.md`：同步默认入口为 24 项测试，启用 cuDNN 构建时为 26 项；历史严格采样记录保留并明确为历史数据。
- 2026-09-14 `operators/reduce`：测试入口固定为两组正常场景的快速正确性与性能测试
  - `src/main.cu`：删除 `kEnableBoundary`、`kStrictBenchmark`、边界/健壮性场景和 `[A]` / `[B]` / `[C]` 分组输出；8 个内核固定运行大规模对齐与尾部非对齐形状，`test_reduce_kernel` 固定使用快速采样（1 次预热 + 100 次迭代）。
  - `operators/reduce/README.md` 与根 `README.md`：同步入口只运行 16 项正常场景；历史严格基准表保留并标注为历史记录。
- 2026-09-14 `operators/softmax`：每个测试报告前打印 GPU 指标并统一报告结构
  - `src/test.cu`：新增进程内缓存的 CUDA Runtime 设备快照；每次 `test_softmax_kernel`
    调用固定输出 `Test Case`、`GPU Configuration`、`Launch Configuration`、`Results`、
    `Validation` 五段。GPU 配置含设备名、计算能力、SM 数、显存、每 block 共享内存、显存
    时钟、总线位宽和按 `显存时钟 × 2 × 位宽/8` 计算的理论峰值带宽；不引入 NVML，不伪造
    温度、功耗或实时频率。
  - `Results` 统一保留并首字母大写截图七项英文字段：`CPU & GPU Results Match`、`CPU Time`、
    `GPU Time`、`Speedup`、`Effective Bandwidth`、`Peak DRAM Bandwidth`、`Bandwidth Utilization`。
    空形状和非法参数仍生成完整报告，不能测量的性能字段显示 `N/A`。
  - `src/main.cu`：删除只在程序开头出现的 `PrintDeviceInfo()`，避免无归属且与每报告设备段
    重复的输出；内核注册、行映射、采样与正确性判据不变。
  - 验证：在独立 WSL Release 构建目录 `build-softmax-report-cuda/`（CUDA 12.9.86、sm_89、
    `SOFTMAX_WITH_CUDNN=OFF`）中重新构建通过；默认 12 内核 × 2 正常场景为 24/24 PASS，
    输出契约检查得到 24 个 `Test Case` 与 24 个 `GPU Configuration`，且截图七项字段均存在；
    `compute-sanitizer --tool memcheck` 退出码 0、`ERROR SUMMARY: 0 errors`。
- 2026-09-14 10:00 `operators/softmax`：重构测试驱动，报告与全部输出文案改为英文
  - `src/test.cu`：`test_softmax_kernel` 由单段长流程收敛为「生成输入与主机参考 → 申请 / 拷入 → 构造启动器 → 预热与计时采样 → 拷回比对 → 渲染报告」的串联结构，各步拆到匿名命名空间的小工具（`PeakDramBandwidthGbps` / `FillDeterministicInput` / `MeasureKernel` / `CompareCellwise` / `PrintMismatchDetail` / `PrintReport`），并引入 `TimingStats` / `CompareStats` / `RunReport` 三个结果结构承载数据；判据（容差 1e-5、跳过 `|ref| < 1e-30`）、采样口径（开发 / 严格两档）与返回值语义均不变
  - 报告改为英文文案，前 7 行与 `common/include/operator_common/cuda_check.h` 的 `PrintDeviceInfo` 同口径：`CPU & GPU Results match` / `CPU time`（主机参考单次耗时，`CpuTimer`）/ `GPU time`（中位数）/ `Speedup` / `Effective Bandwidth` / `Peak DRAM Bandwidth`（`显存时钟 × 2 × 位宽/8`，查询失败显示 `n/a`）/ `Bandwidth Utilization`；其后为 `max rel err`、采样口径、`out[0][0]` 抽样（改为 `%.6e`）与严格档 `P5 / P95`，失败时打印 `mismatched elements` 与最大误差位置附近元素
  - `src/test.cuh`：接口注释同步「英文报告」与新增指标说明
  - `src/main.cu`：被测内核名、场景名、分组标题、跳过提示、汇总行与 cuDNN 参考段文案全部改为英文（`[A] normal` / `[B] boundary` / `[C] abnormal & robustness` / `==== Result: %d/%d items PASS ====`）；`kKernels` 注册项、`RowMap` 与启动配置不变
  - 验证：`cmake --build build --target softmax` 通过；默认档 24/24 PASS；`enable_boundary=true` 全量回归 240/240 PASS；另以 `-DSOFTMAX_WITH_CUDNN=ON` 在单独构建目录编译并跑通 260/260 PASS（临时目录已清理）；3 个改动文件 `clang-format --dry-run --Werror` 通过
  - 文档同步：`operators/softmax/README.md`（补报告字段与英文文案说明）、`docs/benchmark-methodology.md` §3.3（补 CPU 参考耗时、峰值显存带宽与利用率的口径及定位）
- 2026-09-11 17:20 `operators/softmax`：按 `AGENTS.md` §6「不缺 / 不冗余 / 不失真」重写全部源码注释
  - `softmax.cuh` / `.cu`：文件头逐版本演进长段收敛为一行版本索引；删去 v4/v5 的「资源流量对比」与推导叙述（归口 README），实现侧（v3/v4/v5、cuDNN）注释收敛为「结构要点 + 指向 `.cuh` / README」；声明侧保留作用 / 参数 / 返回值 / 启动约束（grid、blockDim、smem 字节数）/ 注意事项
  - `online_softmax.cuh` / `.cu`：同口径收敛，v3 / v3_false 的「真寄存器 / local memory」推导压缩去重、归口 README，保留启动约束与正反对照关系
  - `test.cuh` / `test.cu`：`grid` / `smem_bytes` 参数说明由「逐版本罗列」改为「由调用方按行映射给出（见 `main.cu` 的 `RowMap` / `GridFor` / `SmemFor`）」，避免版本增删后失真；并修正有效带宽注释中「各版本实际都三遍各读行一次」的错误表述（v0~v3 读 3 遍、v4 读 1 遍、v5 与 online 各版读 2 遍）
  - `main.cu`：`kBlock` / `RowMap` / 边界场景 / `GridFor` / `SmemFor` 注释去版本化收敛，版本口径统一指向 `.cuh` 与 README
  - 验证：`cmake --build build --target softmax` 编译通过、默认档 24/24 PASS；`git diff` 核对为纯注释改动（无代码语义变化）
- 2026-09-10 11:10 全仓库严格基准采样口径下调（面向 RTX 4060 Laptop）
  - `docs/benchmark-methodology.md` §3.2/§3.4：严格档由 1000 预热 + 21 组 × 10000 次下调为 100 预热 + 21 组 × 1000 次（总 2.1 万次调用，约 1/10），并说明下调原因（慢内核整表严格基准耗时过长）
  - `AGENTS.md` §5：默认迭代建议同步为严格对比 100 预热 + 21 组 × 1000 次
  - `operators/softmax/src/test.cu`、`operators/reduce/src/test.cu`：严格档参数同步下调（保留 21 组以维持 P5/P95 分位分辨率）
  - `operators/softmax/src/test.cuh`、`operators/reduce/src/test.cuh`：严格采样注释同步
  - `operators/softmax/README.md`、`operators/reduce/README.md`、`operators/reduce/notes/reduce.md`：开关说明与计时口径同步；reduce 结论记录表数字标注为旧口径实测（中位数不受采样密度显著影响）
- 2026-09-10 09:28 全仓库源码统一 `clang-format` 格式化并校正风格文档
  - 对 `common/`、`demo/`、`operators/{reduce,softmax}` 全部自有源文件执行 `clang-format -i --style=file`（排除 `build/`、`.venv/`）：旧文件原为 2 空格缩进，现统一为 `.clang-format` 规定的 Tab 缩进 + 行宽 100 重排，无语义变化
  - 修正 `AGENTS.md` §6 风格描述：`IndentWidth 2` → `IndentWidth 4` + `UseTab: ForIndentation`，与仓库 `.clang-format` 实际取值一致
  - 回归验证：reduce 16/16、softmax 14/14（默认档）PASS，`cmake --build build` 全目标编译通过

### Added

- 2026-09-10 13:50 `operators/softmax`：新增并接入 `online_softmax_v4`（grid-stride 多行处理 / 块复用 + 单趟在线归约 + float4）
  - `online_softmax.cu`：新增 v4 内核 —— 行映射由「每行一个 block」改为「每 block 经 grid-stride 循环处理多行」（`for (row = blockIdx.x; row < M; row += gridDim.x)`），复用同一份寄存器与静态 `__shared__`（`blockReduceOnline` 内部中转）；行内归约 / 访存逐项同 online-v2（单趟在线归约 + 两级 warp shuffle 合并、`N % 4 == 0` 时 float4 否则整行标量）；跨行复用共享内存的顺序由 `blockReduceOnline` 收尾 `__syncthreads` 保证，无需在循环末尾额外同步
  - `online_softmax.cuh`：补 v4 声明与启动约束（grid 由调用方给出、`blockDim.x` 为 32 的倍数且 <= 1024、无动态共享内存、`M == 0` / `N == 0` 空转、非 4 倍列宽整行标量回退），文件头补单元内 v4 说明
  - `main.cu`：被测内核表注册 `online_softmax_v4`；`RowMap` 新增 `kGridStrideRow`（`grid = min(rows, SM 数 × kGridStrideBlocksPerSm)`，`kGridStrideBlocksPerSm = 32`），`main()` 按当前设备 SM 数初始化该上限并打印；`GridFor` / `SmemFor` / `kBlock` / 边界场景注释同步版本口径
  - 验证：`nvcc -Xptxas -v` 对 v4 报告 `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`、`Used 29 registers`；开启 `enable_boundary` 全量回归 12 内核 × 20 场景 **240/240 PASS**（覆盖 `grid = rows` 退化路径、非 4 倍列宽标量回退、空矩阵 / 空行、`N = 0` 等）
  - 实测（同场开发采样）：v4 与 online-v2 持平（4096² 0.6199 ms / 216.51 GB/s vs 0.6167 ms / 217.64 GB/s；16384×1024 0.6220 ms / 215.78 GB/s vs 0.6156 ms / 218.02 GB/s，`max_err` 逐项相同 1.341e-06 / 1.392e-06）—— 访存受限算子下 block 调度 / 建立开销本就可忽略，「块复用」收益被并行度下降抵消；grid 上限系数 8/32/128 同场对比，32 与 128 在噪声内、8（192 block）宽行掉到 ~197 GB/s
  - `operators/softmax/README.md`：状态表 / 指标口径 / 版本规划 / 参考规模 / 目录布局与测试 / 场景说明 / 开关说明补 online-v4，并追加 online-v4 与 online-v0/v1/v2/v3/v3_false 同场开发采样与结论
  - 顶层 `README.md` / `AGENTS.md`：Softmax 状态同步为进行中（v0/v1/v2/v3/v4/v5 与 online-v0/v1/v2/v3/v3_false/v4 完成，全量回归 240 项通过）
- 2026-09-10 13:34 `operators/softmax`：新增并接入 `online_softmax_v3_false`（“假寄存器”反面对照，运行期下标缓存被 ptxas 降级为 local memory）
  - `online_softmax.cu`：新增 v3_false 内核 —— 行映射 / 归约 / 启动约束与 online-v3 逐项同构，唯一区别是缓存本线程列时用运行期下标 `reg_cache[count++]`（而非 v3 的编译期常量下标 `reg[k]`）；寄存器不可被运行期索引，ptxas 把该定长数组整体降级为 local memory，写回遍仍免掉第 2 遍全局读。按设计**不做列宽分派、无回退路径**，始终缓存，要求 `ceil(N/blockDim.x) <= kRegTile`（block = 256 时 `N <= 4096`），超过则越界（由调用方保证）
  - `online_softmax.cuh`：补 v3_false 声明与设计说明（与 v3 的对照关系、“假寄存器真 local memory”的成因与判据），文件头补单元内 v3 / v3_false 差异
  - `main.cu`：被测内核表注册 `online_softmax_v3_false`（复用 `RowMap::kBlockPerRowShuffle`：`grid = rows`、smem = 0），同步 `kBlock` / `RowMap` / `GridFor` / `SmemFor` 注释的版本口径
  - 验证：`nvcc -Xptxas -v` 对 v3_false 报告 `64 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`、`Used 19 registers`（同场 online-v3 为 `0 bytes stack frame`、`Used 63 registers`）—— 64 B 恰为 `kRegTile`(16) × sizeof(float)，证实数组落在 local memory；开启 `enable_boundary` 全量回归 11 内核 × 20 场景 220 项全部 PASS（全部场景每线程元素数 ≤ 16，满足无回退版本的契约）
  - 实测（同场开发采样）：v3_false 宽行明显领先 online-v3、4096² 基本持平（16384×1024 0.7655 ms / 175.34 GB/s vs 0.9951 ms / 134.87 GB/s，+30%；4096² 0.6639 ms / 202.16 GB/s vs 0.6636 ms / 202.25 GB/s，差 0.05%）—— v3 的 63 寄存器使 SM 驻留 block 数由 6 降到 4，占用损失盖过“少读一遍全局”的收益；v3_false 仅 19 寄存器、local memory 由 L1 兜底，宽行上更划算
  - `operators/softmax/README.md`：状态表 / 版本规划 / 指标口径 / 参考规模 / 目录布局与测试 / 开关说明补 online-v3_false，追加 v3_false 与 online-v1/v2/v3 同场开发采样（含寄存器 19/63 与栈帧 64 B 对照）
  - 顶层 `README.md` / `AGENTS.md`：Softmax 状态同步为进行中（v0/v1/v2/v3/v4/v5 与 online-v0/v1/v2/v3/v3_false 完成，全量回归 220 项通过）
- 2026-09-10 12:20 `operators/softmax`：新增并接入 `online_softmax_v3`（寄存器分片缓存，真寄存器 0 spill）
  - `online_softmax.cu`：新增 v3 内核 —— 每线程元素数 `ceil(N/blockDim.x) <= kRegTile`（16）时，第 1 遍在线归约的同时把本线程负责的列缓存进 `float reg[kRegTile]`，写回遍直接取寄存器、省掉第 2 遍全局读；超容量（block=256 时 `N > 4096`）自动回退 online-v1/v2 式两遍重读。真寄存器的前提是静态下标：`kRegTile` 为编译期常量 + `#pragma unroll` 整体展开使下标 `k` 成为常量（草稿的 `reg_cache[count++]` 是运行期下标，会被 ptxas 降级为 local memory）；分派条件只依赖 `N` / `blockDim`、对整 block 一致，两条分支内的 `__syncthreads` 安全。顺带移除草稿遗留的 clang 内部头文件 include（原导致 nvcc 编译失败，与 online-v1/v2 草稿同类缺陷），并把散落的行尾空白 / tab-only 空行按 `.clang-format` 整理
  - `online_softmax.cuh`：补 v3 声明与启动约束（行遍历 / 归约 / 启动约束同 v1/v2，仅写回遍的 x 来源改为寄存器分片；分片容量常数 `kRegTile` = 16，超容量自动回退），文件头补单元内 v0/v1/v2/v3 差异
  - `main.cu`：被测内核表注册 `online_softmax_v3`（复用 `RowMap::kBlockPerRowShuffle`：`grid = rows`、smem = 0），同步 `kBlock` / `RowMap` / `GridFor` / `SmemFor` 注释的版本口径
  - 验证：`nvcc -Xptxas -v` 对 v3 报告 `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`（真寄存器）与 `Used 63 registers`（online-v1 为 19、online-v2 为 26）；回退路径以临时 `kRegTile = 2` 强制正常场景走回退验证通过（结果与寄存器路径逐项一致）；默认档 20/20 PASS
  - 实测（同场开发采样）：v3 低于 online-v1/v2（4096² 183.99 GB/s vs 212.22 / 216.59，16384×1024 132.63 GB/s vs 216.03 / 199.43）—— 寄存器数 63 使 SM 驻留 block 数由 6 降到 4，且 `kRegTile` 固定 16 使每线程元素少时仍执行 16 轮展开（`16384×1024` 每线程仅 4 个元素）；结论：真寄存器目标达成但当前形状无收益，后续需按列宽分列实例化
  - `operators/softmax/README.md`：状态表 / 版本规划 / 指标口径 / 参考规模 / 目录布局 / 测试章节补 online-v3，并追加 online-v3 与 online-v1/v2 同场开发采样（含寄存器数 19/26/63 与 0 spill 对照）
- 2026-09-10 10:59 `operators/softmax`：新增并接入 `online_softmax_v2`（online-v1 + float4 向量化）
  - `online_softmax.cu`：追加 v2 内核并修复草稿缺陷 —— 移除草稿遗留的 clang 内部头文件 include（原导致编译失败）、统一 `.clang-format` 规定的 Tab 缩进、写回改 `float4` 整写（原为标量写且 `y4` 声明未用）、补齐文件头版本清单与内核注释
  - `online_softmax.cuh`：补 v2 声明与启动约束（列宽为 4 的倍数时行内以 `float4` 单趟在线归约 + 整写回，否则整行回退标量；启动约束同 v1），文件头补单元内 v0/v1/v2 差异
  - `main.cu`：被测内核表注册 `online_softmax_v2`（复用 `RowMap::kBlockPerRowShuffle`：`grid = rows`、smem = 0），同步 `kBlock` / `RowMap` / `GridFor` / `SmemFor` / 边界场景注释的版本口径
  - 全量回归 9 内核 × 20 场景 180 项全部通过（默认档 18 项通过），覆盖 float4 对齐边界、非 4 倍列宽标量回退、窄行空子集等路径
  - `operators/softmax/README.md`：状态表 / 版本规划 / 参考规模 / 目录布局 / 测试章节补 online-v2，online 开发采样替换为 v0/v1/v2 同场实测（online-v2 4096² 0.7098 ms / 189.10 GB/s、16384×1024 0.7560 ms / 177.55 GB/s，max_err 1.341e-06 / 1.392e-06；同场相对 online-v1 +15.2% / +9.9%）
  - 顶层 `README.md` / `AGENTS.md`：Softmax 状态同步为进行中（v0/v1/v2/v3/v4/v5 与 online-v0/v1/v2 完成，全量回归 180 项通过）
- 2026-09-10 10:25 `operators/softmax`：新增并接入 `online_softmax_v1`（块内协作版在线归约）
  - `online_softmax.cuh`：补 v1 声明与启动约束（每行一个 block、`row = blockIdx.x`、`grid = M`、`blockDim.x` 为 32 的倍数且 <= 1024、无动态共享内存；未分到元素的线程 / warp 以 `(m = -inf, d = 0)` 作归约单位元），文件头补单元内 v0 / v1 的行映射差异
  - `online_softmax.cu`：补 v1 注释并修复草稿缺陷 —— 提取 `mergeOnline`（二元组结合运算，空集一侧跳过缩放以规避 `(-inf)-(-inf)` 经 `expf` 污染整行分母）、`warpReduceOnline` / `blockReduceOnline` 两级 shuffle 合并并补齐「warp 值写入后」「结果广播前」两处 `__syncthreads`、补 `row >= M` 越界空转守卫；移除草稿遗留的 clang 内部头文件 include 并按 `.clang-format` 统一 Tab 缩进
  - `main.cu`：被测内核表注册 `online_softmax_v1`（复用 `RowMap::kBlockPerRowShuffle`：`grid = rows`、smem = 0），同步 `RowMap` / `GridFor` / `SmemFor` / 边界场景注释的版本口径
  - 全量回归 8 内核 × 20 场景 160 项全部通过（默认档 16 项通过），覆盖窄行空子集、warp 边界、`N=0` 空行等路径
  - `operators/softmax/README.md`：状态表 / 版本规划 / 参考规模 / 目录布局与测试 / 场景说明补 online-v1；online 开发采样表替换为 v0 / v1 同场实测（online-v1 4096² 0.6945 ms / 193.25 GB/s、16384×1024 0.8156 ms / 164.56 GB/s，max_err 1.329e-06 / 1.285e-06；online-v0 同场复测 3.9611 / 3.7119 ms，替换更早的同日采样）
  - 顶层 `README.md` / `AGENTS.md`：Softmax 状态同步为进行中（v0/v1/v2/v3/v4/v5 与 online-v0/v1 完成，全量回归 160 项通过）
- 2026-09-10 09:20 `operators/softmax`：新增 online softmax 独立实现单元并接入测试
  - `online_softmax.cuh`：新增 online softmax 接口声明与设计说明 —— “online” 指单趟在线归约：用「运行最大 m + 运行分母 d」二元组在同一趟遍历里增量维护行最大与 Σexp（`m' = max(m,x)`、`d' = d*exp(m-m') + exp(x-m')`），求 m 与求 Σexp 合并为一趟全局读；声明 `online_softmax_v0`（每行一个 block、`grid = M`、块内 warp shuffle 合并、无动态共享内存、`blockDim.x` 为 32 的倍数且 <= 1024）
  - `online_softmax.cu`：新增 `online_softmax_v0` 实现 —— 每线程以 stride 在线归约本线程子集；`warpMergeOnline` / `blockMergeOnline` 两级 warp shuffle 合并各线程 `(m,d)`（空子集以 `d == 0` 识别、合并时跳过，规避 `-inf - (-inf)` 的 NaN 传播），结果经静态 `__shared__` 广播回全体；第三次遍历重算 exp 归一化写回（全局读 2 遍 + 写 1 遍）。因 `-rdc=false` 无法跨翻译单元引用 `__device__`，该单元自带合并 helper、与 `softmax.cu` 的归约 helper 并列
  - `main.cu`：被测内核表注册 `online_softmax_v0`（复用 `RowMap::kBlockPerRowShuffle`：`grid = rows`、smem = 0），同步 `RowMap` / 文件头注释版本口径
  - 全量回归 7 内核 × 20 场景 140 项全部通过（默认档 14 项通过），覆盖空子集 / `N=0` / warp 边界等路径
  - `operators/softmax/README.md`：状态表 / 版本规划 / 参考规模 / 目录布局与测试章节补 online-v0，并追加 online softmax 单独开发采样（4096² 0.7245 ms / 185.25 GB/s、16384×1024 0.8119 ms / 165.31 GB/s，max_err 1.329e-06 / 1.285e-06）
  - 顶层 `README.md` / `AGENTS.md`：Softmax 状态同步为进行中（v0/v1/v2/v3/v4/v5 与 online-v0 完成，全量回归 140 项通过）
- 2026-09-09 20:10 `operators/softmax`：新增 `softmax_v5`（全局读 2 遍 + float4，动态共享内存只缓存 exp）并接入测试
  - `softmax.cuh`：补 v5 声明与启动约束（与 v4 同为每行一个 block + 整行动态共享内存缓存，但缓存内容是 exp 而非 x：① 全局读 1 遍求行最大、② 再全局读 1 遍算 exp 并 16 B 整写进 smem 累加行和、③ 从 smem 读 exp 归一化 float4 写回 —— 以多读 1 遍全局换掉 v4 的 x smem 写/读往返），同步头部版本演进说明
  - `softmax.cu`：追加 v5 实现并给实现要点注释；文件头版本清单补 v5
  - `main.cu`：被测内核表注册 v5（复用 `RowMap::kBlockPerRowRowCache`：`grid = rows`、smem = `cols * sizeof(float)`），场景注释与 SmemFor / 线程数注释版本口径同步到 v4/v5
  - `test.cuh` / `test.cu`：注释版本口径同步为 v0…v5（v4/v5 的 smem 均随列宽增长，分别缓存整行 x / exp）
  - 全量回归 6 内核 × 20 场景 120 项全部通过（默认档 12 项通过）
  - `operators/softmax/README.md`：状态表 / 版本规划 / 参考规模 / 测试章节补 v5，结论记录表回填同场开发采样（v5 4096² 0.7482 ms / 179.40 GB/s、宽行 0.6703 ms / 200.23 GB/s 为六版本同场最高，max_err 1.223e-06 / 1.237e-06 —— 宽行下缓存整行 / 只算 1 次 exp 的收益兑现，4096² 仍低于 v2/v3 的 ~190 GB/s，占用仍是主瓶颈）
  - 顶层 `README.md` / `AGENTS.md`：Softmax 状态同步为进行中（v0/v1/v2/v3/v4/v5 完成，全量回归 120 项通过）
- 2026-09-09 19:55 `operators/softmax`：注册并补齐 `softmax_v4` 的测试
  - `softmax.cuh`：补 v4 声明与启动约束（每行一个 block、两级 warp shuffle 归约；动态共享内存 `N * sizeof(float)` 缓存整行、全局读 1 遍 / 每元素只算 1 次 exp；默认 48 KiB 上限内免 opt-in），并同步头部版本演进说明
  - `main.cu`：被测内核表注册 v4 —— `RowMap` 新增 `kBlockPerRowRowCache`，`SmemFor` 改为随列宽取 `cols * sizeof(float)`；`GridFor` / 相关注释同步
  - `test.cuh` / `test.cu`：注释版本口径同步为 v0…v4（含 v4 的动态共享内存启动说明）
  - 全量回归 5 内核 × 20 场景 100 项全部通过（默认档 10 项通过）
  - `operators/softmax/README.md`：状态表 / 版本规划说明 / 参考规模 / 测试章节补 v4，结论记录表回填同场开发采样（v4 4096² 0.7828 ms / 171.45 GB/s、宽行 0.7482 ms / 179.40 GB/s，max_err 1.223e-06 / 1.237e-06，同场相对 v3 −8.0% / −11.8% —— 整行缓存少读行数的收益被 smem 往返与占用下降抵消）
  - 顶层 `README.md` / `AGENTS.md`：Softmax 状态同步为进行中（v0/v1/v2/v3/v4 完成，全量回归 100 项通过）
- 2026-09-04 14:18 仓库骨架与工程文档
  - 新增 `README.md`、`CHANGELOG.md`、`LICENSE`（MIT）、`.gitignore`、`.clang-format`
- 2026-09-04 14:18 CMake 构建框架
  - 顶层 `CMakeLists.txt`、`CMakePresets.json`（Release/Debug，Ninja）
  - `common/` 头文件库：CUDA 错误检查与计时工具
- 2026-09-04 14:18 `docs/benchmark-methodology.md`：统一正确性验证与性能基准口径
- 2026-09-04 14:18 算子目录骨架与规划：`operators/{softmax,gemm,attention}` 各自的 `README.md`
- 2026-09-04 14:18 新增 `operators/reduce` 与目标算子扩展
  - 新增 `operators/reduce`（`README.md` + `CMakeLists.txt`），顶层统一注册
  - 目标算子扩展为 Softmax / GEMM / Attention / Reduce
- 2026-09-04 14:18 算子构建约定
  - 每个算子目录自带 `CMakeLists.txt`，顶层统一注册
  - 目录内出现 `src/main.cu` 后自动参与构建，新增 `.cu` 无需改动 CMake
- 2026-09-06 12:58 `operators/reduce`：CUDA 核心版实现（v0/v1）
  - `reduce_cpu` 主机参考、`reduce_v0`（交错寻址共享内存树形归约）与 `reduce_v1`（连续寻址，消除 warp 内分歧）内核
  - 配套可复用测试驱动（`test.cuh/.cu`）与入口（`main.cu`），覆盖正常 / 边界 / 异常（空输入、grid 超配等）三类场景
  - 两内核 18 项测试全部通过
- 2026-09-06 12:58 `operators/reduce`：为 v0/v1 内核补齐注释并同步文档
  - 补齐模块、函数与关键逻辑注释（参数 / 返回值 / 启动约束 / 注意事项）
  - 同步 `CHANGELOG.md`、顶层与算子 `README.md`
- 2026-09-06 13:09 `operators/reduce`：测试入口 `main.cu` 场景开关化
  - 新增 `enable_boundary`（函数参数形式，默认关闭）——控制边界条件与异常 / 健壮性场景（默认只跑正常流程 4 项，开启后 18 项全量回归）
  - 新增 `strict_benchmark`（函数参数形式，默认关闭）——控制严格性能口径（透传给 `test_reduce_kernel`）
  - 重复场景循环收敛为 `RunScenarios`，场景表长度编译期推导
- 2026-09-06 14:49 `operators/reduce`：接入 `reduce_v3`
  - `reduce_v3`（每线程 2 元素展开）由实验性实现正式接入——签名统一为 `ReduceKernel`（`const float*`），声明加入 `reduce.cuh`、注册进 `main.cu` 被测内核表
  - `GridFor` 改为按内核“每线程元素数”计算覆盖 grid（v3 覆盖同一 `n` 所需 block 减半）
  - 同步实现注释、`test.cuh` 注释与算子 `README.md`（状态表 / 版本规划）
- 2026-09-07 10:58 `operators/reduce`：新增并接入 `reduce_v4`
  - `reduce_v4`（每线程 2 元素 + 末 warp 展开归约，volatile 共享内存收尾省去 5 轮 `__syncthreads`）——修正草稿的段基址与越界判定，声明加入 `reduce.cuh`、注册进 `main.cu` 被测内核表
  - 全量回归 5 内核 × 9 场景 45 项全部通过
  - `reduce_v4` 正常流程严格基准 251.5 GB/s（n=2^20）
  - 精简算子源码注释——`cu`/`cuh` 声明与实现注释去重、删除冗余推导细节（并入算子 `README.md` / `docs`），并同步 `test.cuh`/`test.cu`/`main.cu` 注释
- 2026-09-07 14:23 `operators/reduce`：新增并接入 `reduce_v5`
  - `reduce_v5`（v4 + 编译期常量化 block 尺寸）——规整草稿实现并补齐模块/行内注释
  - 因 `-rdc=false` 下跨翻译单元引用 `__global__` 模板特化已被 nvcc 弃用，`reduce_v5` 模板定义整体内联进 `reduce.cuh`（`main.cu` 以 `reduce_v5<kBlock>` 注册进被测内核表，同时同步 `test.cuh` 注释）
  - 全量回归 6 内核 × 9 场景 54 项全部通过
  - `operators/reduce/README.md`：状态表 v5 标记完成、原 `float4` 向量化规划顺延为 v6，结论记录表回填 v5 严格基准（n=2^20 对齐 272.3 GB/s，同场相对 v4 约 +4.6%）
  - 顶层 `README.md`：Reduce 状态同步为 v0…v5
- 2026-09-07 15:55 `operators/reduce`：新增并接入 `reduce_v6`
  - `reduce_v6`（每线程 2 元素 + 两级 warp shuffle）——规整草稿实现并补齐精简注释（移除实验期遗留的 clang 内部头文件 include）
  - 声明加入 `reduce.cuh`、注册进 `main.cu` 被测内核表（同步 `test.cuh` 注释与 block 约束说明）
  - 全量回归 7 内核 × 9 场景 63 项全部通过
  - `operators/reduce/README.md`：状态表 v6 标记完成、原 `float4` 向量化规划顺延为 v7，结论记录表回填 v6 严格基准（n=2^20 对齐 339.68 GB/s，同场 v4/v5 复测 310.89/331.46 → 相对 v5 约 +2.5%）
  - 顶层 `README.md`：Reduce 状态同步为 v0…v6
- 2026-09-07 16:22 `common/cuda_check.h`：`PrintDeviceInfo` 增加带宽相关输出
  - 增加显存时钟（kHz）/ 位宽（bit）/ 理论峰值带宽输出——`cudaDeviceProp` 无带宽字段，按 `2 × memoryClockRate × busWidth/8` 估算
  - 注明该估算仅为理论峰值，实际带宽需跑 benchmark 测得
- 2026-09-07 16:22 `operators/reduce/README.md`：结论记录表补带宽可比性警示
  - n=2^20 工作集仅 ~4 MiB、远小于本机 32 MB L2（预热后输入驻留 L2），有效带宽反映片上 L2 命中带宽而非显存物理带宽
  - v5/v6 超过 GDDR6 理论峰值 256 GB/s 系缓存命中所致、不违反物理上限；要逼近/验证显存带宽需改用远大于 L2 的规模
- 2026-09-07 16:36 `operators/reduce`：新增并接入 `reduce_v7`
  - `reduce_v7`（v6 两级 warp shuffle + `float4` 向量化加载与 grid-stride 扫描）——规整草稿实现（签名统一为 `ReduceKernel` 的 `const float*`，块内两次 shuffle 归约复用 `warpReduceSum`），补齐内核/接口注释
  - 声明加入 `reduce.cuh`、注册进 `main.cu` 被测内核表（每线程元素数 4，推荐 grid = `n/(4*block)` = 1024）
  - 边界场景补 `n=block-2` 覆盖 `float4` 尾部余 2 路径
  - 全量回归 8 内核 × 10 场景 80 项全部通过
  - `operators/reduce/README.md`：状态表 v7 标记完成，结论记录表回填 v7 严格基准（n=2^20 对齐 412.16 GB/s，同场 v5/v6 复测 329.99/341.39 → 相对 v6 约 +20.7%）
  - 顶层 `README.md`：Reduce 状态同步为 v0…v7
- 2026-09-08 13:09 `operators/reduce`：新增讲解型学习文档 `notes/reduce.md`
  - v0→v7 每版按「瓶颈 → 动机 → 设计 → 关键实现 → 约束 → 实测」展开，含问题性能模型、两阶段归约设计说明、演进小结与瓶颈迁移主线、陷阱清单、扩展方向，与算子 `README.md`（规划 + 状态 + 结论表）分工互补
  - 同步 `README.md` 目录布局并更正「notes/ 尚未创建」注记
- 2026-09-08 14:10 `operators/softmax`：实现并接入 `softmax_v0`
  - `softmax_v0`（每行一个 block，行内列维由 blockDim 线程以 stride 协同遍历；共享内存树形归约两次——先求行最大、再求 Σexp，max-shift 数值稳定，写回时第三次读行重算 exp，作正确性基线）与 CPU 参考 `softmax_cpu`
  - 补齐内核/接口/模块注释（移除草稿遗留的 clang 内部头文件 include）
  - 新增仿 reduce 的可复用测试驱动 `test.cuh/.cu`（函数指针统一驱动、逐元素相对误差 ≤ 1e-5、跳过 `|ref|<1e-30`、开发 1+100 / 严格 1000+21×2000 两档采样）
  - 新增入口 `main.cu`（A 正常 / B 边界 / C 异常——空矩阵与空行，默认 2 项、全量回归 10 项全部通过）
  - `operators/softmax/README.md`：状态表 v0 标记完成、目录布局与测试章节补齐，结论记录表回填 v0 严格基准（4096×4096 0.7462 ms / 179.87 GB/s，16384×1024 0.8267 ms / 162.36 GB/s，max_err≈1.2e-6）
  - 顶层 `README.md`：Softmax 状态同步为进行中（v0 完成）
- 2026-09-08 14:38 `operators/softmax`：纠正 v0 实现
  - v0 恢复为最初实现（每线程处理一行，行内串行三遍：求行最大 → 累加 Σexp → 归一化写回；无共享内存/同步，算法代码原样保留，仅移除 nvcc 无法解析的 clang 内部头文件 include 并补充注释），不再采用“每行一个 block + 共享内存树形归约”
  - 测试驱动/入口随映射方式调整——`grid = ceil(rows/block)`、严格档迭代与 reduce 一致（1000 预热 + 21 组 × 10000 次）、去掉共享内存启动配置
  - 边界场景改为围绕行数在 block 线程覆盖边界（block±1、恰满载、末 block 余 1 行）
  - 全量回归 10/10 PASS；默认形状 max_err 5.478e-06（4096×4096）/ 2.537e-06（16384×1024），仍满足 1e-5 容差但已同量级（fp32 串行累加误差随行宽增长，记为 v1/v2 优化动机）
  - `operators/softmax/README.md`：同步 v0 描述与开发采样实测（4.87 ms / 27.56 GB/s、4.24 ms / 31.69 GB/s，max_err 同场实测）
- 2026-09-08 15:10 `operators/softmax`：新增并接入 `softmax_v1`
  - `softmax_v1`（每行一个 block，行内 `blockDim.x` 线程以 stride 协同遍历，行最大与 Σexp 各经一次共享内存折半树形归约，写回时第三次读行重算 exp）——规整草稿实现（补 `row >= M` 越界空转守卫、修正 smem 写入后的同步顺序）
  - `softmax.cu` 注释精简为行内/短注释并移除草稿遗留的 clang 内部头文件 include，声明加入 `softmax.cuh`（含启动 grid 与动态共享内存契约）
  - 测试驱动 `test.cuh/.cu` 增参 `grid`/`smem_bytes`（v0 每线程一行 `grid=ceil(rows/block)`、无共享内存；v1 每行一个 block `grid=rows`、smem=`blockDim.x*sizeof(float)`，由 `main.cu` 的 `RowMap` 推出）
  - `main.cu` 注册 v0/v1 并补行宽维边界场景
  - 全量回归 2 内核 × 13 场景 26 项全部通过
  - `operators/softmax/README.md`：状态表 v1 标记完成、目录/测试章节补双版本启动口径、结论表回填同场开发采样（v0 4096×4096 4.9560 ms/27.08 GB/s vs v1 0.6816 ms/196.93 GB/s，max_err 5.478e-06 → 1.231e-06）
  - 顶层 `README.md`：Softmax 状态同步为进行中（v0/v1 完成）
- 2026-09-08 15:48 `operators/softmax`：新增并接入 `softmax_v2`
  - `softmax_v2`（每行一个 block，行内遍历同 v1——三次 stride 扫行；块内归约由共享内存折半树形改两级 warp shuffle：`warpReduceMax/Sum` 每 warp 归为 1 值，warp 0 再归约 `num_warps` 个 warp 值并经静态 `__shared__` 广播回全体线程，归约在寄存器内完成、块内同步降为每趟 2 次）
  - 规整草稿实现为仓库注释风格——补齐 helper/内核中文注释、签名统一为 `const float*`、补 `row >= M` 越界守卫、移除草稿遗留的 clang 内部头文件 include
  - 声明加入 `softmax.cuh`（启动约束：`grid = rows`、`blockDim.x` 为 32 的倍数且 <= 1024、无动态共享内存）
  - `main.cu` 以 `RowMap::kBlockPerRowShuffle` 注册进被测内核表（`GridFor`/`SmemFor` 按行映射与共享内存分别推出启动配置）
  - 边界场景补行宽 31/32/33 的 warp 边界组覆盖 v2 整 warp 空转的归约单位元路径，测试驱动注释同步 v0/v1/v2 三版本启动口径
  - 全量回归 3 内核 × 16 场景 48 项全部通过
  - `operators/softmax/README.md`：状态表 v2 标记完成、原 `float4` 向量化规划顺延为 v3，结论记录表回填同场开发采样（v0 5.2101/3.7837 ms、25.76/35.47 GB/s，v1 0.7149/0.6912 ms、187.74/194.18 GB/s，v2 0.6938/0.6777 ms、193.45/198.05 GB/s，4096×4096 与 16384×1024；v2 max_err 1.232e-06/1.244e-06，同场相对 v1 +3.0%/+2.0%）
  - 顶层 `README.md`：Softmax 状态同步为进行中（v0/v1/v2 完成）
- 2026-09-09 16:54 `operators/softmax`：新增并接入 `softmax_v3`
  - `softmax_v3`（v2 + `float4` 向量化）——行映射与两级 warp shuffle 归约同 v2，行内访问按列宽分派：列宽为 4 的倍数（行首 16 B 对齐）时主循环以 stride 取 `float4`（读/写指令数为标量 1/4），否则整行回退 v2 式标量三遍（非 4 倍列宽时跨行行首不对齐，`float4` 属未定义行为，尾列处理救不了跨行对齐），任意列宽均正确
  - 规整草稿实现为仓库注释风格——补 `row >= M` 越界守卫、修正行和归约误用 `blockReduceMaxShuffle` 为 `blockReduceSumShuffle`、补 ② Σexp 遍累加、移除草稿遗留的 clang 内部头文件 include
  - 声明加入 `softmax.cuh`（启动约束同 v2：`grid = rows`、blockDim 32 的倍数且 <= 1024、无动态共享内存），`main.cu` 以 `RowMap::kBlockPerRowShuffle` 注册进被测内核表
  - 边界场景补 float4 对齐边界组（列宽 `4×(block±1)`、`4×block`、`4×(2×block-1)`——向量主循环空转 / 恰 1 轮满载 / 第 2 轮余 1 个与余 `block-1` 个 `float4`）覆盖 v3 向量化路径
  - 全量回归 4 内核 × 20 场景 80 项全部通过
  - `operators/softmax/README.md`：状态表 v3 标记完成、原规划段更新为按列宽分派的实现说明，结论记录表刷新为含 v3 的同场开发采样（2026-09-09：v0 5.2876/4.0149 ms、25.38/33.43 GB/s，v1 0.8342/0.8327 ms、160.89/161.18 GB/s，v2 0.7279/0.7163 ms、184.39/187.38 GB/s，v3 0.6904/0.7073 ms、194.41/189.76 GB/s，4096×4096 与 16384×1024；v3 max_err 1.255e-06/1.237e-06，同场相对 v2 +5.4%/+1.3%）
  - 顶层 `README.md`：Softmax 状态同步为进行中（v0/v1/v2/v3 完成）

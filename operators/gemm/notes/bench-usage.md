# GEMM `--bench` 扫描模式用法

> 定位：本文是 `operators/gemm/README.md` 的配套操作说明，只讲 `gemm --bench` 这条路径的
> **命令行用法、产物格式与复现步骤**。版本状态与性能结论看 README，正确性/性能的统一度量
> 口径看 `docs/benchmark-methodology.md`；实现细节见 `src/main.cu`（选项解析、扫描循环、
> CSV 写出）与 `src/test.cu`（`bench_gemm_kernel` / `bench_cublas_sgemm` 计时）。

## 1. 前置：编译

```bash
cd /home/loker1/my_operator        # 所有命令都在仓库根执行（默认 CSV 是相对路径）
cmake --preset release            # 首次构建或增删 .cu 后需要重新 configure
cmake --build build --target gemm
```

产物为 `build/operators/gemm/gemm`。同一份可执行文件有两条入口：

| 入口 | 行为 |
| --- | --- |
| 不带参数 | 固定 `512×512×512`：cuBLAS 对照 + 各内核的**正确性验证 + 性能基准** |
| `--bench` | 多尺寸扫描：**只计时，不做正确性校验**，最后写出 CSV |

`--bench` 必须出现在**第一个参数位**（`main.cu` 只比较 `argv[1]`）；其余选项随意顺序，跟在
`--bench` 之后。发现未知选项、缺参数或取值非法时打印 usage 并以退出码 `2` 结束。

## 2. 最简用法

```bash
./build/operators/gemm/gemm --bench
```

默认参数下的行为：尺寸 `128,256,512,1024,2048,4096`；每点 1 次预热 + 自适应迭代；结果写到
`operators/gemm/bench/<YYYYmmddHHMMSS>/gemm_bench.csv`，终端会打印实际路径。

## 3. 参数

| 参数 | 默认 | 含义与约束 |
| --- | --- | --- |
| `--sizes a,b,c` | `128,256,512,1024,2048,4096` | 逗号分隔的正方形边长（M=N=K）。每个值必须 ≥1，非法即退出码 2；写完列表后不能为空 |
| `--csv <path>` | `operators/gemm/bench/<时间戳>/gemm_bench.csv` | 输出路径，父目录自动创建；`fopen` 失败时 CSV 退回标准输出（终端管道仍可用） |
| `--warmup <n>` | `1` | 每个采样点计时前的空转调用次数，`n ≥ 0`；只影响进入计时前的时钟/缓存状态 |
| `--budget <ms>` | `200` | 单采样点的计时预算，`> 0`；只用于反推每组迭代数，不改变采样组数（固定 3 组取中位数） |

组合示例（只跑中等尺寸、加长预热、放宽预算、输出到临时路径）：

```bash
./build/operators/gemm/gemm --bench --sizes 512,1024,2048 --warmup 10 --budget 500 --csv /tmp/gemm_bench.csv
```

## 4. 终端输出样例

下面是一次真实的最小复现（`--sizes 128 --budget 50`，2026-09-15 采集；该次扫描的临时产物
未入库，仅用于展示输出格式）：

```text
==== GEMM bench sweep (performance only, no correctness check) ====
sizes: 128
sampling: 1 warmup, adaptive iterations (budget 50 ms per point, median of 3 samples)

  cuBLAS       128^3:    0.0184 ms    228.19 GFLOP/s (x100 iters)
  sgemm_v0     128^3:    0.0369 ms    113.53 GFLOP/s (x100 iters)
  sgemm_v1     128^3:    0.0174 ms    240.80 GFLOP/s (x100 iters)

CSV written to operators/gemm/bench/20260915150632/gemm_bench.csv
points: 3, skipped: 0
```

采样顺序固定：先 cuBLAS 对照，再按 `src/main.cu` 的 `kKernels` 表顺序逐内核、逐尺寸。
启动失败的点打印 `SKIPPED (launch or configuration failed)` 并跳过，不写入 CSV。

## 5. 产物格式

每次扫描各占一个时间戳目录，多次扫描互不覆盖；CSV 列固定四列：

| 列 | 含义 |
| --- | --- |
| `label` | 内核短名（`kKernels` 的 `plot_name`）或 `cuBLAS` |
| `size` | 正方形边长，形状为 `size×size×size` |
| `median_ms` | 单次调用耗时的中位数（ms） |
| `gflops` | `2·size³ / (median_ms × 10⁶)` |
| `iters` | 该采样点每组实际迭代数（自适应结果，见 §7） |

已入库的完整扫描示例：`operators/gemm/bench/20260915145532/gemm_bench.csv`。

## 6. 出图

```bash
pip install -r scripts/requirements.txt        # 首次
python3 scripts/plot_kernel_perf.py --csv operators/gemm/bench/20260915145532/gemm_bench.csv
```

绘图脚本只依赖 `label,size,gflops` 三列，不传 `--out` 时把 PNG 写在 CSV 同目录同名。
**不要用通配符把多次扫描的 CSV 一起传入**：`(label, size)` 重复会直接报错。配色、标题与
坐标轴文字可用 `--palette` / `--title` / `--xlabel` / `--ylabel` 覆盖。

## 7. 计时口径（为什么是「自适应迭代」）

`src/test.cu:MeasureKernelAdaptive` 的实际流程：

1. 预热 `--warmup` 次并同步；
2. 取 1 次调用估计单次耗时 `single_ms`；
3. 每组迭代数 `iters = clamp(round(budget_ms / (3 × single_ms)), 1, 100)`；
4. 连续采 3 组，每组 `iters` 次调用，取单次耗时的**中位数**。

固定的「100 次迭代」在大尺寸下不可用：朴素 v0 在 4096³ 下单次约 0.95 s，固定 100 次会让
**单个采样点**就耗掉数分钟。改用预算反推后，慢内核自动退化为每组 1–3 次迭代——趋势可用，
但样本量小、抖动大，因此扫描结果只用于画曲线看趋势；严格性能对比走
`docs/benchmark-methodology.md` 的 100 预热 / 21 组 × 1000 次口径（`test_gemm_kernel`
的 `strict_benchmark = true`）。

## 8. 常见坑

- **不在仓库根执行**：默认 CSV 是相对路径，会在当前目录下新建 `operators/gemm/bench/...`。
- **扫描不校验正确性**：曲线好看 ≠ 算得对；判据只在默认入口（正确性 + 性能）里执行。
  扫描模式的输入以 0 填充（避免未初始化内存里的 NaN / 非规格化数干扰吞吐），不生成主机端
  输入、不做 CPU 参考、不回拷结果。
- **`--sizes` 不能有非法值**：`0`、负数、非数字都会打印 usage 并返回 `2`（列表末尾多一个
  逗号也会命中）。
- **退出码语义**：`0` = 至少一个采样点成功；`1` = 全部采样点失败；`2` = 参数非法。部分内核
  失败不影响整体成功。
- **样本量随尺寸缩水**：大尺寸点的 `iters` 被预算压到 1–3，和开发阶段固定 100 次的口径
  不可直接比较；同机同构建内比较趋势即可。

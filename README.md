# my_operator

CUDA 并行算子学习项目。目前包含共享内存归约（Reduction）的多个逐步优化版本，用于对比线程访问方式、同步开销和每线程处理元素数对性能的影响。

## 内容

| 目录 / 目标 | 说明 |
| --- | --- |
| `Reduce` / `Reduce.exe` | CUDA 浮点求和归约的正确性验证与性能测试。 |
| `Sgemm` / `Sgemm.exe` | SGEMM 实验入口（当前为基础占位程序）。 |

`Reduce/src/reduce.cu` 提供了以下内核：

| 内核 | 策略 |
| --- | --- |
| `reduce_v0` | 交错寻址归约，用作基础对照。 |
| `reduce_v1` | 连续活跃线程的树形归约。 |
| `reduce_v2` | 从大步长到小步长的顺序寻址归约。 |
| `reduce_v3` | 每线程先在寄存器中累加两个元素，再执行 v2 风格的共享内存归约。 |

所有版本先在 GPU 上产生每个 block 的部分和，最终总和由主机端累加并与 CPU 参考值比较。

## 环境要求

- 支持 CUDA 的 NVIDIA GPU 与 CUDA Toolkit
- CMake 3.24 或更高版本
- 支持 C++17 的编译器；Windows 下建议使用 MSVC x64 开发者命令行

## 构建与运行

在 Windows 的 **x64 Native Tools Command Prompt for VS** 或 **Developer PowerShell for VS** 中执行：

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target Reduce
.\build\Reduce.exe
```

若未使用 Ninja，可省略 `-G Ninja`，由 CMake 选择本机可用生成器。

## 归约测试配置

`main` 中可调整以下参数：

```cpp
constexpr int n = 1 << 20;
constexpr int blockSize = 256;
constexpr bool strictBenchmark = true;
```

- `strictBenchmark = false`：1 次预热、100 次内核迭代，适合快速验证。
- `strictBenchmark = true`：1000 次预热、21 组 × 10000 次迭代，输出中位数及 P5/P95 时间范围，适合稳定地比较版本性能。
- `v0`~`v2` 使用 `ceil(n / blockSize)` 个 block；`v3` 因每线程处理两个元素，使用 `ceil(n / (2 * blockSize))` 个 block。

归约循环假定 `blockSize` 为 2 的幂。性能输出中的带宽是按输入读取和部分和写回计算的**有效带宽**，并非显存物理带宽。

## 许可证

当前仓库尚未声明许可证；如需复用或发布，请先补充合适的 LICENSE 文件。

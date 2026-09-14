# Online Softmax 内核拆分实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将六个 online softmax CUDA 内核拆成同名 `.cuh/.cu`，并将共享在线归约逻辑集中到一个公共头文件。

**架构：** 每个版本头文件只暴露该版本 kernel 声明和启动契约；同名实现文件只定义该 kernel。`online_softmax_common.cuh` 内联定义可跨翻译单元复用的设备端归约工具；不保留聚合头或聚合实现。

**技术栈：** CUDA C++17、nvcc、CMake/Ninja、PowerShell 文件布局断言。

---

## 文件结构

- 创建：`operators/softmax/src/online_softmax_common.cuh` — 内联在线二元组合并、warp/block 归约和 `kRegTile`。
- 创建：`operators/softmax/src/online_softmax_v0.cuh/.cu` — 每线程一行的 online softmax。
- 创建：`operators/softmax/src/online_softmax_v1.cuh/.cu` — 块内在线归约版。
- 创建：`operators/softmax/src/online_softmax_v2.cuh/.cu` — float4 分派版。
- 创建：`operators/softmax/src/online_softmax_v3.cuh/.cu` — 真寄存器分片版。
- 创建：`operators/softmax/src/online_softmax_v3_false.cuh/.cu` — local-memory 对照版。
- 创建：`operators/softmax/src/online_softmax_v4.cuh/.cu` — grid-stride 多行版。
- 修改：`operators/softmax/src/main.cu` — 显式包含六个版本头，不再包含聚合头。
- 删除：`operators/softmax/src/online_softmax.cuh/.cu` — 已被新文件替代的聚合单元。

### 任务 1：以布局断言驱动拆分

**文件：**
- 创建：上述 13 个新文件
- 删除：`operators/softmax/src/online_softmax.cuh/.cu`
- 修改：`operators/softmax/src/main.cu`

- [x] **步骤 1：运行失败的文件布局断言**

```powershell
$required = @(
  'online_softmax_common.cuh',
  'online_softmax_v0.cuh', 'online_softmax_v0.cu',
  'online_softmax_v1.cuh', 'online_softmax_v1.cu',
  'online_softmax_v2.cuh', 'online_softmax_v2.cu',
  'online_softmax_v3.cuh', 'online_softmax_v3.cu',
  'online_softmax_v3_false.cuh', 'online_softmax_v3_false.cu',
  'online_softmax_v4.cuh', 'online_softmax_v4.cu')
$src = 'operators/softmax/src'
$missing = $required | Where-Object { -not (Test-Path (Join-Path $src $_)) }
if ($missing) { throw "missing: $($missing -join ', ')" }
if ((Test-Path (Join-Path $src 'online_softmax.cuh')) -or
    (Test-Path (Join-Path $src 'online_softmax.cu'))) { throw 'legacy aggregate remains' }
```

预期：失败并列出尚未创建的同名单元。

- [x] **步骤 2：创建公共设备端实现头**

```cpp
// online_softmax_common.cuh
__device__ __forceinline__ void mergeOnline(float& m, float& d, float m2, float d2);
__device__ __forceinline__ void warpReduceOnline(float& m, float& d);
__device__ __forceinline__ void blockReduceOnline(float& m, float& d);
inline constexpr int kRegTile = 16;
```

将旧文件中的实际函数体移入该头，并保持空集合 `(m=-inf,d=0)` 的 NaN 防护与静态共享内存归约不变。

- [x] **步骤 3：创建版本配对文件**

```cpp
// online_softmax_v2.cuh
#pragma once
#include <cuda_runtime.h>
__global__ void online_softmax_v2(const float* input, float* output, int M, int N);

// online_softmax_v2.cu
#include "online_softmax_v2.cuh"
#include "online_softmax_common.cuh"
__global__ void online_softmax_v2(const float* input, float* output, int M, int N) {
    // N%4==0 时以 float4 扫描与写回，否则以标量 stride 扫描与写回。
}
```

对 v0、v1、v2、v3、v3_false、v4 分别完成同样的头/源配对；将各版本原有启动约束转移到自己的 `.cuh` 文件头注释。

- [x] **步骤 4：删除旧聚合单元**

```text
operators/softmax/src/online_softmax.cuh
operators/softmax/src/online_softmax.cu
```

- [x] **步骤 5：重新运行布局断言**

运行：步骤 1 的 PowerShell 脚本。

预期：退出码 0；13 个新文件存在，两个聚合文件不存在。

### 任务 2：验证独立翻译单元与 CMake 门控

**文件：**
- 验证：任务 1 创建的六个 `.cu`
- 验证：`operators/softmax/CMakeLists.txt`

- [x] **步骤 1：以 nvcc 单独编译六个实现单元**

```powershell
$src = 'operators/softmax/src'
$out = Join-Path $env:TEMP 'online-softmax-split-obj'
New-Item -ItemType Directory -Force $out | Out-Null
'v0','v1','v2','v3','v3_false','v4' | ForEach-Object {
  nvcc --std=c++17 -c (Join-Path $src "online_softmax_$_.cu") `
    -o (Join-Path $out "online_softmax_$_.o")
  if ($LASTEXITCODE -ne 0) { throw "nvcc failed: $_" }
}
```

预期：六次 `nvcc -c` 均退出 0，证明公共设备函数可被各独立翻译单元使用。

- [x] **步骤 2：重新 configure 并确认 Softmax 门控状态**

```bash
cmake -S . -B /tmp/my_operator_softmax_baseline_01a09da7 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DSOFTMAX_WITH_CUDNN=OFF
cmake --build /tmp/my_operator_softmax_baseline_01a09da7 --target softmax
/tmp/my_operator_softmax_baseline_01a09da7/operators/softmax/softmax
```

预期：重新 configure 后将新增的六个 `.cu` 纳入 `softmax`；构建成功后运行
`build/operators/softmax/softmax`，保留其现有正确性回归结果。

- [x] **步骤 3：检查变更范围**

```powershell
git diff --check
git status --short -- operators/softmax/src docs/superpowers
```

预期：无空白错误；仅出现本计划、规格和本次拆分新增/删除的 online softmax 文件，既有删除的 `main.cu`、`test.cu`、`test.cuh` 保持未恢复。

# Softmax 内核拆分实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将普通 Softmax 的六个 CUDA kernel 拆为同名独立实现单元，并分离其支持代码。

**架构：** 共享两级 shuffle 归约内联定义在 `softmax_common.cuh`；CPU 与可选 cuDNN 主机 API
由单独支持单元提供。入口和测试接口显式包含所需头文件，避免重新引入聚合头。

**技术栈：** CUDA C++17、CMake/Ninja、WSL CUDA 12.9、PowerShell 文件布局断言。

---

### 任务 1：以文件布局断言驱动拆分

**文件：**
- 创建：`operators/softmax/src/softmax_common.cuh`、`softmax_reference.cuh/.cu`、
  `softmax_cudnn.cuh/.cu`、`softmax_v0.cuh/.cu` 至 `softmax_v5.cuh/.cu`
- 修改：`operators/softmax/src/main.cu`、`operators/softmax/src/test.cuh`
- 删除：`operators/softmax/src/softmax.cuh/.cu`

- [x] **步骤 1：运行失败的布局断言**

```powershell
$src = 'operators/softmax/src'
$required = @('softmax_common.cuh', 'softmax_reference.cuh', 'softmax_reference.cu',
  'softmax_cudnn.cuh', 'softmax_cudnn.cu',
  'softmax_v0.cuh', 'softmax_v0.cu', 'softmax_v1.cuh', 'softmax_v1.cu',
  'softmax_v2.cuh', 'softmax_v2.cu', 'softmax_v3.cuh', 'softmax_v3.cu',
  'softmax_v4.cuh', 'softmax_v4.cu', 'softmax_v5.cuh', 'softmax_v5.cu')
$missing = $required | Where-Object { -not (Test-Path (Join-Path $src $_)) }
if ($missing) { throw "missing: $($missing -join ', ')" }
```

预期：失败并列出尚未创建的文件。

- [x] **步骤 2：迁移支持实现**

```cpp
// softmax_common.cuh
static __device__ __forceinline__ float blockReduceMaxShuffle(float value);
static __device__ __forceinline__ float blockReduceSumShuffle(float value);

// softmax_reference.cuh
using SoftmaxKernel = void (*)(const float*, float*, int, int);
void softmax_cpu(const float* input, float* output, int M, int N);
```

将原 CPU double 参考、cuDNN 的 `MODE_INSTANCE` 映射和两级 shuffle 归约原样迁移到各自职责文件。

- [x] **步骤 3：迁移六个 kernel 与消费者包含关系**

```cpp
// softmax_v3.cu
#include "softmax_common.cuh"
#include "softmax_v3.cuh"
__global__ void softmax_v3(const float* input, float* output, int M, int N) {
    // N%4==0 时 float4 三遍，否则标量三遍。
}
```

`main.cu` 显式包含 v0–v5 和 cuDNN 头；`test.cuh` 只包含参考接口头。

- [x] **步骤 4：删除聚合文件并重跑布局断言**

预期：所有新文件存在，`softmax.cuh/.cu` 不存在。

### 任务 2：验证独立翻译单元与回归

- [x] **步骤 1：单独编译六个 kernel 源文件**

```bash
nvcc --std=c++17 -c operators/softmax/src/softmax_v0.cu -o /tmp/softmax_v0.o
# 对 softmax_v1.cu 至 softmax_v5.cu 重复上述命令。
```

预期：六个对象文件均成功生成。

- [x] **步骤 2：重新 configure、构建并运行默认 Softmax 回归**

```bash
cmake -S . -B /tmp/my_operator_softmax_baseline_01a09da7 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DSOFTMAX_WITH_CUDNN=OFF
cmake --build /tmp/my_operator_softmax_baseline_01a09da7 --target softmax
/tmp/my_operator_softmax_baseline_01a09da7/operators/softmax/softmax
```

预期：构建成功，12 个内核 × 2 个正常场景均通过。

- [x] **步骤 3：检查格式、遗留包含和变更空白**

```bash
clang-format --dry-run --Werror operators/softmax/src/softmax_*.cu operators/softmax/src/softmax_*.cuh
git diff --check
```

预期：无格式或空白错误，也没有 `#include "softmax.cuh"`。

# Softmax 单报告 GPU 指标实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 为每个 Softmax 测试结果块打印缓存的 GPU 配置、启动配置和完整英文性能/正确性报告。

**架构：** `test.cu` 查询一次设备快照并在每个测试调用的报告中复用；`main.cu` 不再打印无归属的设备信息。计算、比较和计时语义保持不变。

**技术栈：** C++17、CUDA Runtime API、CUDA Events、PowerShell、CMake/Ninja。

---

## 文件结构

- 修改：`operators/softmax/src/test.cu`——缓存设备快照、统一报告状态和英文渲染。
- 修改：`operators/softmax/src/test.cuh`——报告行为与空形状语义说明。
- 修改：`operators/softmax/src/main.cu`——移除顶层 `PrintDeviceInfo()`。
- 修改：`operators/softmax/README.md`、`CHANGELOG.md`——说明输出契约并记录实际验证结果。

### 任务 1：建立会在旧实现失败的输出契约

**文件：** 构建后的 `build/operators/softmax/softmax` 标准输出。

- [x] 构建并运行当前 Softmax 可执行文件；旧实现的真实输出为 24 份报告、0 个
  `GPU Configuration` 段，已按预期失败。

```powershell
cmake --build build --target softmax
$output = & ./build/operators/softmax/softmax 2>&1
$text = $output -join "`n"
$reports = ([regex]::Matches($text, '(?m)^\[.*\] rows=')).Count
$gpuSections = ([regex]::Matches($text, '(?m)^GPU Configuration$')).Count
if ($reports -lt 1 -or $gpuSections -ne $reports) {
    throw "Expected one GPU Configuration section per report; reports=$reports, gpu_sections=$gpuSections"
}
```

预期：失败。旧实现只在 `main()` 开头调用一次 `PrintDeviceInfo()`，没有每报告的 GPU 段。

### 任务 2：实现独立的英文报告块

**文件：**
- 修改：`operators/softmax/src/test.cu`
- 修改：`operators/softmax/src/test.cuh`
- 修改：`operators/softmax/src/main.cu`

- [x] 定义缓存设备快照；每个返回路径均输出以下顺序：

```text
Test Case
GPU Configuration
Launch Configuration
Results
Validation
```

- [x] 正常运行显示截图的七项英文指标；空形状和非法参数显示 `N/A`，不伪造测量值。
- [x] 移除顶层设备打印，使用独立的 WSL Release 构建目录运行：

```powershell
clang-format -i --style=file operators/softmax/src/test.cu operators/softmax/src/test.cuh operators/softmax/src/main.cu
cmake --build build --target softmax
```

### 任务 3：验证与文档同步

**文件：**
- 修改：`operators/softmax/README.md`
- 修改：`CHANGELOG.md`

- [x] 运行真实程序：24 个 `Test Case:` 对应 24 个 `GPU Configuration`，包含七项截图指标，
  并以 `==== Result: 24/24 items PASS ====` 结束。
- [x] 运行 `compute-sanitizer --tool memcheck`：退出码 0，`ERROR SUMMARY: 0 errors`。
- [x] 同步 README、CHANGELOG，最后运行 `git diff --check` 检查格式。

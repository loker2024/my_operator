# GEMM v0 测试与入口实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 为行主序 SGEMM v0 补齐可复用的正确性与性能测试、公开头文件和固定两场景的可执行入口。

**架构：** `sgemm_v0.cuh` 公开 CUDA 内核签名，`sgemm_reference.cuh/.cu` 提供独立的 double 累加 CPU 参考，`test.cuh/.cu` 负责确定性输入、计时和逐元素比较，`main.cu` 只注册内核与场景。测试驱动通过 `cudaLaunchKernel` 接收内核函数指针，使后续 v1/v2 可复用同一流程。

**技术栈：** CUDA Runtime、C++17、`operator_common` 的 CPU/GPU 计时器、CMake/Ninja。

---

## 文件结构

- 创建：`operators/gemm/src/include/sgemm_v0.cuh` —— v0 的行主序接口与启动契约。
- 创建：`operators/gemm/src/include/sgemm_reference.cuh` —— 内核函数指针类型与 CPU 参考接口。
- 创建：`operators/gemm/src/include/test.cuh` —— 单内核单场景测试接口。
- 创建：`operators/gemm/src/sgemm_reference.cu` —— double 累加 CPU 参考实现。
- 创建：`operators/gemm/src/test.cu` —— 可复用测试与性能报告。
- 修改：`operators/gemm/src/sgemm_v0.cu` —— 写回累加结果并包含公开声明。
- 修改：`operators/gemm/src/main.cu` —— 注册 v0、配置 16×16 block 和两个固定场景。
- 修改：`operators/gemm/README.md`、`README.md`、`CHANGELOG.md` —— 同步实际状态与测试入口。

### 任务 1：建立会失败的 v0 集成测试入口

**文件：**
- 创建：`operators/gemm/src/include/sgemm_v0.cuh`
- 创建：`operators/gemm/src/include/sgemm_reference.cuh`
- 创建：`operators/gemm/src/include/test.cuh`
- 创建：`operators/gemm/src/sgemm_reference.cu`
- 创建：`operators/gemm/src/test.cu`
- 创建：`operators/gemm/src/main.cu`

- [x] **步骤 1：编写调用真实 CUDA 内核的测试驱动与入口**

在 `main.cu` 注册 `sgemm_v0`，运行 `512×512×512` 和 `513×511×509`；`test_gemm_kernel` 用独立 CPU 参考比较每个 `C[row*N+col]`。测试要捕捉的破坏是“内核没有把累加值写到 C”，该破坏会使 GPU 输出与 CPU 参考不一致。

- [x] **步骤 2：重新 configure、构建并验证预期失败**

运行：`cmake --preset release && cmake --build build --target gemm && ./build/operators/gemm/gemm`

预期：构建或运行失败，原因是 v0 尚无可供入口包含的声明或未写回 `C`。

### 任务 2：以最小 v0 实现满足测试

**文件：**
- 修改：`operators/gemm/src/sgemm_v0.cu`
- 修改：`operators/gemm/src/include/sgemm_v0.cuh`

- [x] **步骤 1：声明并实现行主序 v0 契约**

内核签名固定为：

```cpp
__global__ void sgemm_v0(const float* A, const float* B, float* C, int M, int N, int K);
```

启动为 `grid=((M+15)/16, (N+15)/16)`、`block=(16,16)`、动态共享内存为 0；边界线程不读写。内层从 `k=0` 到 `K-1` 进行 fp32 累加，随后执行 `C[row * N + col] = sum`。

- [x] **步骤 2：重新 configure、构建并验证通过**

运行：`cmake --preset release && cmake --build build --target gemm && ./build/operators/gemm/gemm`

预期：两个场景均报告 `PASS`，最大相对误差不超过 `1e-3`，进程退出码为 0。

### 任务 3：同步项目说明并复核

**文件：**
- 修改：`operators/gemm/README.md`
- 修改：`README.md`
- 修改：`CHANGELOG.md`

- [x] **步骤 1：更新实际状态与测试口径**

把 GEMM v0 标为“完成（已接入测试）”，说明默认入口的两组快速回归形状、容差、采样为 1 次预热加 100 次迭代；不填入跨本次运行不可复现的性能结论。

- [x] **步骤 2：运行最终验证与格式检查**

运行：`cmake --preset release && cmake --build build --target gemm && ./build/operators/gemm/gemm`，随后执行 `git diff --check` 和 `git status --short`。

预期：构建退出码 0、GEMM 两个测试项通过、无空白错误；状态仅包含本任务文件与既有无关改动。

# Softmax 单报告 GPU 指标设计

## 目标

将 Softmax 测试驱动的每个 `kernel × scenario` 报告改为独立、可复制的英文结果块。每块在性能与正确性结论之前显示同一份 GPU 静态配置和本次启动配置，并保留已有的截图七项指标。

## 范围与约束

- 只修改 Softmax 测试/报告层，不改 CUDA 内核、场景矩阵、启动映射、采样参数或判据。
- 终端运行时文本全部使用英文；源码注释、README 与 CHANGELOG 继续使用项目的中文规范。
- GPU 静态属性由 CUDA Runtime 查询一次并缓存；每份报告复用该快照，避免重复驱动调用。
- 不引入 NVML，因此不报告温度、功耗、实时频率或 GPU 实时占用率。
- 峰值 DRAM 带宽继续按 `memoryClockRate × 2 × memoryBusWidth / 8` 计算；有效带宽和带宽利用率分别保持既有的逻辑 I/O 与派生指标口径。

## 报告结构

每次调用 `test_softmax_kernel` 都生成一个报告块，顺序固定如下：

1. `Test Case`：内核名、场景名与结果状态。
2. `GPU Configuration`：设备名、计算能力、SM 数、全局显存、每 block 共享内存、显存时钟、总线位宽和峰值 DRAM 带宽。
3. `Launch Configuration`：矩阵形状、grid、block、动态共享内存；主机 API 路径明确显示 `Host API`，不伪造 CUDA kernel 启动参数。
4. `Results`：`CPU & GPU Results Match`、`CPU Time`、`GPU Time`、`Speedup`、`Effective Bandwidth`、`Peak DRAM Bandwidth`、`Bandwidth Utilization`。
5. `Validation`：最大相对误差、容差、被跳过元素、采样口径和失败定位。

空矩阵/空行、非法参数和正常测量均使用同一报告外框；没有数据量时延时、带宽、加速比显示 `N/A`，不以零值冒充测量值。

## 验收标准

- 默认 Softmax 运行的每个结果块都恰有一个 `GPU Configuration` 标题，且其前无未归属的设备信息。
- 每个非空报告块都含截图的七项英文指标；空形状以 `N/A` 明确标识不可测项。
- 输出保持原有的正确性判据、采样策略和成功/失败退出码。
- Release 构建通过，运行默认场景后所有项目 PASS；报告结构检查中 GPU 标题数与测试报告数相等。

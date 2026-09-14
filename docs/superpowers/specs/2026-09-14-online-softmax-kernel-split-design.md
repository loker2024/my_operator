# Online Softmax 内核拆分设计

## 目标

将 `online_softmax.cu/.cuh` 中的六个 CUDA 内核拆为各自同名的实现单元，使每个版本的
声明、启动约束和实现能够独立阅读与演进。

## 文件布局

- `include/online_softmax_common.cuh`：只放各版本共用的设备端在线二元组合并、warp/block 归约，及
  `kRegTile` 常量。
- `include/online_softmax_v0.cuh` 至 `include/online_softmax_v4.cuh`、同名 `.cu`：每对文件仅声明并实现其对应版本。
  `online_softmax_v3_false` 按现有命名单独成对拆分。
- `operators/softmax/src/main.cu`：显式包含六个版本头文件，不再依赖聚合头。

不保留聚合头 `online_softmax.cuh`，也不保留旧的聚合实现 `online_softmax.cu`。

## 依赖与行为

每个 `.cu` 包含本版本的 `.cuh`，并在需要块内在线归约时包含
`online_softmax_common.cuh`。内核名称、参数、数值路径、行映射和启动约束保持不变；本次不
改算法、不新增版本、不调整性能结论。

## 构建与验证

算子 CMake 使用 `src/*.cu` 的 `GLOB CONFIGURE_DEPENDS`，因此新实现文件由重新 configure
自动纳入。`main.cu`、`test.cu`、`test.cuh` 在实施时已存在但属于既有修改；本次仅修改
`main.cu` 的 online softmax 头文件引用，并运行现有 Softmax 回归。

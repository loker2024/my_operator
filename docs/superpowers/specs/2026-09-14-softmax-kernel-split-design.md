# Softmax 内核拆分设计

## 目标

将普通 Softmax 的 `softmax_v0` 至 `softmax_v5` 拆为各自同名 `.cuh/.cu`，不保留
`softmax.cuh/.cu` 聚合实现。

## 文件边界

- `include/softmax_common.cuh`：两级 warp shuffle 的 max/sum 块归约。
- `include/softmax_v0.cuh` 至 `include/softmax_v5.cuh`、同名 `.cu`：仅声明和实现对应 CUDA kernel。
- `include/softmax_reference.cuh`、`softmax_reference.cu`：`SoftmaxKernel`、`SoftmaxHostKernel` 类型及 CPU 参考。
- `include/softmax_cudnn.cuh`、`softmax_cudnn.cu`：可选 cuDNN 主机 API 对照。
- `main.cu` / `test.cuh`：显式包含所需版本或支持接口。

## 行为与验证

内核名称、参数、启动约束、动态共享内存需求、数值路径与 cuDNN 张量映射保持不变。先以
目标文件布局断言验证拆分前失败、拆分后通过；随后使用 WSL CUDA 的独立构建目录重新
configure、编译并运行现有 Softmax 默认回归。

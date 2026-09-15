# MEMORY.md — 长期记忆（my_operator）

## 开发环境硬件规格（2026-09-15 实测，cudaDeviceGetAttribute）

- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，compute capability 8.9（sm_89），驱动 595.71，CUDA 12.9。
- 共享内存：每 block 默认上限 48 KB（49152 B）；动态 opt-in 上限 99 KB（101376 B，需 `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`）；每 SM 总量 100 KB（102400 B）。
- SM 数 24；每 block 最大线程 1024；每 SM 寄存器 65536。
- L2 32 MB；全局内存 8187 MB。
- maxGridSize = (2147483647, 65535, 65535)；maxThreadsDim = (1024, 1024, 64)。

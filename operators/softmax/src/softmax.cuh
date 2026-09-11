#pragma once
// ============================================================================
// softmax.cuh —— Softmax 算子接口声明（行主序 fp32 矩阵，逐行数值稳定 softmax）
//   实现见同目录 softmax.cu；可复用测试驱动见 test.cuh/test.cu；执行入口见
//   main.cu。数学公式、版本推导与实测结论见 operators/softmax/README.md。
//
// 输出契约：整矩阵算好并写满 output[0, rows*cols)。softmax_cpu 为正确性基线，
//   各 GPU 版本共用 SoftmaxKernel 签名，使 test_softmax_kernel 能以函数指针统一
//   驱动（启动配置随版本由 main.cu 按 RowMap 给出）。
//
// GPU 内核版本演进（行映射与启动约束见各版本声明）：
//   v0 每线程处理一行、行内串行三遍（正确性基线）；
//   v1 每行一个 block、块内共享内存折半树形归约；
//   v2 行遍历同 v1，块内归约改两级 warp shuffle；
//   v3 行遍历与归约同 v2，行内按列宽分派 float4 向量化；
//   v4 同 v2/v3 框架，动态共享内存按行宽缓存整行 x、全局读降为 1 遍；
//   v5 同 v4 框架，改“全局读 2 遍 + float4”，动态共享内存只缓存整行 exp。
// ============================================================================

#include <cuda_runtime.h>  // __global__、cudaError_t 等 CUDA 基本定义

// ---------------------------------------------------------------------------
// CPU 参考实现（主机端正确性基线）
// ---------------------------------------------------------------------------
// 逐行同公式，内部用 double 求最大值 / 指数 / 行和（参考值不自带 fp32 舍入
// 误差），返回前转回 float 便于与 GPU 的 fp32 结果同类型比较。
void softmax_cpu(const float* input, float* output, int M, int N);

// ---------------------------------------------------------------------------
// GPU 内核。公共输出契约见文件头；行宽 N 均可任意（含 N == 0 的空行：遍历循环
// 0 次、不读不写）。各版本的行映射 / 启动配置不同，见各自声明。
// ---------------------------------------------------------------------------

// v0 每线程处理一行（正确性基线）：
//   * row = blockIdx.x * blockDim.x + threadIdx.x，行内由该线程串行三遍遍历
//     （求行最大 → Σexp → 归一化写回）；每元素读行 3 次、warp 内各线程读不同
//     行 → 访存不合并；
//   * grid = ceil(M / blockDim.x)（M == 0 时也须 >= 1，由 row >= M 越界空转）；
//   * 无共享内存 / 同步（动态共享内存 = 0）。
__global__ void softmax_v0(const float* input, float* output, const int M, const int N);

// v1 每行一个 block，行内列维由 blockDim.x 个线程以 stride = blockDim.x 协同
//   遍历三次：行最大与行和各经一次共享内存折半树形归约，写回时第三次读行重算
//   exp（不再经共享内存）：
//   * row = blockIdx.x，grid = M（M == 0 时也须 >= 1，由 row >= M 越界空转）；
//   * warp 内各线程同轮访问相邻列 → 全局读合并；N 不足 blockDim.x 时多余线程以
//     -inf / 0（归约单位元）空转，N 超过 blockDim.x 时多轮 stride；
//   * 动态共享内存 = blockDim.x * sizeof(float)（只装归约中间量，与行宽无关）；
//   * blockDim.x 应为 2 的幂（默认 256），保证折半归约各轮均匀配对。
__global__ void softmax_v1(const float* input, float* output, const int M, const int N);

// v2 每行一个 block，行内遍历与归约语义同 v1（三次 stride 扫行），仅把共享内存
//   折半树形归约换成两级 warp shuffle（helper 与实现见 softmax.cu）：
//   * row = blockIdx.x，grid = M（M == 0 时也须 >= 1，由 row >= M 越界空转）；
//   * 读合并 / 空转线程 / 多轮 stride 等语义同 v1；
//   * 额外要求 blockDim.x 为 32 的倍数（默认 256）且 <= 1024 —— shuffle 需整
//     warp 收敛，且各 warp 归约值要能装进中转的 warp_results[32]；
//   * 无动态共享内存（内部仅静态 __shared__ 做跨 warp 中转）：smem_bytes = 0。
__global__ void softmax_v2(const float* input, float* output, const int M, const int N);

// v3 行遍历与归约同 v2（每行一个 block、两级 warp shuffle、无动态共享内存），
//   行内访问按列宽分派以支持任意列宽：
//   * N % 4 == 0：行首必然 16 B 对齐（row*N 为 4 的倍数），主循环每轮 stride 取
//     1 个 float4（4 列）向量化 —— 读/写指令数为标量的 1/4；无标量尾部；
//   * 否则：整行回退 v2 式标量三遍 —— 非 4 倍列宽时第 row>=1 行的行首 16 B 不
//     对齐，float4 重解释是未定义行为（正确性不受影响，仅无向量化收益）；
//   * 启动约束同 v2：row = blockIdx.x、grid = M、blockDim.x 为 32 的倍数（默认
//     256）且 <= 1024；smem_bytes = 0。
__global__ void softmax_v3(const float* input, float* output, const int M, const int N);

// v4 每行一个 block，行内遍历改“整行缓存一遍读”（行映射 / 两级 warp shuffle
//   归约 / 按列宽分派同 v2/v3 框架）：
//   * 动态共享内存 = N * sizeof(float)，按行宽缓存整行 x：① 读全局求行最大的
//     同时把该行写进缓存（N % 4 == 0 时以 float4 槽 16 B 整写，否则标量槽写）；
//     ② 从缓存读 x 重算 exp、原地覆盖为 exp 值并累加行和；③ 从缓存读 exp 归一
//     化写回 —— 全局流量 = 读 1 遍 + 写 1 遍；
//   * 启动约束同 v2/v3：row = blockIdx.x、grid = M、blockDim.x 为 32 的倍数
//     （默认 256）且 <= 1024；额外要求 N * sizeof(float) <= 每 block 动态共享
//     内存上限（默认 48 KiB 内免 opt-in，如 N = 4096 需 16 KiB；更大行宽须以
//     cudaFuncSetAttribute 提额）。N == 0 时空转，smem_bytes = 0 即可。
__global__ void softmax_v4(const float* input, float* output, const int M, const int N);

// v5 每行一个 block，行内遍历改“全局读 2 遍 + float4”（动态共享内存缓存整行，
//   但缓存内容为 exp 而非 v4 的 x）：
//   * ① 全局 float4 读 1 遍求行最大（读后即弃）；② 再全局 float4 读 1 遍算 exp、
//     16 B 整写进动态共享内存并累加行和；③ 从 smem 读 exp 乘 inv_sum 后 float4
//     整写回 y —— 每元素全局读 2 遍 + 写 1 遍、exp 只算 1 次；
//   * 动态共享内存 = N * sizeof(float)（只装整行 exp），启动约束与上限同 v4；
//   * 列宽为 4 的倍数时 ①②③ 全走 float4，否则整行回退标量；N == 0 时空转。
__global__ void softmax_v5(const float* input, float* output, const int M, const int N);

// ---------------------------------------------------------------------------
// softmax 内核统一签名（仅输出约定一致；启动配置随内核版本由 main.cu 给出）
// ---------------------------------------------------------------------------
using SoftmaxKernel = void (*)(const float* input, float* output, int M, int N);

// ---------------------------------------------------------------------------
// 可选：厂商库（cuDNN）对照参考
// ---------------------------------------------------------------------------
// 参考实现是主机 API（内部自行启动计算），无法用 SoftmaxKernel 表示，故单列一个
// 签名：参数与 SoftmaxKernel 同序，语义为「返回后 output[0, M*N) 已写好」，供
// test.cu 的 host_kernel 通道驱动。实现只在 -DSOFTMAX_WITH_CUDNN=ON 时编译（见
// CMakeLists.txt）；类型别名恒可见，使测试驱动的该参数在未启用时也成立。
using SoftmaxHostKernel = void (*)(const float* input, float* output, int M, int N);

#ifdef SOFTMAX_WITH_CUDNN
// softmax_cudnn —— 用 cuDNN 的 cudnnSoftmaxForward 算逐行 softmax：
//   * 算法取 CUDNN_SOFTMAX_ACCURATE（先减行最大再算，即本仓库的 max-shift），
//     模式取 CUDNN_SOFTMAX_MODE_INSTANCE；
//   * 张量映射 [n=M, c=1, h=1, w=N] + nStride=N 是易错点：必须配 MODE_INSTANCE
//     （对每个 n 在 C·H·W = N 上归一）；误用 MODE_CHANNEL 时 C=1 使每个元素
//     自成一「通道」、归一化退化为恒等（张量描述符设置见 softmax.cu 实现）；
//   * M <= 0 或 N <= 0 直接返回（空矩阵 / 空行无元素可算）；
//   * 句柄与张量描述符在首次调用时创建、之后复用，调用方不必管理生命周期；
//     非线程安全（仅按单线程测试驱动使用）；与调用方同用默认流，故与
//     cudaMemcpy / CUDA event 计时天然有序。
void softmax_cudnn(const float* input, float* output, int M, int N);
#endif

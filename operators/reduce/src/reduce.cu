// ============================================================================
// reduce.cu —— reduce.cuh 声明的算子实现（CPU 参考 + GPU 内核 v0…v4、v6、v7）。
// 接口契约、版本差异与启动约束见 reduce.cuh；详细推导与实测结论见
// operators/reduce/README.md 与 notes/reduce.md。此处只保留实现侧的必要说明。
// reduce_v5 为模板内核，因实例化点需可见其定义，实现整体内联于 reduce.cuh。
// ============================================================================

#include "reduce.cuh"

// ---------------------------------------------------------------------------
// reduce_cpu —— 主机端参考实现
// ---------------------------------------------------------------------------
// 内部用 double 累加：fp32 顺序累加的舍入误差随项数增长（长序列可达 1e-4 量级，
// 见 docs/benchmark-methodology.md），double 可避免参考值自身引入额外误差；
// 返回前转回 float，便于与 GPU 的 fp32 结果同类型比较。
float reduce_cpu(const float* input, int n) {
	double sum = 0.0;
	for (int i = 0; i < n; ++i) {
		sum += static_cast<double>(input[i]);
	}
	return static_cast<float>(sum);
}

// ---------------------------------------------------------------------------
// reduce_v0 —— 交错寻址树形归约（正确性基线）
// ---------------------------------------------------------------------------
// 覆盖口径与启动约束（grid / blockDim.x / 共享内存）见 reduce.cuh 的公共说明
// 与 v0 声明。实现流程见函数体行内注释。
__global__ void reduce_v0(const float* input, float* output, int n) {
	extern __shared__ float smem[];

	const int tid = threadIdx.x;
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;

	smem[tid] = (gid < n) ? input[gid] : 0.0f;
	__syncthreads();  // 槽位全部就绪后才能开始归约

	for (size_t step = 1; step < blockDim.x; step *= 2) {
		if (tid % (2 * step) == 0) {
			smem[tid] += smem[tid + step];
		}
		__syncthreads();  // 下一轮读本轮刚写入的局部和，防止读到旧值
	}

	if (tid == 0) {
		output[blockIdx.x] = smem[0];
	}
}

// ---------------------------------------------------------------------------
// reduce_v1 —— 连续寻址树形归约
// ---------------------------------------------------------------------------
// 与 v0 差异（活跃线程由“交错”改为“连续前缀”）与启动约束见 reduce.cuh 的
// v1 声明；实现流程见函数体行内注释。
__global__ void reduce_v1(const float* input, float* output, int n) {
	extern __shared__ float smem[];

	const int tid = threadIdx.x;
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;

	smem[tid] = (gid < n) ? input[gid] : 0.0f;
	__syncthreads();

	for (size_t step = 1; step < blockDim.x; step *= 2) {
		const int index = 2 * static_cast<int>(step) * tid;
		if (index < blockDim.x) {
			smem[index] += smem[index + static_cast<int>(step)];
		}
		__syncthreads();
	}

	if (tid == 0) {
		output[blockIdx.x] = smem[0];
	}
}

// ---------------------------------------------------------------------------
// reduce_v2 —— 折半步长树形归约
// ---------------------------------------------------------------------------
// 与 v0/v1 差异（步长方向、bank 冲突）与启动约束见 reduce.cuh 的 v2 声明；
// 实现流程见函数体行内注释。
__global__ void reduce_v2(const float* input, float* output, int n) {
	extern __shared__ float smem[];

	const int tid = threadIdx.x;
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;

	smem[tid] = (gid < n) ? input[gid] : 0.0f;
	__syncthreads();

	for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (static_cast<size_t>(tid) < stride) {
			smem[tid] += smem[tid + static_cast<int>(stride)];
		}
		__syncthreads();
	}

	if (tid == 0) {
		output[blockIdx.x] = smem[0];
	}
}

// ---------------------------------------------------------------------------
// reduce_v3 —— 每线程 2 元素（标量加载）
// ---------------------------------------------------------------------------
// 覆盖口径（每 block 覆盖 2*blockDim.x 个元素、grid 计算）与启动约束见
// reduce.cuh 的 v3 声明。实现：线程 tid 以 gid = blockIdx.x * (2*blockDim.x) + tid
// 为“段内前半”下标，寄存器预加和 gid 与 gid + blockDim.x 两个元素（越界跳过，
// 等价补 0），再把预加和存进 smem，复用 v2 的折半步长归约。两次加载在 warp 内
// 各自连续且互不依赖，可提升内存级并行。
__global__ void reduce_v3(const float* input, float* output, int n) {
	extern __shared__ float smem[];

	const int tid = threadIdx.x;
	const int gid = blockIdx.x * (2 * blockDim.x) + threadIdx.x;

	float val = 0.0f;
	if (gid < n) val += input[gid];
	if (gid + blockDim.x < n) val += input[gid + blockDim.x];
	smem[tid] = val;  // smem 存“每线程 2 元素预加和”而非原始元素
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (static_cast<unsigned int>(tid) < s) {
			smem[tid] += smem[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		output[blockIdx.x] = smem[0];
	}
}

// ---------------------------------------------------------------------------
// warpReduce —— 单 warp 归约（smem 前端 32 个部分和 → smem[0]）
// ---------------------------------------------------------------------------
// 同一 warp 内指令按 SIMT 同步推进，展开写可省去 __syncthreads；volatile 强制
// 每次读写真实落内存，保证各线程读到的是其他线程刚写入的值（编译器不会把中间
// 结果缓存在寄存器里而错过别人的写入）。
__device__ void warpReduce(volatile float* smem, int tid) {
	smem[tid] += smem[tid + 32];
	smem[tid] += smem[tid + 16];
	smem[tid] += smem[tid + 8];
	smem[tid] += smem[tid + 4];
	smem[tid] += smem[tid + 2];
	smem[tid] += smem[tid + 1];
}

// ---------------------------------------------------------------------------
// reduce_v4 —— 每线程 2 元素 + 末 warp 展开归约
// ---------------------------------------------------------------------------
// 加载与覆盖口径同 v3、启动约束见 reduce.cuh 的 v4 声明。实现：折半归约到
// stride = 32 即停，剩余 5 轮改由 warp 0 调 warpReduce 展开完成（见其上方
// 注释），省去这些轮次的 __syncthreads。
__global__ void reduce_v4(const float* input, float* output, int n) {
	extern __shared__ float smem[];

	const int tid = threadIdx.x;
	const int gid = blockIdx.x * (2 * blockDim.x) + threadIdx.x;

	float val = 0.0f;
	if (gid < n) val += input[gid];
	if (gid + blockDim.x < n) val += input[gid + blockDim.x];
	smem[tid] = val;
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (static_cast<unsigned int>(tid) < s) {
			smem[tid] += smem[tid + s];
		}
		__syncthreads();
	}

	if (static_cast<unsigned int>(tid) < 32) {
		warpReduce(smem, tid);
	}

	if (tid == 0) {
		output[blockIdx.x] = smem[0];
	}
}

// reduce_v5 —— 模板常量化版本：与 v4 算法一致，仅把 block 尺寸以模板参数
// BLOCK_SIZE 常量化、让归约步骤在编译期整体展开。因各实例化点需可见其定义
// （-rdc=false 下跨翻译单元引用 __global__ 模板特化已被 nvcc 弃用），实现整体
// 内联于 reduce.cuh（设计说明见其 v5 声明注释）。

// ---------------------------------------------------------------------------
// warpReduceSum —— 单 warp shuffle 归约（每 lane 1 个局部和 → 全 lane 同值）
// ---------------------------------------------------------------------------
// 5 轮 __shfl_down_sync（offset 16 → 1）在寄存器间归约，不碰共享内存、无需同步；
// mask 0xffffffff 要求完整 warp 收敛调用，调用点不可只留单条 lane。
__device__ float warpReduceSum(float val) {
	for (int offset = 16; offset > 0; offset >>= 1) {
		val += __shfl_down_sync(0xffffffff, val, offset);
	}
	return val;
}

// ---------------------------------------------------------------------------
// reduce_v6 —— 每线程 2 元素 + 两级 warp shuffle 归约
// ---------------------------------------------------------------------------
// 加载与覆盖口径同 v3/v4/v5（每 block 覆盖 2*blockDim.x 个元素），块内约束
// （blockDim.x 为 2 的幂且 32 ~ 1024）见 reduce.cuh 的 v6 声明。实现：每 warp
// 先经 warpReduceSum 归为 1 个部分和，再由 warp 0 归约 numWarps 个部分和，
// 见函数体行内注释。
__global__ void reduce_v6(const float* input, float* output, int n) {
	__shared__ float warp_results[32];  // 各 warp 的部分和

	const int tid = threadIdx.x;
	const int gid = blockIdx.x * (2 * blockDim.x) + threadIdx.x;
	const int lane = tid % 32;
	const int wid = tid / 32;

	float val = 0.0f;
	if (gid < n) val += input[gid];
	if (gid + blockDim.x < n) val += input[gid + blockDim.x];

	val = warpReduceSum(val);  // ① warp 内归约
	if (lane == 0) {
		warp_results[wid] = val;
	}

	__syncthreads();

	// ② 每 lane 取 1 个 warp 的部分和再归约一次（lane >= numWarps 视为 0）。
	// shuffle 需整 warp 参与，故仍让整个 warp 0 执行。
	const int numWarps = blockDim.x / 32;
	if (wid == 0) {
		val = (lane < numWarps) ? warp_results[lane] : 0.0f;
		val = warpReduceSum(val);
	}

	if (tid == 0) {
		output[blockIdx.x] = val;
	}
}

// ---------------------------------------------------------------------------
// reduce_v7 —— float4 向量化加载 + grid-stride 扫描 + 两级 warp shuffle
// ---------------------------------------------------------------------------
// 覆盖口径（grid-stride、对齐要求、启动网格）与启动约束见 reduce.cuh 的 v7
// 声明；实现结构（float4 主循环 → 尾部标量 → 块内归约）见函数体行内注释。
__global__ void reduce_v7(const float* input, float* output, int n) {
	const int tid = threadIdx.x;
	const int lane = tid % 32;
	const int wid = tid / 32;

	// float4 主循环：input 整体视作 n/4 个 float4（要求 16 字节对齐）。
	const float4* input4 = reinterpret_cast<const float4*>(input);
	const int n4 = n / 4;

	float val = 0.0f;
	for (int idx = blockIdx.x * blockDim.x + tid; idx < n4; idx += gridDim.x * blockDim.x) {
		const float4 data = input4[idx];
		val += data.x + data.y + data.z + data.w;
	}

	// 尾部标量循环：覆盖 [n4*4, n)，grid-stride 步长与主循环一致。
	const int tail_start = n4 * 4;
	for (int idx = tail_start + blockIdx.x * blockDim.x + tid; idx < n;
	     idx += gridDim.x * blockDim.x) {
		val += input[idx];
	}

	// ① warp 内归约
	val = warpReduceSum(val);

	__shared__ float warp_results[32];  // 各 warp 的部分和
	if (lane == 0) {
		warp_results[wid] = val;
	}

	__syncthreads();

	// ② warp 0 归约 numWarps 个部分和（lane >= numWarps 视为 0）。
	const int numWarps = blockDim.x / 32;
	if (wid == 0) {
		val = (lane < numWarps) ? warp_results[lane] : 0.0f;
		val = warpReduceSum(val);
	}

	if (tid == 0) {
		output[blockIdx.x] = val;
	}
}

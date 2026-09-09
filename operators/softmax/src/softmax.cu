// softmax.cu —— softmax.cuh 声明的实现：CPU 参考 + softmax_v0/v1/v2/v3/v4/v5 内核。
// 接口契约 / 启动约束 / 版本差异见 softmax.cuh，推导与实测见 README.md。


#include <cmath>    // expf / fmaxf / INFINITY / std::exp
#include <cstddef>  // std::size_t

#include "softmax.cuh"

// softmax_cpu —— 主机端参考（测试的正确性基线）
void softmax_cpu(const float* input, float* output, int M, int N) {
	// 逐行 softmax：求行最大 m → Σexp(x-m) → 归一化。全程 double：参考值不自带
	// fp32 舍入误差（GPU 侧还要经受 1e-5 判据），返回前转回 float。
	for (int row = 0; row < M; ++row) {
		const float* x = input + static_cast<std::size_t>(row) * N;
		float* y = output + static_cast<std::size_t>(row) * N;
		if (N <= 0) continue;  // 空行：无元素可写

		double m = -INFINITY;
		for (int c = 0; c < N; ++c) {
			m = (static_cast<double>(x[c]) > m) ? static_cast<double>(x[c]) : m;
		}

		double sum = 0.0;
		for (int c = 0; c < N; ++c) {
			sum += std::exp(static_cast<double>(x[c]) - m);
		}

		const double inv_sum = 1.0 / sum;
		for (int c = 0; c < N; ++c) {
			y[c] = static_cast<float>(std::exp(static_cast<double>(x[c]) - m) * inv_sum);
		}
	}
}

// softmax_v0 —— 每线程处理一行，行内串行三遍（正确性基线）
__global__ void softmax_v0(const float* input, float* output, const int M, const int N) {
	const int row = blockIdx.x * blockDim.x + threadIdx.x;  // 线程铺满行号
	if (row >= M) return;  // 空矩阵 / 超配 grid：越界行空转

	const float* x = input + row * N;
	float* y = output + row * N;

	// ① 行最大 m（max-shift：exp 参数 <= 0、行和 >= 1，数值稳定）
	float max_val = -INFINITY;
	for (size_t i = 0; i < N; ++i) {
		max_val = fmaxf(max_val, x[i]);
	}

	// ② Σexp(x - m)
	float sum = 0.0f;
	for (size_t i = 0; i < N; ++i) {
		sum += expf(x[i] - max_val);
	}

	// ③ 归一化写回 y = exp(x - m) / 行和
	const float inv_sum = 1.0f / sum;
	for (size_t i = 0; i < N; ++i) {
		y[i] = expf(x[i] - max_val) * inv_sum;
	}
}

// softmax_v1 —— 每行一个 block，行内协作 + 共享内存树形归约
// （启动约束与行映射见 softmax.cuh 的 v1 声明）
__global__ void softmax_v1(const float* input, float* output, const int M, const int N) {
	extern __shared__ float smem[];  // 动态共享内存：blockDim.x * sizeof(float)

	const int row = blockIdx.x;  // 每 block 处理一行
	const int tid = threadIdx.x;
	if (row >= M) return;  // 空矩阵 / 超配 grid：空转

	const float* x = input + row * N;
	float* y = output + row * N;

	// ① 各线程 stride 扫行求局部最大 → 折半 fmaxf 树形归约出行最大
	float local_max = -INFINITY;
	for (int i = tid; i < N; i += blockDim.x) {
		local_max = fmaxf(local_max, x[i]);
	}
	smem[tid] = local_max;
	__syncthreads();  // 槽位全部就绪后才能开始归约
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (tid < stride) {
			smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
		}
		__syncthreads();  // 下一轮读本轮刚写入的局部最大
	}
	const float row_max = smem[0];
	__syncthreads();  // 全部线程读走 m 后，smem 才能被 ② 复用

	// ② 同样的 stride 扫行累加局部 Σexp → 树形加法归约出行和
	float local_sum = 0.0f;
	for (int i = tid; i < N; i += blockDim.x) {
		local_sum += expf(x[i] - row_max);
	}
	smem[tid] = local_sum;
	__syncthreads();
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (tid < stride) {
			smem[tid] += smem[tid + stride];
		}
		__syncthreads();
	}
	const float row_sum = smem[0];

	// ③ 第三次读行重算 exp 并归一化写回（smem 只装规约中间量，与行宽无关）
	const float inv_sum = 1.0f / row_sum;
	for (int i = tid; i < N; i += blockDim.x) {
		y[i] = expf(x[i] - row_max) * inv_sum;
	}
}

// ---------------------------------------------------------------------------
// warpReduceMax / warpReduceSum —— 单 warp shuffle 归约（每 lane 1 个局部值 →
// 归约值收敛到 lane 0）
// ---------------------------------------------------------------------------
// 5 轮 __shfl_down_sync（offset 16 → 1）在寄存器间两两归约，不碰共享内存、无
// 同步开销；mask 0xffffffff 要求完整 warp 收敛调用（调用点不可让 warp 内部分
// 线程提前退出，否则行为未定义）。
__device__ float warpReduceMax(float val) {
	for (int offset = 16; offset > 0; offset >>= 1) {
		val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
	}
	return val;
}

__device__ float warpReduceSum(float val) {
	for (int offset = 16; offset > 0; offset >>= 1) {
		val += __shfl_down_sync(0xffffffff, val, offset);
	}
	return val;
}

// ---------------------------------------------------------------------------
// blockReduceMaxShuffle / blockReduceSumShuffle —— 块内归约（两级 warp shuffle，
// 结果经共享内存广播回全体线程）
// ---------------------------------------------------------------------------
// ① warpReduce{Max,Sum} 先把每 warp 归为 1 个值，lane 0 写入 warp_results[wid]；
// ② __syncthreads 后 warp 0 再归约 num_warps 个 warp 值（lane >= num_warps 以归约
//    单位元 -INFINITY / 0 空转，不影响结果），块归约值收敛到 tid 0；
// ③ 经 block_result + __syncthreads 广播：softmax 的 ②③ 遍需要每个线程都拿到
//    行最大 / 行和，不能只由 tid 0 持有（区别于 reduce 只需 tid 0 写输出）。
// 约束：blockDim.x 为 32 的倍数（默认 256）且 num_warps <= 32（即 blockDim.x
// <= 1024），warp 值才装得进 warp_results[32]。
__device__ float blockReduceMaxShuffle(float val) {
	__shared__ float warp_results[32];  // 各 warp 的归约值（每 warp 由 lane 0 写入）
	__shared__ float block_result;      // 广播用块归约值

	const int lane = threadIdx.x % 32;
	const int wid = threadIdx.x / 32;

	val = warpReduceMax(val);  // ① warp 内归约：lane 0 持该 warp 归约值
	if (lane == 0) warp_results[wid] = val;
	__syncthreads();  // 全部 warp 值就绪后才能被 warp 0 读取

	const int num_warps = blockDim.x / 32;
	val = (lane < num_warps) ? warp_results[lane] : -INFINITY;
	if (wid == 0) {
		val = warpReduceMax(val);  // ② warp 0 归约 num_warps 个值 → 块最大值
	}
	if (threadIdx.x == 0) block_result = val;  // tid 0（warp 0 lane 0）持有块结果
	__syncthreads();  // ③ 广播：全体读到块归约值后才允许继续（写回前读行）

	return block_result;
}

__device__ float blockReduceSumShuffle(float val) {
	__shared__ float warp_results[32];
	__shared__ float block_result;

	const int lane = threadIdx.x % 32;
	const int wid = threadIdx.x / 32;

	val = warpReduceSum(val);
	if (lane == 0) warp_results[wid] = val;
	__syncthreads();

	const int num_warps = blockDim.x / 32;
	val = (lane < num_warps) ? warp_results[lane] : 0.0f;
	if (wid == 0) {
		val = warpReduceSum(val);
	}
	if (threadIdx.x == 0) block_result = val;
	__syncthreads();

	return block_result;
}

// ---------------------------------------------------------------------------
// softmax_v2 —— 每行一个 block，块内两级 warp shuffle 归约
// ---------------------------------------------------------------------------
// 行映射 / 启动约束见 softmax.cuh 的 v2 声明；与 v1 的差异仅在块内归约：共享
// 内存折半树换成两级 warp shuffle（结构见上方 blockReduce* helper 注释）。
__global__ void softmax_v2(const float* input, float* output, const int M, const int N) {
	const int row = blockIdx.x;  // 每 block 处理一行
	if (row >= M) return;        // 空矩阵 / 超配 grid：越界行空转
	const int tid = threadIdx.x;

	const float* x = input + row * N;
	float* y = output + row * N;

	// ① 行最大 m：stride 扫行局部最大 → 块内两级 warp shuffle 归约（max-shift
	//    使 exp 参数 <= 0、行和 >= 1，数值稳定）
	float local_max = -INFINITY;
	for (int i = tid; i < N; i += blockDim.x) {
		local_max = fmaxf(local_max, x[i]);
	}
	const float row_max = blockReduceMaxShuffle(local_max);

	// ② Σexp(x - m)：同样的 stride 扫行分段累加 → 块内归约
	float local_sum = 0.0f;
	for (int i = tid; i < N; i += blockDim.x) {
		local_sum += expf(x[i] - row_max);
	}
	const float row_sum = blockReduceSumShuffle(local_sum);

	// ③ 归一化写回 y = exp(x - m) / 行和（第三次读行重算 exp）
	const float inv_sum = 1.0f / row_sum;
	for (int i = tid; i < N; i += blockDim.x) {
		y[i] = expf(x[i] - row_max) * inv_sum;
	}
}

// ---------------------------------------------------------------------------
// softmax_v3 —— 每行一个 block，v2 行遍历 + float4 向量化（列宽为 4 的倍数时）
// ---------------------------------------------------------------------------
// 行映射 / 归约 / 启动约束同 v2，见 softmax.cuh 的 v3 声明（含“非 4 倍列宽为何
// 整行回退标量”的说明）。实现要点：N % 4 == 0 时行首 16 B 对齐，主循环每轮取
// 1 个 float4（4 列，读/写指令数为标量 1/4）；否则走 else 分支的 v2 式标量三遍。
__global__ void softmax_v3(const float* input, float* output, const int M, const int N) {
	const int row = blockIdx.x;  // 每 block 处理一行
	if (row >= M) return;        // 空矩阵 / 超配 grid：越界行空转
	const int tid = threadIdx.x;

	const float* x = input + row * N;
	float* y = output + row * N;

	if (N % 4 == 0) {
		// 列宽为 4 的倍数：行首 16 B 对齐，float4 主循环（每轮 stride 处理 4 列；
		// 空行 N == 0 时 n4 == 0，各遍循环 0 次、不读不写）
		const int n4 = N / 4;
		const float4* x4 = reinterpret_cast<const float4*>(x);
		float4* y4 = reinterpret_cast<float4*>(y);

		// ① 行最大 m：float4 各分量逐一 fmaxf（max-shift 使 exp 参数 <= 0）
		float local_max = -INFINITY;
		for (int i = tid; i < n4; i += blockDim.x) {
			local_max = fmaxf(local_max, x4[i].x);
			local_max = fmaxf(local_max, x4[i].y);
			local_max = fmaxf(local_max, x4[i].z);
			local_max = fmaxf(local_max, x4[i].w);
		}
		const float row_max = blockReduceMaxShuffle(local_max);

		// ② Σexp(x - m)：同一 float4 的 4 分量一次读出后逐分量累加
		float local_sum = 0.0f;
		for (int i = tid; i < n4; i += blockDim.x) {
			const float4 v = x4[i];
			local_sum += expf(v.x - row_max) + expf(v.y - row_max) + expf(v.z - row_max) +
			             expf(v.w - row_max);
		}
		const float row_sum = blockReduceSumShuffle(local_sum);

		// ③ 归一化写回：float4 整写（N % 4 == 0，无标量尾部）
		const float inv_sum = 1.0f / row_sum;
		for (int i = tid; i < n4; i += blockDim.x) {
			const float4 v = x4[i];
			y4[i] = make_float4(expf(v.x - row_max) * inv_sum, expf(v.y - row_max) * inv_sum,
			                    expf(v.z - row_max) * inv_sum, expf(v.w - row_max) * inv_sum);
		}
	} else {
		// 列宽非 4 的倍数：行首不保证 16 B 对齐，整行回退标量三遍（语义同 v2）
		float local_max = -INFINITY;
		for (int i = tid; i < N; i += blockDim.x) {
			local_max = fmaxf(local_max, x[i]);
		}
		const float row_max = blockReduceMaxShuffle(local_max);

		float local_sum = 0.0f;
		for (int i = tid; i < N; i += blockDim.x) {
			local_sum += expf(x[i] - row_max);
		}
		const float row_sum = blockReduceSumShuffle(local_sum);

		const float inv_sum = 1.0f / row_sum;
		for (int i = tid; i < N; i += blockDim.x) {
			y[i] = expf(x[i] - row_max) * inv_sum;
		}
	}
}

__global__ void softmax_v4(const float* input, float* output, const int M, const int N) {
	// 动态共享内存按 float4 槽使用（16 B 对齐），字节数仍为 N * sizeof(float)。
	// 整行只从全局读 1 遍：① 读 x 的同时把整行缓存进 smem（并求行最大）；② 从
	// smem 读缓存 x 算 exp、原地覆盖为 exp 值并累加行和；③ 从 smem 读 exp 归一化
	// 写回 —— 全局流量 = 读 1 遍 + 写 1 遍（理论下限）。整除判断只分派一次：
	// N % 4 == 0 时 ①② 都走 float4（全局读 + smem 槽 16 B 整写），否则整行标量。
	// 缓冲区两段生命周期（x → exp）之间各隔一次 blockReduce —— 其尾部 __syncthreads
	// 保证写者全部就绪后才被下一遍读取，见 helper 注释。
	extern __shared__ float4 smem4[];
	const int row = blockIdx.x;
	if (row >= M) return;
	const int tid = threadIdx.x;

	const float* x = input + row * N;
	float* y = output + row * N;
	float* smem = reinterpret_cast<float*>(smem4);  // 标量槽视图（[col] 布局）
	float sum = 0.0f;

	if (N % 4 == 0) {
		// 列宽为 4 的倍数：行首 16 B 对齐（空行 N == 0 时 n4 == 0，各遍循环 0 次）
		const int n4 = N / 4;
		const float4* x4 = reinterpret_cast<const float4*>(x);

		// ① 行最大 + 整行缓存：读 1 个 float4 即 16 B 整写进 smem4[i]，分量逐一
		// fmaxf（max-shift 使 exp 参数 <= 0）；此后 x 不再从全局读
		float local_max = -INFINITY;
		for (int i = tid; i < n4; i += blockDim.x) {
			const float4 v = x4[i];
			smem4[i] = v;
			local_max = fmaxf(local_max, v.x);
			local_max = fmaxf(local_max, v.y);
			local_max = fmaxf(local_max, v.z);
			local_max = fmaxf(local_max, v.w);
		}
		const float row_max = blockReduceMaxShuffle(local_max);  // 尾部同步：x 缓存全就绪

		// ② 从 smem 读缓存 x 算 exp：4 个 exp 一次算完、16 B 整写回同一槽（覆盖 x），
		// 就地累加 local_sum（warp 内相邻 lane 写相邻 16 B 槽 → 无 bank conflict）
		float local_sum = 0.0f;
		for (int i = tid; i < n4; i += blockDim.x) {
			const float4 v = smem4[i];
			const float4 e = make_float4(expf(v.x - row_max), expf(v.y - row_max),
			                             expf(v.z - row_max), expf(v.w - row_max));
			smem4[i] = e;
			local_sum += (e.x + e.y) + (e.z + e.w);
		}
		sum = blockReduceSumShuffle(local_sum);  // 尾部同步：exp 缓存全就绪
	} else {
		// 列宽非 4 的倍数：行首不保证 16 B 对齐，①② 整行回退标量（语义同 v2）
		float local_max = -INFINITY;
		for (int i = tid; i < N; i += blockDim.x) {
			const float xv = x[i];
			smem[i] = xv;
			local_max = fmaxf(local_max, xv);
		}
		const float row_max = blockReduceMaxShuffle(local_max);

		float local_sum = 0.0f;
		for (int i = tid; i < N; i += blockDim.x) {
			const float xv = smem[i];
			const float ev = expf(xv - row_max);
			smem[i] = ev;
			local_sum += ev;
		}
		sum = blockReduceSumShuffle(local_sum);
	}

	// ③ 归一化写回：经 float 视图按 [col] 读 exp 缓存（两种模式的写入布局一致）
	const float inv_sum = 1.0f / sum;
	for (int i = tid; i < N; i += blockDim.x) {
		y[i] = smem[i] * inv_sum;
	}
}

// ---------------------------------------------------------------------------
// softmax_v5 —— 每行一个 block：全局读 2 遍 + float4，exp 只算 1 次
// ---------------------------------------------------------------------------
// v4 前身思路的原样接入：行内不再把 x 缓存进 smem，而是改全局读 2 遍 —— ① 读全局
// 求行最大（读后即弃）；② 再读全局算 exp、把 exp 值写进动态共享内存（整行）并累加
// 行和；③ 从 smem 读 exp 乘 inv_sum 归一化写回。三种资源流量对比：
//   * v3：全局读 3 遍 + exp 算 2 次 + 写 1 遍（②③ 各算一次 exp）；
//   * v4：全局读 1 遍 + exp 算 1 次 + 写 1 遍，但 x 要 smem 写 1 次 + 读 1 次往返
//     （x、exp 两轮缓存，smem 流量是 v5 的两倍）；
//   * v5：全局读 2 遍 + exp 算 1 次 + 写 1 遍，smem 只存 exp（写 1 读 1）—— 以
//     “多读 1 遍全局”换掉 v4 的 “x 经 smem 往返”，全局读带宽富余时更划算（v4 的
//     smem 往返被怀疑是主要开销，见 README 结论记录）。
// 启动约束同 v4，见 softmax.cuh 的 v5 声明。实现要点与 v4 一致：N % 4 == 0 时行首
// 16 B 对齐，①/②/③ 全走 float4（② 的 4 个 exp 一次算完并 16 B 整写进 smem4[i]，
// ③ 乘 inv_sum 后 float4 整写回 y —— 读/写指令数均为标量 1/4）；否则整行回退标量。
__global__ void softmax_v5(const float* input, float* output, const int M, const int N) {
	extern __shared__ float4 smem4[];  // 动态共享内存：N * sizeof(float)，只装 exp 值
	const int row = blockIdx.x;
	if (row >= M) return;  // 空矩阵 / 超配 grid：越界行空转
	const int tid = threadIdx.x;

	const float* x = input + row * N;
	float* y = output + row * N;
	float* smem = reinterpret_cast<float*>(smem4);  // 标量槽视图（[col] 布局）

	if (N % 4 == 0) {
		// 列宽为 4 的倍数：行首 16 B 对齐（空行 N == 0 时 n4 == 0，各遍循环 0 次）
		const int n4 = N / 4;
		const float4* x4 = reinterpret_cast<const float4*>(x);
		float4* y4 = reinterpret_cast<float4*>(y);

		// ① 行最大 m：float4 全局读 1 遍、分量逐一 fmaxf，读后即弃（不落 smem）
		float local_max = -INFINITY;
		for (int i = tid; i < n4; i += blockDim.x) {
			const float4 v = x4[i];
			local_max = fmaxf(local_max, v.x);
			local_max = fmaxf(local_max, v.y);
			local_max = fmaxf(local_max, v.z);
			local_max = fmaxf(local_max, v.w);
		}
		const float row_max = blockReduceMaxShuffle(local_max);

		// ② Σexp：再全局读 1 遍，4 个 exp 一次算完、16 B 整写进 smem4[i]（存 exp
		// 供 ③ 用），同时就地累加 local_sum —— 每元素至此只算 1 次 exp
		float local_sum = 0.0f;
		for (int i = tid; i < n4; i += blockDim.x) {
			const float4 v = x4[i];
			const float4 e = make_float4(expf(v.x - row_max), expf(v.y - row_max),
			                             expf(v.z - row_max), expf(v.w - row_max));
			smem4[i] = e;
			local_sum += (e.x + e.y) + (e.z + e.w);
		}
		const float row_sum = blockReduceSumShuffle(local_sum);  // 尾部同步：exp 缓存全就绪

		// ③ 归一化写回：从 smem 读 exp、乘 inv_sum 后 float4 整写回 y（无标量尾部）
		const float inv_sum = 1.0f / row_sum;
		for (int i = tid; i < n4; i += blockDim.x) {
			const float4 e = smem4[i];
			y4[i] = make_float4(e.x * inv_sum, e.y * inv_sum, e.z * inv_sum,
			                    e.w * inv_sum);
		}
	} else {
		// 列宽非 4 的倍数：行首不保证 16 B 对齐，①/②/③ 整行回退标量（语义同 v2）
		float local_max = -INFINITY;
		for (int i = tid; i < N; i += blockDim.x) {
			local_max = fmaxf(local_max, x[i]);
		}
		const float row_max = blockReduceMaxShuffle(local_max);

		float local_sum = 0.0f;
		for (int i = tid; i < N; i += blockDim.x) {
			const float ev = expf(x[i] - row_max);
			smem[i] = ev;  // 存 exp 值供 ③ 归一化用（缓存内容为 exp，而非 v4 的 x）
			local_sum += ev;
		}
		const float row_sum = blockReduceSumShuffle(local_sum);

		const float inv_sum = 1.0f / row_sum;
		for (int i = tid; i < N; i += blockDim.x) {
			y[i] = smem[i] * inv_sum;
		}
	}
}

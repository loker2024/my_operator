// online_softmax.cu —— online_softmax.cuh 的实现：online_softmax_v0/v1/v2（单趟在线归约）。
// 接口契约 / 启动约束见 online_softmax.cuh，推导与实测见 README.md。

#include <cmath>  // expf / fmaxf / INFINITY

#include "online_softmax.cuh"

// online_softmax_v0 —— 每线程处理一行，行内单趟 online 归约
__global__ void online_softmax_v0(const float* input, float* output, const int M, const int N) {
	const int row = blockIdx.x * blockDim.x + threadIdx.x;  // 线程铺满行号
	if (row >= M) return;  // 空矩阵 / 超配 grid：越界行空转

	const float* x = input + row * N;
	float* y = output + row * N;

	// ① 一趟读行在线归约：m = 已扫子集最大值，d = Σexp(x_k - m)（以 m 为基准）
	float m = -INFINITY;
	float d = 0.0f;
	for (int i = 0; i < N; ++i) {
		const float xi = x[i];
		const float m_new = fmaxf(m, xi);
		d = d * expf(m - m_new) + expf(xi - m_new);  // 旧分母缩放回新基准 + 新元素贡献
		m = m_new;
	}

	// ② 归一化写回（不缓存整行，故 exp 每元素算 2 次：在线更新 1 次 + 此处 1 次）。
	//    空行 N == 0：inv_d == inf，但循环 0 次、不写任何元素
	const float inv_d = 1.0f / d;
	for (int i = 0; i < N; ++i) {
		y[i] = expf(x[i] - m) * inv_d;
	}
}

// ---------------------------------------------------------------------------
// mergeOnline —— 二元组 (m, d) 的结合运算（online 归约的核心，单元素插入与
// 子集合并都走它）
// ---------------------------------------------------------------------------
// 把右侧 (m2, d2) 并入左侧：m_new = max(m, m2)，两侧分母各按「自身基准 → 新基准」
// 之差缩放后相加。单元素插入即 d2 = 1（exp(x - x) = 1）。
// 空集（m == -INFINITY、d == 0）跳过该侧缩放：否则 (-inf) - (-inf) = NaN 会经
// expf 污染整个分母 —— 未分到元素的线程 / warp 必然以空集参与合并（N 不整除或
// 小于 blockDim.x，见 v1 声明）。
__device__ void mergeOnline(float& m, float& d, const float m2, const float d2) {
	const float m_new = fmaxf(m, m2);
	const float scale = (m == -INFINITY) ? 0.0f : expf(m - m_new);
	const float scale2 = (m2 == -INFINITY) ? 0.0f : expf(m2 - m_new);
	d = d * scale + d2 * scale2;
	m = m_new;
}

// ---------------------------------------------------------------------------
// warpReduceOnline —— 单 warp 内 shuffle 归约 (m, d)（结果收敛到 lane 0）
// ---------------------------------------------------------------------------
// 5 轮 __shfl_down_sync（offset 16 → 1）在寄存器间两两 mergeOnline，不碰共享内存；
// mask 0xffffffff 要求完整 warp 收敛调用，故调用点不可让 warp 内部分线程提前退出。
__device__ void warpReduceOnline(float& m, float& d) {
	for (int offset = 16; offset > 0; offset >>= 1) {
		const float m2 = __shfl_down_sync(0xffffffff, m, offset);
		const float d2 = __shfl_down_sync(0xffffffff, d, offset);
		mergeOnline(m, d, m2, d2);
	}
}

// ---------------------------------------------------------------------------
// blockReduceOnline —— 块内两级 warp shuffle 归约 (m, d)，结果广播回全体线程
// ---------------------------------------------------------------------------
// 结构同 softmax.cu 的 blockReduce*Shuffle：① warpReduceOnline 把每 warp 归为 1
// 个二元组，lane 0 写入 warp_m/warp_d[wid]；② __syncthreads 后 warp 0 再合并
// num_warps 个 warp 值（lane >= num_warps 以空集单位元 -inf / 0 参与）；③ 结果
// 经 block_m/block_d + __syncthreads 广播 —— 写回遍需要每个线程都拿到整行的
// (m, d)，不能只由 tid 0 持有（区别于 reduce 只需 tid 0 写输出）。
// 约束：blockDim.x 为 32 的倍数且 <= 1024，warp 值才装得进 warp_m[32] / warp_d[32]。
__device__ void blockReduceOnline(float& m, float& d) {
	__shared__ float warp_m[32];  // 各 warp 的归约二元组（每 warp 由 lane 0 写入）
	__shared__ float warp_d[32];
	__shared__ float block_m;  // 广播用块归约二元组
	__shared__ float block_d;

	const int lane = threadIdx.x % 32;
	const int wid = threadIdx.x / 32;

	warpReduceOnline(m, d);  // ① warp 内归约：lane 0 持有该 warp 结果
	if (lane == 0) {
		warp_m[wid] = m;
		warp_d[wid] = d;
	}
	__syncthreads();  // 全部 warp 值就绪后才能被 warp 0 读取

	const int num_warps = blockDim.x / 32;
	if (wid == 0) {  // ② warp 0 合并 num_warps 个 warp 值
		m = (lane < num_warps) ? warp_m[lane] : -INFINITY;
		d = (lane < num_warps) ? warp_d[lane] : 0.0f;
		warpReduceOnline(m, d);

		if (lane == 0) {  // ③ tid 0（warp 0 lane 0）持有块结果，广播给全 block
			block_m = m;
			block_d = d;
		}
	}
	__syncthreads();  // 广播：全体读到块归约值后才允许进入写回遍

	m = block_m;
	d = block_d;
}

// ---------------------------------------------------------------------------
// online_softmax_v1 —— 每行一个 block，单趟在线归约 + 两级 warp shuffle 合并
// ---------------------------------------------------------------------------
// 行映射 / 启动约束见 online_softmax.cuh 的 v1 声明；与 v0 的差异仅在把「一线程
// 独占一行」换成「块内协作」：读合并、并行度由行数提升到 rows × blockDim.x。
__global__ void online_softmax_v1(const float* input, float* output, const int M, const int N) {
	const int row = blockIdx.x;  // 每 block 处理一行
	if (row >= M) return;        // 空矩阵 / 超配 grid：越界行空转
	const int tid = threadIdx.x;

	const float* x = input + row * N;
	float* y = output + row * N;

	// ① 一趟 stride 扫行：每线程在线归约自己的列子集 → 块内两级 shuffle 合并出
	//    整行 (m, d)（求 m 与求 Σexp 合并为一趟全局读）
	float m = -INFINITY;
	float d = 0.0f;
	for (int i = tid; i < N; i += blockDim.x) {
		mergeOnline(m, d, x[i], 1.0f);  // 插入单元素：该元素自身基准下的分母为 1
	}
	blockReduceOnline(m, d);

	// ② 第二趟读行重算 exp(x - m) / d 写回（不缓存整行，exp 每元素算 2 次）。
	//    空行 N == 0：d == 0 → inv_d == inf，但循环 0 次、不写任何元素
	const float inv_d = 1.0f / d;
	for (int i = tid; i < N; i += blockDim.x) {
		y[i] = expf(x[i] - m) * inv_d;
	}
}

// ---------------------------------------------------------------------------
// online_softmax_v2 —— v1 + float4 向量化（列宽为 4 的倍数时）
// ---------------------------------------------------------------------------
// 行映射 / 启动约束同 v1，见 online_softmax.cuh 的 v2 声明（含“非 4 倍列宽为何整行
// 回退标量”的说明）。与 v1 的差异仅在行内访问：N % 4 == 0 时行首 16 B 对齐，单趟
// 在线归约与写回都以 float4 为单位 stride 扫行（每轮 4 列、读/写指令数为标量 1/4，
// 4 分量逐一 mergeOnline），否则整行走 v1 式标量两遍。
__global__ void online_softmax_v2(const float* input, float* output, const int M, const int N) {
	const int row = blockIdx.x;  // 每 block 处理一行
	if (row >= M) return;        // 空矩阵 / 超配 grid：越界行空转
	const int tid = threadIdx.x;

	const float* x = input + row * N;
	float* y = output + row * N;

	float m = -INFINITY;
	float d = 0.0f;

	if (N % 4 == 0) {
		// 列宽为 4 的倍数：行首 16 B 对齐，float4 主循环（每轮 stride 处理 4 列；
		// 空行 N == 0 时 n4 == 0，两遍循环 0 次、不读不写）
		const int n4 = N / 4;
		const float4* x4 = reinterpret_cast<const float4*>(x);
		float4* y4 = reinterpret_cast<float4*>(y);

		// ① 一趟 float4 stride 扫行：4 分量逐一插入在线归约 → 块内 shuffle 合并
		for (int i = tid; i < n4; i += blockDim.x) {
			const float4 v = x4[i];
			mergeOnline(m, d, v.x, 1.0f);
			mergeOnline(m, d, v.y, 1.0f);
			mergeOnline(m, d, v.z, 1.0f);
			mergeOnline(m, d, v.w, 1.0f);
		}
		blockReduceOnline(m, d);

		// ② 第二趟 float4 读行重算 exp(x - m) / d 并整写回（无标量尾部）
		const float inv_d = 1.0f / d;
		for (int i = tid; i < n4; i += blockDim.x) {
			const float4 v = x4[i];
			y4[i] = make_float4(expf(v.x - m) * inv_d, expf(v.y - m) * inv_d, expf(v.z - m) * inv_d,
			                    expf(v.w - m) * inv_d);
		}
	} else {
		// 列宽非 4 的倍数：行首不保证 16 B 对齐，整行回退标量两遍（语义同 v1）
		// ① 一趟 stride 扫行：每线程在线归约自己的列子集 → 块内两级 shuffle 合并出
		//    整行 (m, d)（求 m 与求 Σexp 合并为一趟全局读）
		for (int i = tid; i < N; i += blockDim.x) {
			mergeOnline(m, d, x[i], 1.0f);  // 插入单元素：该元素自身基准下的分母为 1
		}
		blockReduceOnline(m, d);

		// ② 第二趟读行重算 exp(x - m) / d 写回（不缓存整行，exp 每元素算 2 次）。
		//    空行 N == 0：d == 0 → inv_d == inf，但循环 0 次、不写任何元素
		const float inv_d = 1.0f / d;
		for (int i = tid; i < N; i += blockDim.x) {
			y[i] = expf(x[i] - m) * inv_d;
		}
	}
}

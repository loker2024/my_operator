#pragma once
// online_softmax_common.cuh —— online softmax 各版本共享的设备端在线归约工具。
// 只供 online_softmax_v1/v2/v3/v3_false/v4 的实现单元包含；不导出 host API。

#include <cuda_runtime.h>

#include <cmath>

// 每线程寄存器分片容量。仅 online_softmax_v3/v3_false 使用；block=256 时覆盖 N<=4096。
constexpr int kOnlineSoftmaxRegTile = 16;

// 将右侧子集 (m2,d2) 合入左侧 (m,d)。空集以 (-inf,0) 表示，缩放前显式跳过空集，
// 避免 (-inf)-(-inf) 经 expf 传播 NaN。
static __device__ __forceinline__ void mergeOnline(float& m, float& d, const float m2,
                                                   const float d2) {
	const float m_new = fmaxf(m, m2);
	const float scale = (m == -INFINITY) ? 0.0f : expf(m - m_new);
	const float scale2 = (m2 == -INFINITY) ? 0.0f : expf(m2 - m_new);
	d = d * scale + d2 * scale2;
	m = m_new;
}

// 在完整收敛的一个 warp 内归约在线二元组，结果存于 lane 0。
static __device__ __forceinline__ void warpReduceOnline(float& m, float& d) {
	for (int offset = 16; offset > 0; offset >>= 1) {
		const float m2 = __shfl_down_sync(0xffffffff, m, offset);
		const float d2 = __shfl_down_sync(0xffffffff, d, offset);
		mergeOnline(m, d, m2, d2);
	}
}

// 两级 warp shuffle 块归约，并将最终 (m,d) 广播给全体线程。
// 启动约束：blockDim.x 必须为 32 的倍数且不超过 1024。
static __device__ __forceinline__ void blockReduceOnline(float& m, float& d) {
	__shared__ float warp_m[32];
	__shared__ float warp_d[32];
	__shared__ float block_m;
	__shared__ float block_d;

	const int lane = threadIdx.x % 32;
	const int wid = threadIdx.x / 32;
	warpReduceOnline(m, d);
	if (lane == 0) {
		warp_m[wid] = m;
		warp_d[wid] = d;
	}
	__syncthreads();

	const int num_warps = blockDim.x / 32;
	if (wid == 0) {
		m = (lane < num_warps) ? warp_m[lane] : -INFINITY;
		d = (lane < num_warps) ? warp_d[lane] : 0.0f;
		warpReduceOnline(m, d);
		if (lane == 0) {
			block_m = m;
			block_d = d;
		}
	}
	__syncthreads();
	m = block_m;
	d = block_d;
}

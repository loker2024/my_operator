#pragma once
// softmax_common.cuh —— softmax_v2 至 v5 共享的两级 warp shuffle 块归约。

#include <cuda_runtime.h>

#include <cmath>

static __device__ __forceinline__ float warpReduceMax(float value) {
	for (int offset = 16; offset > 0; offset >>= 1) {
		value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
	}
	return value;
}

static __device__ __forceinline__ float warpReduceSum(float value) {
	for (int offset = 16; offset > 0; offset >>= 1) {
		value += __shfl_down_sync(0xffffffff, value, offset);
	}
	return value;
}

// blockDim.x 必须为 32 的倍数且 <= 1024；结果经静态共享内存广播给全 block。
static __device__ __forceinline__ float blockReduceMaxShuffle(float value) {
	__shared__ float warp_results[32];
	__shared__ float block_result;
	const int lane = threadIdx.x % 32;
	const int wid = threadIdx.x / 32;
	value = warpReduceMax(value);
	if (lane == 0) warp_results[wid] = value;
	__syncthreads();
	const int num_warps = blockDim.x / 32;
	value = (lane < num_warps) ? warp_results[lane] : -INFINITY;
	if (wid == 0) value = warpReduceMax(value);
	if (threadIdx.x == 0) block_result = value;
	__syncthreads();
	return block_result;
}

static __device__ __forceinline__ float blockReduceSumShuffle(float value) {
	__shared__ float warp_results[32];
	__shared__ float block_result;
	const int lane = threadIdx.x % 32;
	const int wid = threadIdx.x / 32;
	value = warpReduceSum(value);
	if (lane == 0) warp_results[wid] = value;
	__syncthreads();
	const int num_warps = blockDim.x / 32;
	value = (lane < num_warps) ? warp_results[lane] : 0.0f;
	if (wid == 0) value = warpReduceSum(value);
	if (threadIdx.x == 0) block_result = value;
	__syncthreads();
	return block_result;
}

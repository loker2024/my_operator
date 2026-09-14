// softmax_v1.cu —— 共享内存折半树形归约的块协作 Softmax。

#include <cmath>

#include "include/softmax_v1.cuh"

__global__ void softmax_v1(const float* input, float* output, const int M, const int N) {
	extern __shared__ float smem[];
	const int row = blockIdx.x;
	const int tid = threadIdx.x;
	if (row >= M) return;
	const float* x = input + row * N;
	float* y = output + row * N;
	float local_max = -INFINITY;
	for (int i = tid; i < N; i += blockDim.x) local_max = fmaxf(local_max, x[i]);
	smem[tid] = local_max;
	__syncthreads();
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (tid < stride) smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
		__syncthreads();
	}
	const float row_max = smem[0];
	__syncthreads();
	float local_sum = 0.0f;
	for (int i = tid; i < N; i += blockDim.x) local_sum += expf(x[i] - row_max);
	smem[tid] = local_sum;
	__syncthreads();
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (tid < stride) smem[tid] += smem[tid + stride];
		__syncthreads();
	}
	const float inv_sum = 1.0f / smem[0];
	for (int i = tid; i < N; i += blockDim.x) y[i] = expf(x[i] - row_max) * inv_sum;
}

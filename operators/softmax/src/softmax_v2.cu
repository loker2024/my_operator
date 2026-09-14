// softmax_v2.cu —— 两级 warp shuffle 归约的块协作 Softmax。

#include <cmath>

#include "include/softmax_common.cuh"
#include "include/softmax_v2.cuh"

__global__ void softmax_v2(const float* input, float* output, const int M, const int N) {
	const int row = blockIdx.x;
	if (row >= M) return;
	const int tid = threadIdx.x;
	const float* x = input + row * N;
	float* y = output + row * N;
	float local_max = -INFINITY;
	for (int i = tid; i < N; i += blockDim.x) local_max = fmaxf(local_max, x[i]);
	const float row_max = blockReduceMaxShuffle(local_max);
	float local_sum = 0.0f;
	for (int i = tid; i < N; i += blockDim.x) local_sum += expf(x[i] - row_max);
	const float inv_sum = 1.0f / blockReduceSumShuffle(local_sum);
	for (int i = tid; i < N; i += blockDim.x) y[i] = expf(x[i] - row_max) * inv_sum;
}

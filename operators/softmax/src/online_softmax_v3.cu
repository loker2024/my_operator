// online_softmax_v3.cu —— 静态下标寄存器分片缓存实现。

#include <cmath>

#include "include/online_softmax_common.cuh"
#include "include/online_softmax_v3.cuh"

__global__ void online_softmax_v3(const float* input, float* output, const int M, const int N) {
	const int row = blockIdx.x;
	if (row >= M) return;
	const int tid = threadIdx.x;
	const float* x = input + row * N;
	float* y = output + row * N;
	float local_m = -INFINITY;
	float local_d = 0.0f;

	if ((N + blockDim.x - 1) / blockDim.x <= kOnlineSoftmaxRegTile) {
		float reg[kOnlineSoftmaxRegTile];
#pragma unroll
		for (int k = 0; k < kOnlineSoftmaxRegTile; ++k) {
			const int i = tid + k * blockDim.x;
			if (i < N) {
				const float xi = x[i];
				reg[k] = xi;
				mergeOnline(local_m, local_d, xi, 1.0f);
			} else {
				reg[k] = 0.0f;
			}
		}
		blockReduceOnline(local_m, local_d);
		const float row_max = local_m;
		const float inv_sum = 1.0f / local_d;
#pragma unroll
		for (int k = 0; k < kOnlineSoftmaxRegTile; ++k) {
			const int i = tid + k * blockDim.x;
			if (i < N) y[i] = expf(reg[k] - row_max) * inv_sum;
		}
	} else {
		for (int i = tid; i < N; i += blockDim.x) {
			mergeOnline(local_m, local_d, x[i], 1.0f);
		}
		blockReduceOnline(local_m, local_d);
		const float row_max = local_m;
		const float inv_sum = 1.0f / local_d;
		for (int i = tid; i < N; i += blockDim.x) {
			y[i] = expf(x[i] - row_max) * inv_sum;
		}
	}
}

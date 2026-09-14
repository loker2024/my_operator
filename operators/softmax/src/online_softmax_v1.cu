// online_softmax_v1.cu —— 块内在线归约 + 两级 warp shuffle 实现。

#include <cmath>

#include "include/online_softmax_common.cuh"
#include "include/online_softmax_v1.cuh"

__global__ void online_softmax_v1(const float* input, float* output, const int M, const int N) {
	const int row = blockIdx.x;
	if (row >= M) return;
	const int tid = threadIdx.x;
	const float* x = input + row * N;
	float* y = output + row * N;

	float m = -INFINITY;
	float d = 0.0f;
	for (int i = tid; i < N; i += blockDim.x) {
		mergeOnline(m, d, x[i], 1.0f);
	}
	blockReduceOnline(m, d);

	const float inv_d = 1.0f / d;
	for (int i = tid; i < N; i += blockDim.x) {
		y[i] = expf(x[i] - m) * inv_d;
	}
}

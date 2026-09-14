// online_softmax_v0.cu —— 每线程一行的单趟在线归约实现。

#include <cmath>

#include "include/online_softmax_v0.cuh"

__global__ void online_softmax_v0(const float* input, float* output, const int M, const int N) {
	const int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= M) return;

	const float* x = input + row * N;
	float* y = output + row * N;
	float m = -INFINITY;
	float d = 0.0f;
	for (int i = 0; i < N; ++i) {
		const float xi = x[i];
		const float m_new = fmaxf(m, xi);
		d = d * expf(m - m_new) + expf(xi - m_new);
		m = m_new;
	}

	const float inv_d = 1.0f / d;
	for (int i = 0; i < N; ++i) {
		y[i] = expf(x[i] - m) * inv_d;
	}
}

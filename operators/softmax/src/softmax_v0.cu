// softmax_v0.cu —— 每线程独占一行的三遍数值稳定 Softmax。

#include <cmath>
#include <cstddef>

#include "include/softmax_v0.cuh"

__global__ void softmax_v0(const float* input, float* output, const int M, const int N) {
	const int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= M) return;
	const float* x = input + row * N;
	float* y = output + row * N;
	float maximum = -INFINITY;
	for (std::size_t col = 0; col < static_cast<std::size_t>(N); ++col) {
		maximum = fmaxf(maximum, x[col]);
	}
	float sum = 0.0f;
	for (std::size_t col = 0; col < static_cast<std::size_t>(N); ++col) {
		sum += expf(x[col] - maximum);
	}
	const float inv_sum = 1.0f / sum;
	for (std::size_t col = 0; col < static_cast<std::size_t>(N); ++col) {
		y[col] = expf(x[col] - maximum) * inv_sum;
	}
}

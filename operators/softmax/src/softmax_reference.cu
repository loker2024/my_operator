// softmax_reference.cu —— 正确性测试使用的逐行 double 精度 Softmax 参考。

#include <cmath>
#include <cstddef>

#include "include/softmax_reference.cuh"

void softmax_cpu(const float* input, float* output, int M, int N) {
	for (int row = 0; row < M; ++row) {
		const float* x = input + static_cast<std::size_t>(row) * N;
		float* y = output + static_cast<std::size_t>(row) * N;
		if (N <= 0) continue;
		double maximum = -INFINITY;
		for (int col = 0; col < N; ++col) {
			maximum = static_cast<double>(x[col]) > maximum ? static_cast<double>(x[col]) : maximum;
		}
		double sum = 0.0;
		for (int col = 0; col < N; ++col) sum += std::exp(static_cast<double>(x[col]) - maximum);
		const double inv_sum = 1.0 / sum;
		for (int col = 0; col < N; ++col) {
			y[col] = static_cast<float>(std::exp(static_cast<double>(x[col]) - maximum) * inv_sum);
		}
	}
}

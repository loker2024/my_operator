// softmax_v3.cu —— v2 的 float4 向量化与标量回退实现。

#include <cmath>

#include "include/softmax_common.cuh"
#include "include/softmax_v3.cuh"

__global__ void softmax_v3(const float* input, float* output, const int M, const int N) {
	const int row = blockIdx.x;
	if (row >= M) return;
	const int tid = threadIdx.x;
	const float* x = input + row * N;
	float* y = output + row * N;
	if (N % 4 == 0) {
		const int n4 = N / 4;
		const float4* x4 = reinterpret_cast<const float4*>(x);
		float4* y4 = reinterpret_cast<float4*>(y);
		float local_max = -INFINITY;
		for (int i = tid; i < n4; i += blockDim.x) {
			const float4 value = x4[i];
			local_max = fmaxf(local_max, value.x);
			local_max = fmaxf(local_max, value.y);
			local_max = fmaxf(local_max, value.z);
			local_max = fmaxf(local_max, value.w);
		}
		const float row_max = blockReduceMaxShuffle(local_max);
		float local_sum = 0.0f;
		for (int i = tid; i < n4; i += blockDim.x) {
			const float4 value = x4[i];
			local_sum += expf(value.x - row_max) + expf(value.y - row_max) +
			             expf(value.z - row_max) + expf(value.w - row_max);
		}
		const float inv_sum = 1.0f / blockReduceSumShuffle(local_sum);
		for (int i = tid; i < n4; i += blockDim.x) {
			const float4 value = x4[i];
			y4[i] =
			    make_float4(expf(value.x - row_max) * inv_sum, expf(value.y - row_max) * inv_sum,
			                expf(value.z - row_max) * inv_sum, expf(value.w - row_max) * inv_sum);
		}
	} else {
		float local_max = -INFINITY;
		for (int i = tid; i < N; i += blockDim.x) local_max = fmaxf(local_max, x[i]);
		const float row_max = blockReduceMaxShuffle(local_max);
		float local_sum = 0.0f;
		for (int i = tid; i < N; i += blockDim.x) local_sum += expf(x[i] - row_max);
		const float inv_sum = 1.0f / blockReduceSumShuffle(local_sum);
		for (int i = tid; i < N; i += blockDim.x) y[i] = expf(x[i] - row_max) * inv_sum;
	}
}

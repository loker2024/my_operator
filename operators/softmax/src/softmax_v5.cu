// softmax_v5.cu —— 两遍全局读、动态共享内存仅缓存 exp 的 Softmax。

#include <cmath>

#include "include/softmax_common.cuh"
#include "include/softmax_v5.cuh"

__global__ void softmax_v5(const float* input, float* output, const int M, const int N) {
	extern __shared__ float4 smem4[];
	const int row = blockIdx.x;
	if (row >= M) return;
	const int tid = threadIdx.x;
	const float* x = input + row * N;
	float* y = output + row * N;
	float* smem = reinterpret_cast<float*>(smem4);
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
			const float4 exponent = make_float4(expf(value.x - row_max), expf(value.y - row_max),
			                                    expf(value.z - row_max), expf(value.w - row_max));
			smem4[i] = exponent;
			local_sum += (exponent.x + exponent.y) + (exponent.z + exponent.w);
		}
		const float inv_sum = 1.0f / blockReduceSumShuffle(local_sum);
		for (int i = tid; i < n4; i += blockDim.x) {
			const float4 exponent = smem4[i];
			y4[i] = make_float4(exponent.x * inv_sum, exponent.y * inv_sum, exponent.z * inv_sum,
			                    exponent.w * inv_sum);
		}
	} else {
		float local_max = -INFINITY;
		for (int i = tid; i < N; i += blockDim.x) local_max = fmaxf(local_max, x[i]);
		const float row_max = blockReduceMaxShuffle(local_max);
		float local_sum = 0.0f;
		for (int i = tid; i < N; i += blockDim.x) {
			const float exponent = expf(x[i] - row_max);
			smem[i] = exponent;
			local_sum += exponent;
		}
		const float inv_sum = 1.0f / blockReduceSumShuffle(local_sum);
		for (int i = tid; i < N; i += blockDim.x) y[i] = smem[i] * inv_sum;
	}
}

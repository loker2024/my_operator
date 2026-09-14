// online_softmax_v2.cu —— 在线归约的 float4/标量分派实现。

#include <cmath>

#include "include/online_softmax_common.cuh"
#include "include/online_softmax_v2.cuh"

__global__ void online_softmax_v2(const float* input, float* output, const int M, const int N) {
	const int row = blockIdx.x;
	if (row >= M) return;
	const int tid = threadIdx.x;
	const float* x = input + row * N;
	float* y = output + row * N;
	float m = -INFINITY;
	float d = 0.0f;

	if (N % 4 == 0) {
		const int n4 = N / 4;
		const float4* x4 = reinterpret_cast<const float4*>(x);
		float4* y4 = reinterpret_cast<float4*>(y);
		for (int i = tid; i < n4; i += blockDim.x) {
			const float4 v = x4[i];
			mergeOnline(m, d, v.x, 1.0f);
			mergeOnline(m, d, v.y, 1.0f);
			mergeOnline(m, d, v.z, 1.0f);
			mergeOnline(m, d, v.w, 1.0f);
		}
		blockReduceOnline(m, d);
		const float inv_d = 1.0f / d;
		for (int i = tid; i < n4; i += blockDim.x) {
			const float4 v = x4[i];
			y4[i] = make_float4(expf(v.x - m) * inv_d, expf(v.y - m) * inv_d, expf(v.z - m) * inv_d,
			                    expf(v.w - m) * inv_d);
		}
	} else {
		for (int i = tid; i < N; i += blockDim.x) {
			mergeOnline(m, d, x[i], 1.0f);
		}
		blockReduceOnline(m, d);
		const float inv_d = 1.0f / d;
		for (int i = tid; i < N; i += blockDim.x) {
			y[i] = expf(x[i] - m) * inv_d;
		}
	}
}

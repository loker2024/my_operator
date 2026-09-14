// online_softmax_v3_false.cu —— 运行期下标缓存（刻意落入 local memory）的对照实现。

#include <cmath>

#include "include/online_softmax_common.cuh"
#include "include/online_softmax_v3_false.cuh"

__global__ void online_softmax_v3_false(const float* input, float* output, const int M,
                                        const int N) {
	const int row = blockIdx.x;
	if (row >= M) return;
	const int tid = threadIdx.x;
	const float* x = input + row * N;
	float* y = output + row * N;
	float local_m = -INFINITY;
	float local_d = 0.0f;
	float reg_cache[kOnlineSoftmaxRegTile];

	int count = 0;
	for (int i = tid; i < N; i += blockDim.x) {
		const float xi = x[i];
		reg_cache[count++] = xi;
		mergeOnline(local_m, local_d, xi, 1.0f);
	}
	blockReduceOnline(local_m, local_d);
	const float row_max = local_m;
	const float inv_sum = 1.0f / local_d;

	count = 0;
	for (int i = tid; i < N; i += blockDim.x) {
		y[i] = expf(reg_cache[count++] - row_max) * inv_sum;
	}
}

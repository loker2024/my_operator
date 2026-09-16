// ============================================================================
// sgemm_v0.cu —— SGEMM v0 内核：每线程计算一个输出元素，K 方向直接读全局内存。
// ============================================================================

#include "include/sgemm_v0.cuh"

__global__ void sgemm_v0(const float* A, const float* B, float* C, int M, int N, int K) {
	const int row = blockIdx.x * blockDim.x + threadIdx.x;  // 当前线程负责的输出行
	const int col = blockIdx.y * blockDim.y + threadIdx.y;  // 当前线程负责的输出列
	if (row >= M || col >= N) return;

	// 累加 K 方向的乘积。
	float sum = 0.0f;
	for (int k = 0; k < K; ++k) {
		sum += A[row * K + k] * B[k * N + col];
	}
	C[row * N + col] = sum;
}

// ============================================================================
// sgemm_v0.cu —— 每线程计算一个输出元素的 SGEMM 正确性基线。
// 启动约束见 include/sgemm_v0.cuh：block=(16,16)，grid 覆盖 M×N，动态共享内存为 0。
// 每次内层循环直接从全局内存读取 A/B；实现直观但没有数据复用，仅用于正确性和优化基线。
// ============================================================================

#include "include/sgemm_v0.cuh"

__global__ void sgemm_v0(const float* A, const float* B, float* C, int M, int N, int K) {
	const int row = blockIdx.x * blockDim.x + threadIdx.x;
	const int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row >= M || col >= N) return;

	float sum = 0.0f;
	for (int k = 0; k < K; ++k) {
		sum += A[row * K + k] * B[k * N + col];
	}
	C[row * N + col] = sum;
}

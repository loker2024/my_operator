// ============================================================================
// sgemm_reference.cu —— SGEMM 正确性验证使用的主机参考实现。
// 计算行主序 C(M×N) = A(M×K) × B(K×N)。参数均须非负；测试驱动先验证该契约。
// double 累加只用于提高参考精度，不作为性能基线。
// ============================================================================

#include <cstddef>

#include "include/sgemm_reference.cuh"

void sgemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
	for (int row = 0; row < M; ++row) {
		for (int col = 0; col < N; ++col) {
			double sum = 0.0;
			for (int k = 0; k < K; ++k) {
				sum += static_cast<double>(A[static_cast<std::size_t>(row) * K + k]) *
				       B[static_cast<std::size_t>(k) * N + col];
			}
			C[static_cast<std::size_t>(row) * N + col] = static_cast<float>(sum);
		}
	}
}

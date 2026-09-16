#pragma once
// sgemm_v3.cuh —— SGEMM v3 共享内存分块与寄存器行分块模板内核。

#include <cuda_runtime.h>

template <int BM, int BN, int BK, int TM>
__global__ void sgemm_v3(const float* A, const float* B, float* C, const int M, const int N,
                         const int K) {
	static_assert(BM % TM == 0, "TM must divide BM");
	constexpr int kThreads = BM * BN / TM;

	__shared__ float sharedA[BM * BK];  // A 的当前 k tile
	__shared__ float sharedB[BK * BN];  // B 的当前 k tile

	const int tid = threadIdx.x;
	const int block_row = blockIdx.y * BM;
	const int block_col = blockIdx.x * BN;
	const int thread_row_base = (tid / BN) * TM;
	const int thread_col = tid % BN;
	float thread_results[TM] = {};

	if (blockDim.x != kThreads) return;

	for (int k_base = 0; k_base < K; k_base += BK) {
		// 协作装载 A 的当前 k tile，越界元素补 0。
		for (int index = tid; index < BM * BK; index += kThreads) {
			const int row = index / BK;
			const int col = index % BK;
			const int global_row = block_row + row;
			const int global_col = k_base + col;
			sharedA[index] =
			    (global_row < M && global_col < K) ? A[global_row * K + global_col] : 0.0f;
		}

		// 协作装载 B 的当前 k tile，越界元素补 0。
		for (int index = tid; index < BK * BN; index += kThreads) {
			const int row = index / BN;
			const int col = index % BN;
			const int global_row = k_base + row;
			const int global_col = block_col + col;
			sharedB[index] =
			    (global_row < K && global_col < N) ? B[global_row * N + global_col] : 0.0f;
		}
		__syncthreads();

		// 累加当前线程负责的 TM 个输出行。
		for (int k = 0; k < BK; ++k) {
			const float b_value = sharedB[k * BN + thread_col];
#pragma unroll
			for (int row_offset = 0; row_offset < TM; ++row_offset) {
				thread_results[row_offset] +=
				    sharedA[(thread_row_base + row_offset) * BK + k] * b_value;
			}
		}
		__syncthreads();
	}

	// 将合法范围内的 TM 个输出元素写回全局内存。
#pragma unroll
	for (int row_offset = 0; row_offset < TM; ++row_offset) {
		const int global_row = block_row + thread_row_base + row_offset;
		const int global_col = block_col + thread_col;
		if (global_row < M && global_col < N) {
			C[global_row * N + global_col] = thread_results[row_offset];
		}
	}
}

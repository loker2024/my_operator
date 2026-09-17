#pragma once
// sgemm_v3_user_previous.cuh —— 保留的用户原始 SGEMM v3 实现，不由运行入口包含。

#include <cuda_runtime.h>

#include <cassert>

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_v3_user_previous(const float* A, const float* B, float* C, const int M,
                                       const int N, const int K) {
	__shared__ float sharedA[BM * BK];
	__shared__ float sharedB[BN * BK];

	const int cRow = blockIdx.y;
	const int cCol = blockIdx.x;

	const int threadRow = threadIdx.x / BN;
	const int threadCol = threadIdx.x % BN;

	A += cRow * BM * K;
	B += cCol * BN;
	C += cRow * BM * N + cCol * BN;

	assert(BM * BK == blockDim.x);
	assert(BK * BN == blockDim.x);

	const int innerRowA = threadIdx.x / BK;
	const int innerColA = threadIdx.x % BK;
	const int innerRowB = threadIdx.x / BN;
	const int innerColB = threadIdx.x % BN;

	float threadResults[TM] = {0.0};

	for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
		sharedA[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
		sharedB[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];

		__syncthreads();
		A += BK;
		B += BK * N;

		for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
			const float tmpB = sharedB[dotIdx * BN + threadCol];
			for (int resIdx = 0; resIdx < TM; ++resIdx) {
				threadResults[resIdx] += sharedA[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
			}
		}
		__syncthreads();
	}
	for (int resIdx = 0; resIdx < TM; ++resIdx) {
		C[(threadRow * TM + resIdx) * N + threadCol] = threadResults[resIdx];
	}
}

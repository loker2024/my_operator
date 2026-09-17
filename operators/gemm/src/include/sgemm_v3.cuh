#pragma once
// sgemm_v3.cuh —— SGEMM v3 的一维 block-tiling 模板内核，由运行入口实例化 TM1 与 TM8 配置。

#include <cuda_runtime.h>

#include <cassert>

// 启动契约：BM=BN=BK×TM，blockDim.x=BM×BK=BK²×TM，且线程数不超过 1024。
template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_v3(const float* A, const float* B, float* C, const int M, const int N,
                         const int K) {
	__shared__ float sharedA[BM * BK];  // A 的当前 k tile
	__shared__ float sharedB[BN * BK];  // B 的当前 k tile

	const int cRow = blockIdx.y;
	const int cCol = blockIdx.x;

	// 将线性线程索引拆分为输出 tile 内的行组与列坐标。
	const int threadRow = threadIdx.x / BN;
	const int threadCol = threadIdx.x % BN;

	// 将输入和输出指针定位到当前输出 tile 的起始位置。
	A += cRow * BM * K;
	B += cCol * BN;
	C += cRow * BM * N + cCol * BN;

	// 当前 block 的每个线程分别装载 A / B tile 的一个元素。
	assert(BM * BK == blockDim.x);
	assert(BK * BN == blockDim.x);

	const int innerRowA = threadIdx.x / BK;
	const int innerColA = threadIdx.x % BK;
	const int innerRowB = threadIdx.x / BN;
	const int innerColB = threadIdx.x % BN;

	float threadResults[TM] = {0.0};  // 当前线程负责的 TM 个输出元素

	// 逐个处理 K 方向的 tile。
	for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
		// 将当前 A / B tile 装入共享内存。
		sharedA[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
		sharedB[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];

		__syncthreads();
		// 将 A / B 指针推进到下一个 K tile。
		A += BK;
		B += BK * N;

		// 累加当前线程同一列的 TM 个输出行。
		for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
			const float tmpB = sharedB[dotIdx * BN + threadCol];
			for (int resIdx = 0; resIdx < TM; ++resIdx) {
				threadResults[resIdx] += sharedA[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
			}
		}
		__syncthreads();
	}
	// 将当前线程负责的 TM 个结果写回 C。
	for (int resIdx = 0; resIdx < TM; ++resIdx) {
		C[(threadRow * TM + resIdx) * N + threadCol] = threadResults[resIdx];
	}
}

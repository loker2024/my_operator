#pragma once
// sgemm_v4.cuh —— SGEMM v4 的二维线程分块模板内核，由运行入口实例化 tile 与每线程输出元素数。

#include <cuda_runtime.h>

#include <cassert>

// 启动契约：blockDim.x = BM×BN/(TM×TN) ≤ 1024，BM 能被 blockDim.x/BK 整除，BK 能被
// blockDim.x/BN 整除；M、N、K 分别须为 BM、BN、BK 的倍数。
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_v4(const float* A, const float* B, float* C, const int M, const int N,
                         const int K) {
	__shared__ float sharedA[BM * BK];  // A 的当前 k tile
	__shared__ float sharedB[BK * BN];  // B 的当前 k tile

	const int cRow = blockIdx.y;  // 本 block 的输出 tile 行号
	const int cCol = blockIdx.x;  // 本 block 的输出 tile 列号

	// 每线程负责输出 tile 内的 TM×TN 个元素，线程数由 tile 元素数除以每线程元素数得到。
	const int numThreadsBlocktile = BM * BN / (TM * TN);
	assert(numThreadsBlocktile == blockDim.x);

	// 将线性线程索引拆分为输出 tile 内的行组号与列组号。
	const int threadRow = threadIdx.x / (BN / TN);
	const int threadCol = threadIdx.x % (BN / TN);

	// 将 A / B / C 指针移到本 block 负责区域的左上角。
	A += cRow * BM * K;
	B += cCol * BN;
	C += cRow * BM * N + cCol * BN;

	// 装载阶段把线性线程索引拆成 tile 内的行列坐标，再按行步长遍历整个 tile。
	const int innerRowA = threadIdx.x / BK;
	const int innerColA = threadIdx.x % BK;
	const int innerRowB = threadIdx.x / BN;
	const int innerColB = threadIdx.x % BN;
	const int strideA = numThreadsBlocktile / BK;
	const int strideB = numThreadsBlocktile / BN;

	float threadResults[TM * TN] = {};  // 当前线程负责的 TM×TN 个输出元素
	float regM[TM] = {};                // 当前 k 步取出的 A 值
	float regN[TN] = {};                // 当前 k 步取出的 B 值

	// 沿 k 方向按 BK 分块推进。
	for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
		// 将 A 的当前 tile 装入共享内存，每次装载前进 strideA 行。
		for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
			sharedA[(innerRowA + loadOffset) * BK + innerColA] =
			    A[(innerRowA + loadOffset) * K + innerColA];
		}
		// 将 B 的当前 tile 装入共享内存，每次装载前进 strideB 行。
		for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
			sharedB[(innerRowB + loadOffset) * BN + innerColB] =
			    B[(innerRowB + loadOffset) * N + innerColB];
		}
		__syncthreads();

		// A / B 指针前进到下一个 k tile。
		A += BK;
		B += BK * N;

		// 每个 k 步取出 TM 个 A 值与 TN 个 B 值，累加到 TM×TN 个输出上。
		for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
			for (int i = 0; i < TM; ++i) {
				regM[i] = sharedA[(threadRow * TM + i) * BK + dotIdx];
			}
			for (int i = 0; i < TN; ++i) {
				regN[i] = sharedB[dotIdx * BN + threadCol * TN + i];
			}
			for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
				for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
					threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
				}
			}
		}
		__syncthreads();
	}

	// 将当前线程负责的 TM×TN 个结果写回 C。
	for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
		for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
			C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
			    threadResults[resIdxM * TN + resIdxN];
		}
	}
}

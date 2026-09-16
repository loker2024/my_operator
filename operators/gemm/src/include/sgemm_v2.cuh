#pragma once
// ============================================================================
// sgemm_v2.cuh —— SGEMM v2 共享内存分块模板内核。
// 计算行主序 C(M×N) = A(M×K) × B(K×N)：一个 block 覆盖一个 TILE_SIZE×TILE_SIZE 输出 tile，
// k 方向按 TILE_SIZE 分块，A / B 的当前 tile 先装入共享内存再参与乘加；模板内联定义在头文件，
// 由 main.cu 以具体 TILE_SIZE 显式实例化并注册。
// ============================================================================

// 每线程计算一个输出元素；TILE_SIZE 是输出 tile 边长，同时也是共享内存 tile 边长与 k 方向
// 分块步长，每 block 必须起满 TILE_SIZE² 个线程（线性索引按行优先拆成 tile 内行列坐标）。
template <const int TILE_SIZE>
__global__ void sgemm_v2(const float* A, const float* B, float* C, const int M, const int N,
                         const int K) {
	__shared__ float sharedA[TILE_SIZE * TILE_SIZE];  // A 的当前 k tile
	__shared__ float sharedB[TILE_SIZE * TILE_SIZE];  // B 的当前 k tile

	const int cRow = blockIdx.y;  // 本 block 的输出 tile 行号
	const int cCol = blockIdx.x;  // 本 block 的输出 tile 列号

	// 将线性线程索引拆分为 tile 内行列坐标。
	const int threadRow = threadIdx.x / TILE_SIZE;
	const int threadCol = threadIdx.x % TILE_SIZE;

	const int globalRow = cRow * TILE_SIZE + threadRow;  // 当前线程负责的输出行
	const int globalCol = cCol * TILE_SIZE + threadCol;  // 当前线程负责的输出列

	// 将 A / B / C 指针移到本 block 负责区域的左上角。
	A += cRow * TILE_SIZE * K;
	B += cCol * TILE_SIZE;
	C += cRow * TILE_SIZE * N + cCol * TILE_SIZE;

	float sum = 0.0f;

	for (int bkIdx = 0; bkIdx < K; bkIdx += TILE_SIZE) {
		// 将 A / B 的当前 k tile 装入共享内存，越界元素补 0。
		sharedA[threadRow * TILE_SIZE + threadCol] =
		    (globalRow < M && bkIdx + threadCol < K) ? A[threadRow * K + threadCol] : 0.0f;
		sharedB[threadRow * TILE_SIZE + threadCol] =
		    (bkIdx + threadRow < K && globalCol < N) ? B[threadRow * N + threadCol] : 0.0f;
		__syncthreads();

		// 用共享内存中的 tile 累加出当前线程的输出值。
		for (int k = 0; k < TILE_SIZE; ++k) {
			sum += sharedA[threadRow * TILE_SIZE + k] * sharedB[k * TILE_SIZE + threadCol];
		}
		__syncthreads();

		// A / B 指针前进到下一个 k tile。
		A += TILE_SIZE;
		B += TILE_SIZE * N;
	}

	// 仅合法行列范围内的结果写回全局内存。
	if (globalRow < M && globalCol < N) {
		C[threadRow * N + threadCol] = sum;
	}
}

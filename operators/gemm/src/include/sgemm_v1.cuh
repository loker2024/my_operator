#pragma once
// ============================================================================
// sgemm_v1.cuh —— SGEMM v1 模板内核：每线程一个输出元素，不使用共享内存，
// k 方向直接读全局内存；模板内联定义在头文件，由 main.cu 以具体 TILE_SIZE 显式实例化并注册。
// ============================================================================

// 每线程计算一个输出元素；TILE_SIZE 是输出 tile 边长，每 block 必须起满 TILE_SIZE² 个线程
// （线性索引按行优先拆成 tile 内行列坐标）。
template <const uint TILE_SIZE>
__global__ void sgemm_v1(const float* A, const float* B, float* C, const uint M, const uint N,
                         const uint K) {
	// 将线性线程索引拆分为 tile 内行列坐标。
	const int cRow = blockIdx.x * TILE_SIZE + threadIdx.x / TILE_SIZE;  // 当前线程负责的输出行
	const int cCol = blockIdx.y * TILE_SIZE + threadIdx.x % TILE_SIZE;  // 当前线程负责的输出列

	if (cRow < M && cCol < N) {
		// 累加 K 方向的乘积。
		float sum = 0.0f;
		for (int k = 0; k < K; k++) {
			sum += A[cRow * K + k] * B[k * N + cCol];
		}
		C[cRow * N + cCol] = sum;
	}
}

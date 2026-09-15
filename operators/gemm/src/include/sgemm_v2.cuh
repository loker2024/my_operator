#pragma once
// ============================================================================
// sgemm_v2.cuh —— SGEMM v2 共享内存分块模板内核（内联定义，由 main.cu 显式实例化）。
// 计算行主序 C(M×N) = A(M×K) × B(K×N)：一个 block 以 TILE_SIZE² 个线性线程覆盖一个
// TILE_SIZE×TILE_SIZE 的输出 tile；k 方向按 TILE_SIZE 步长把 A / B 的 tile 搬进共享内存，
// 每线程再从共享内存做 TILE_SIZE 次乘加 —— 全局访存量降到 v1 的 1/TILE_SIZE，tile 内
// 每个元素被 TILE_SIZE 个线程复用。
// 模板内核内联定义在头文件（-rdc=false 下跨翻译单元引用 __global__ 模板特化已被 nvcc
// 弃用），故 main.cu 直接包含本文件并以具体 TILE_SIZE 显式实例化、注册进被测内核表。
// ============================================================================

// sgemm_v2 —— 共享内存分块版：每线程一个输出元素，A / B 的 tile 进共享内存复用。
// 作用：C[globalRow*N + globalCol] = Σ_k A[globalRow*K + k] · B[k*N + globalCol]，fp32 累加。
// 参数：A(M×K) / B(K×N) / C(M×N) 为行主序设备指针；M / N / K 为维度。
// 返回值：无（结果写回 C）。
// 启动约束：block=(TILE_SIZE², 1)、grid=(ceil(M/TILE_SIZE), ceil(N/TILE_SIZE))、动态共享内存
//   0 B（静态共享内存 2·TILE_SIZE²·4 B，TILE_SIZE=32 时为 8 KiB）。blockDim.x 必须给满
//   TILE_SIZE² —— 线性映射把 threadIdx.x / TILE_SIZE 当作 tile 内行号，线程数不足时每 block
//   只覆盖 tile 的首行，未写入的输出保持原值且不报任何错误。
// 注意事项：M / N 越界线程只空转、不写 C；K 不是 TILE_SIZE 倍数时尾块在共享内存里补 0，
//   K == 0 时循环不执行、输出写 0；写回 C 必须在 k 循环外做一次，放在循环内会每轮多写
//   一次全局内存（512³ / TILE_SIZE=32 时约 15 次额外全量写）；共享内存 tile 的写入与读取
//   各配一次 __syncthreads，warp 内 A 同址广播、B 连续读取，均无 bank conflict。
template <const int TILE_SIZE>
__global__ void sgemm_v2(const float* A, const float* B, float* C, const int M, const int N,
                         const int K) {
	__shared__ float shared_A[TILE_SIZE * TILE_SIZE];  // A 的 tile，沿 k 方向向右滑动
	__shared__ float shared_B[TILE_SIZE * TILE_SIZE];  // B 的 tile，沿 k 方向向下滑动

	const int cRow = blockIdx.x;  // 本 block 的输出 tile 行号
	const int cCol = blockIdx.y;  // 本 block 的输出 tile 列号

	// 线性线程 → tile 内行列：warp 内 threadCol 连续，A / B 的全局装载因此是合并访问。
	const int threadRow = threadIdx.x / TILE_SIZE;
	const int threadCol = threadIdx.x % TILE_SIZE;

	const int globalRow = cRow * TILE_SIZE + threadRow;
	const int globalCol = cCol * TILE_SIZE + threadCol;

	// A / B / C 先偏移到本 block 负责区域的左上角，循环内只按 tile 步长前进。
	A += cRow * TILE_SIZE * K;                     // A 指向 A[cRow*TILE_SIZE][0]
	B += cCol * TILE_SIZE;                         // B 指向 B[0][cCol*TILE_SIZE]
	C += cRow * TILE_SIZE * N + cCol * TILE_SIZE;  // C 指向输出 tile 左上角

	float sum = 0.0f;

	for (int bkIdx = 0; bkIdx < K; bkIdx += TILE_SIZE) {
		shared_A[threadRow * TILE_SIZE + threadCol] =
		    (globalRow < M && bkIdx + threadCol < K) ? A[threadRow * K + threadCol] : 0.0f;
		shared_B[threadRow * TILE_SIZE + threadCol] =
		    (bkIdx + threadRow < K && globalCol < N) ? B[threadRow * N + threadCol] : 0.0f;
		__syncthreads();

		for (int k = 0; k < TILE_SIZE; ++k) {
			sum += shared_A[threadRow * TILE_SIZE + k] * shared_B[k * TILE_SIZE + threadCol];
		}
		__syncthreads();

		A += TILE_SIZE;
		B += TILE_SIZE * N;
	}

	if (globalRow < M && globalCol < N) {
		C[threadRow * N + threadCol] = sum;
	}
}

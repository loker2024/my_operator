#pragma once
// ============================================================================
// sgemm_v2.cuh —— SGEMM v2 共享内存分块模板内核（内联定义，由 main.cu 显式实例化）。
// 计算行主序 C(M×N) = A(M×K) × B(K×N)：一个 block 以 BLOCKSIZE² 个线性线程覆盖一个
// BLOCKSIZE×BLOCKSIZE 的输出 tile；k 方向按 BLOCKSIZE 步长把 A / B 的 tile 搬进共享内存，
// 每线程再从共享内存做 BLOCKSIZE 次乘加 —— 全局访存量降到 v1 的 1/BLOCKSIZE，tile 内
// 每个元素被 BLOCKSIZE 个线程复用。
// 模板内核内联定义在头文件（-rdc=false 下跨翻译单元引用 __global__ 模板特化已被 nvcc
// 弃用），故 main.cu 直接包含本文件并以具体 BLOCKSIZE 显式实例化、注册进被测内核表。
// ============================================================================

// sgemm_v2 —— 共享内存分块版：每线程一个输出元素，A / B 的 tile 进共享内存复用。
// 作用：C[globalRow*N + globalCol] = Σ_k A[globalRow*K + k] · B[k*N + globalCol]，fp32 累加。
// 参数：A(M×K) / B(K×N) / C(M×N) 为行主序设备指针；M / N / K 为维度。
// 返回值：无（结果写回 C）。
// 启动约束：block=(BLOCKSIZE², 1)、grid=(ceil(M/BLOCKSIZE), ceil(N/BLOCKSIZE))、动态共享内存
//   0 B（静态共享内存 2·BLOCKSIZE²·4 B，BLOCKSIZE=32 时为 8 KiB）。blockDim.x 必须给满
//   BLOCKSIZE² —— 线性映射把 threadIdx.x / BLOCKSIZE 当作 tile 内行号，线程数不足时每 block
//   只覆盖 tile 的首行，未写入的输出保持原值且不报任何错误。
// 注意事项：M / N 越界线程只空转、不写 C；K 不是 BLOCKSIZE 倍数时尾块在共享内存里补 0，
//   K == 0 时循环不执行、输出写 0；写回 C 必须在 k 循环外做一次，放在循环内会每轮多写
//   一次全局内存（512³ / BLOCKSIZE=32 时约 15 次额外全量写）；共享内存 tile 的写入与读取
//   各配一次 __syncthreads，warp 内 A 同址广播、B 连续读取，均无 bank conflict。
template <const int BLOCKSIZE>
__global__ void sgemm_v2(const float* A, const float* B, float* C, const int M, const int N,
                         const int K) {
	__shared__ float shared_A[BLOCKSIZE * BLOCKSIZE];  // A 的 tile，沿 k 方向向右滑动
	__shared__ float shared_B[BLOCKSIZE * BLOCKSIZE];  // B 的 tile，沿 k 方向向下滑动

	const int cRow = blockIdx.x;  // 本 block 的输出 tile 行号
	const int cCol = blockIdx.y;  // 本 block 的输出 tile 列号

	// 线性线程 → tile 内行列：warp 内 threadCol 连续，A / B 的全局装载因此是合并访问。
	const int threadRow = threadIdx.x / BLOCKSIZE;
	const int threadCol = threadIdx.x % BLOCKSIZE;

	const int globalRow = cRow * BLOCKSIZE + threadRow;
	const int globalCol = cCol * BLOCKSIZE + threadCol;

	// A / B / C 先偏移到本 block 负责区域的左上角，循环内只按 tile 步长前进。
	A += cRow * BLOCKSIZE * K;                     // A 指向 A[cRow*BLOCKSIZE][0]
	B += cCol * BLOCKSIZE;                         // B 指向 B[0][cCol*BLOCKSIZE]
	C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;  // C 指向输出 tile 左上角

	float sum = 0.0f;

	for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
		shared_A[threadRow * BLOCKSIZE + threadCol] =
		    (globalRow < M && bkIdx + threadCol < K) ? A[threadRow * K + threadCol] : 0.0f;
		shared_B[threadRow * BLOCKSIZE + threadCol] =
		    (bkIdx + threadRow < K && globalCol < N) ? B[threadRow * N + threadCol] : 0.0f;
		__syncthreads();

		for (int k = 0; k < BLOCKSIZE; ++k) {
			sum += shared_A[threadRow * BLOCKSIZE + k] * shared_B[k * BLOCKSIZE + threadCol];
		}
		__syncthreads();

		A += BLOCKSIZE;
		B += BLOCKSIZE * N;
	}

	if (globalRow < M && globalCol < N) {
		C[threadRow * N + threadCol] = sum;
	}
}

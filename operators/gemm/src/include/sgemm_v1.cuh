#pragma once
// ============================================================================
// sgemm_v1.cuh —— SGEMM v1 教学版模板内核（内联定义，由 main.cu 显式实例化注册）。
// 计算行主序 C(M×N) = A(M×K) × B(K×N)：一个 block 以 BLOCKSIZE² 个线性线程覆盖一个
// BLOCKSIZE×BLOCKSIZE 的输出 tile，每线程一个输出元素；不使用共享内存与寄存器分块，
// k 方向每步直接读全局内存 —— warp 内 B 的读取按列连续合并，A 的读取同址广播。
// 模板内核内联定义在头文件（-rdc=false 下跨翻译单元引用 __global__ 模板特化已被 nvcc
// 弃用），故 main.cu 直接包含本文件并以 BLOCKSIZE=32 显式实例化、注册进被测内核表。
// ============================================================================

// sgemm_v1 —— 每线程一个输出元素的全局内存版本（保留最初教学实现的线性索引映射）。
// 作用：C[cRow*N + cCol] = Σ_k A[cRow*K + k] · B[k*N + cCol]，k 方向 fp32 顺序累加。
// 参数：A(M×K) / B(K×N) / C(M×N) 为行主序设备指针；M / N / K 为无符号维度。
// 返回值：无（结果写回 C）。
// 启动约束：block=(BLOCKSIZE², 1)、grid=(ceil(M/BLOCKSIZE), ceil(N/BLOCKSIZE))、动态共享内存
//   0 B。blockDim.x 必须给满 BLOCKSIZE² —— 线性映射把 threadIdx.x / BLOCKSIZE 当作 tile 内
//   行号，线程数不足时每 block 只覆盖 tile 的首行，未写入的输出保持原值且不报任何错误。
// 注意事项：M / N 越界线程空转；K == 0 时输出写 0；累加为 fp32，长 K 的舍入误差由测试按
//   1e-3 容差校验；每个输出元素各读 K 次 A、K 次 B，无数据复用，性能受全局内存带宽限制。
template <const uint BLOCKSIZE>
__global__ void sgemm_v1(const float* A, const float* B, float* C, const uint M, const uint N,
                         const uint K) {
	// 线性线程 → tile 内行列：前 BLOCKSIZE 个线程填第 0 行，依此类推。
	const int cRow = blockIdx.x * BLOCKSIZE + threadIdx.x / BLOCKSIZE;
	const int cCol = blockIdx.y * BLOCKSIZE + threadIdx.x % BLOCKSIZE;

	if (cRow < M && cCol < N) {
		float sum = 0.0f;
		for (int k = 0; k < K; k++) {
			sum += A[cRow * K + k] * B[k * N + cCol];
		}
		C[cRow * N + cCol] = sum;
	}
}

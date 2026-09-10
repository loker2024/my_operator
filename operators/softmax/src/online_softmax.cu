// online_softmax.cu —— online_softmax.cuh 的实现：online_softmax_v0（单趟在线归约）。
// 接口契约 / 启动约束见 online_softmax.cuh，推导与实测见 README.md。

#include <cmath>  // expf / fmaxf / INFINITY

#include "online_softmax.cuh"

// online_softmax_v0 —— 每线程处理一行，行内单趟 online 归约
__global__ void online_softmax_v0(const float* input, float* output, const int M, const int N) {
	const int row = blockIdx.x * blockDim.x + threadIdx.x;  // 线程铺满行号
	if (row >= M) return;  // 空矩阵 / 超配 grid：越界行空转

	const float* x = input + row * N;
	float* y = output + row * N;

	// ① 一趟读行在线归约：m = 已扫子集最大值，d = Σexp(x_k - m)（以 m 为基准）
	float m = -INFINITY;
	float d = 0.0f;
	for (int i = 0; i < N; ++i) {
		const float xi = x[i];
		const float m_new = fmaxf(m, xi);
		d = d * expf(m - m_new) + expf(xi - m_new);  // 旧分母缩放回新基准 + 新元素贡献
		m = m_new;
	}

	// ② 归一化写回（不缓存整行，故 exp 每元素算 2 次：在线更新 1 次 + 此处 1 次）。
	//    空行 N == 0：inv_d == inf，但循环 0 次、不写任何元素
	const float inv_d = 1.0f / d;
	for (int i = 0; i < N; ++i) {
		y[i] = expf(x[i] - m) * inv_d;
	}
}

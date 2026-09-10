// online_softmax.cu —— online_softmax.cuh 的实现：online_softmax_v0/v1/v2/v3/v3_false
//（单趟在线归约）。
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

// ---------------------------------------------------------------------------
// mergeOnline —— 二元组 (m, d) 的结合运算（online 归约的核心，单元素插入与
// 子集合并都走它）
// ---------------------------------------------------------------------------
// 把右侧 (m2, d2) 并入左侧：m_new = max(m, m2)，两侧分母各按「自身基准 → 新基准」
// 之差缩放后相加。单元素插入即 d2 = 1（exp(x - x) = 1）。
// 空集（m == -INFINITY、d == 0）跳过该侧缩放：否则 (-inf) - (-inf) = NaN 会经
// expf 污染整个分母 —— 未分到元素的线程 / warp 必然以空集参与合并（N 不整除或
// 小于 blockDim.x，见 v1 声明）。
__device__ void mergeOnline(float& m, float& d, const float m2, const float d2) {
	const float m_new = fmaxf(m, m2);
	const float scale = (m == -INFINITY) ? 0.0f : expf(m - m_new);
	const float scale2 = (m2 == -INFINITY) ? 0.0f : expf(m2 - m_new);
	d = d * scale + d2 * scale2;
	m = m_new;
}

// ---------------------------------------------------------------------------
// warpReduceOnline —— 单 warp 内 shuffle 归约 (m, d)（结果收敛到 lane 0）
// ---------------------------------------------------------------------------
// 5 轮 __shfl_down_sync（offset 16 → 1）在寄存器间两两 mergeOnline，不碰共享内存；
// mask 0xffffffff 要求完整 warp 收敛调用，故调用点不可让 warp 内部分线程提前退出。
__device__ void warpReduceOnline(float& m, float& d) {
	for (int offset = 16; offset > 0; offset >>= 1) {
		const float m2 = __shfl_down_sync(0xffffffff, m, offset);
		const float d2 = __shfl_down_sync(0xffffffff, d, offset);
		mergeOnline(m, d, m2, d2);
	}
}

// ---------------------------------------------------------------------------
// blockReduceOnline —— 块内两级 warp shuffle 归约 (m, d)，结果广播回全体线程
// ---------------------------------------------------------------------------
// 结构同 softmax.cu 的 blockReduce*Shuffle：① warpReduceOnline 把每 warp 归为 1
// 个二元组，lane 0 写入 warp_m/warp_d[wid]；② __syncthreads 后 warp 0 再合并
// num_warps 个 warp 值（lane >= num_warps 以空集单位元 -inf / 0 参与）；③ 结果
// 经 block_m/block_d + __syncthreads 广播 —— 写回遍需要每个线程都拿到整行的
// (m, d)，不能只由 tid 0 持有（区别于 reduce 只需 tid 0 写输出）。
// 约束：blockDim.x 为 32 的倍数且 <= 1024，warp 值才装得进 warp_m[32] / warp_d[32]。
__device__ void blockReduceOnline(float& m, float& d) {
	__shared__ float warp_m[32];  // 各 warp 的归约二元组（每 warp 由 lane 0 写入）
	__shared__ float warp_d[32];
	__shared__ float block_m;  // 广播用块归约二元组
	__shared__ float block_d;

	const int lane = threadIdx.x % 32;
	const int wid = threadIdx.x / 32;

	warpReduceOnline(m, d);  // ① warp 内归约：lane 0 持有该 warp 结果
	if (lane == 0) {
		warp_m[wid] = m;
		warp_d[wid] = d;
	}
	__syncthreads();  // 全部 warp 值就绪后才能被 warp 0 读取

	const int num_warps = blockDim.x / 32;
	if (wid == 0) {  // ② warp 0 合并 num_warps 个 warp 值
		m = (lane < num_warps) ? warp_m[lane] : -INFINITY;
		d = (lane < num_warps) ? warp_d[lane] : 0.0f;
		warpReduceOnline(m, d);

		if (lane == 0) {  // ③ tid 0（warp 0 lane 0）持有块结果，广播给全 block
			block_m = m;
			block_d = d;
		}
	}
	__syncthreads();  // 广播：全体读到块归约值后才允许进入写回遍

	m = block_m;
	d = block_d;
}

// ---------------------------------------------------------------------------
// online_softmax_v1 —— 每行一个 block，单趟在线归约 + 两级 warp shuffle 合并
// ---------------------------------------------------------------------------
// 行映射 / 启动约束见 online_softmax.cuh 的 v1 声明；与 v0 的差异仅在把「一线程
// 独占一行」换成「块内协作」：读合并、并行度由行数提升到 rows × blockDim.x。
__global__ void online_softmax_v1(const float* input, float* output, const int M, const int N) {
	const int row = blockIdx.x;  // 每 block 处理一行
	if (row >= M) return;        // 空矩阵 / 超配 grid：越界行空转
	const int tid = threadIdx.x;

	const float* x = input + row * N;
	float* y = output + row * N;

	// ① 一趟 stride 扫行：每线程在线归约自己的列子集 → 块内两级 shuffle 合并出
	//    整行 (m, d)（求 m 与求 Σexp 合并为一趟全局读）
	float m = -INFINITY;
	float d = 0.0f;
	for (int i = tid; i < N; i += blockDim.x) {
		mergeOnline(m, d, x[i], 1.0f);  // 插入单元素：该元素自身基准下的分母为 1
	}
	blockReduceOnline(m, d);

	// ② 第二趟读行重算 exp(x - m) / d 写回（不缓存整行，exp 每元素算 2 次）。
	//    空行 N == 0：d == 0 → inv_d == inf，但循环 0 次、不写任何元素
	const float inv_d = 1.0f / d;
	for (int i = tid; i < N; i += blockDim.x) {
		y[i] = expf(x[i] - m) * inv_d;
	}
}

// ---------------------------------------------------------------------------
// online_softmax_v2 —— v1 + float4 向量化（列宽为 4 的倍数时）
// ---------------------------------------------------------------------------
// 行映射 / 启动约束同 v1，见 online_softmax.cuh 的 v2 声明（含“非 4 倍列宽为何整行
// 回退标量”的说明）。与 v1 的差异仅在行内访问：N % 4 == 0 时行首 16 B 对齐，单趟
// 在线归约与写回都以 float4 为单位 stride 扫行（每轮 4 列、读/写指令数为标量 1/4，
// 4 分量逐一 mergeOnline），否则整行走 v1 式标量两遍。
__global__ void online_softmax_v2(const float* input, float* output, const int M, const int N) {
	const int row = blockIdx.x;  // 每 block 处理一行
	if (row >= M) return;        // 空矩阵 / 超配 grid：越界行空转
	const int tid = threadIdx.x;

	const float* x = input + row * N;
	float* y = output + row * N;

	float m = -INFINITY;
	float d = 0.0f;

	if (N % 4 == 0) {
		// 列宽为 4 的倍数：行首 16 B 对齐，float4 主循环（每轮 stride 处理 4 列；
		// 空行 N == 0 时 n4 == 0，两遍循环 0 次、不读不写）
		const int n4 = N / 4;
		const float4* x4 = reinterpret_cast<const float4*>(x);
		float4* y4 = reinterpret_cast<float4*>(y);

		// ① 一趟 float4 stride 扫行：4 分量逐一插入在线归约 → 块内 shuffle 合并
		for (int i = tid; i < n4; i += blockDim.x) {
			const float4 v = x4[i];
			mergeOnline(m, d, v.x, 1.0f);
			mergeOnline(m, d, v.y, 1.0f);
			mergeOnline(m, d, v.z, 1.0f);
			mergeOnline(m, d, v.w, 1.0f);
		}
		blockReduceOnline(m, d);

		// ② 第二趟 float4 读行重算 exp(x - m) / d 并整写回（无标量尾部）
		const float inv_d = 1.0f / d;
		for (int i = tid; i < n4; i += blockDim.x) {
			const float4 v = x4[i];
			y4[i] = make_float4(expf(v.x - m) * inv_d, expf(v.y - m) * inv_d, expf(v.z - m) * inv_d,
			                    expf(v.w - m) * inv_d);
		}
	} else {
		// 列宽非 4 的倍数：行首不保证 16 B 对齐，整行回退标量两遍（语义同 v1）
		// ① 一趟 stride 扫行：每线程在线归约自己的列子集 → 块内两级 shuffle 合并出
		//    整行 (m, d)（求 m 与求 Σexp 合并为一趟全局读）
		for (int i = tid; i < N; i += blockDim.x) {
			mergeOnline(m, d, x[i], 1.0f);  // 插入单元素：该元素自身基准下的分母为 1
		}
		blockReduceOnline(m, d);

		// ② 第二趟读行重算 exp(x - m) / d 写回（不缓存整行，exp 每元素算 2 次）。
		//    空行 N == 0：d == 0 → inv_d == inf，但循环 0 次、不写任何元素
		const float inv_d = 1.0f / d;
		for (int i = tid; i < N; i += blockDim.x) {
			y[i] = expf(x[i] - m) * inv_d;
		}
	}
}

// ---------------------------------------------------------------------------
// online_softmax_v3 —— v1/v2 的归约框架 + 寄存器分片缓存（真正落在寄存器）
// ---------------------------------------------------------------------------
// 行映射 / 归约 / 启动约束见 online_softmax.cuh 的 v3 声明。与 v1/v2 的差异只在
// 写回遍的 x 来源：每线程元素数 <= kRegTile（列宽可被单 block 一轮扫完）时，第 1 遍
// 在线归约的同时把本线程负责的列缓存进寄存器数组，写回遍直接取寄存器、省掉第 2 遍
// 全局读；列宽过大、寄存器分片装不下时自动回退 v1/v2 式两遍重读。
//
// 真寄存器的关键（对照朴素的 reg_cache[count++] 被 ptxas 降级为 local memory）：
//   * 循环上界 kRegTile 是编译期常量、循环体无运行期计数 → #pragma unroll 整体展开；
//   * 展开后下标 k 是常量，reg[k] 静态寻址 —— 寄存器不可被运行期索引，只有静态下标
//     才留在寄存器；若改用 i（运行期列号）或 count++ 作下标必退化为 local memory
//     （≈显存，只是有 L1 缓存，见 notes）。
// 分派条件 (N + blockDim.x - 1) / blockDim.x <= kRegTile 只依赖 N 与 blockDim、对整
// block 一致，故两条分支各自调用含 __syncthreads 的 blockReduceOnline 不会死锁。
constexpr int kRegTile = 16;  // 每线程寄存器分片容量（block = 256 时覆盖 N <= 4096）

__global__ void online_softmax_v3(const float* input, float* output, const int M, const int N) {
	const int row = blockIdx.x;  // 每 block 处理一行
	if (row >= M) return;        // 空矩阵 / 超配 grid：越界行空转
	const int tid = threadIdx.x;

	const float* x = input + row * N;
	float* y = output + row * N;

	float local_m = -INFINITY;
	float local_d = 0.0f;

	if ((N + blockDim.x - 1) / blockDim.x <= kRegTile) {
		// 寄存器路径：定长全展开循环，k 为编译期常量 → reg[k] 真进寄存器
		float reg[kRegTile];
#pragma unroll
		for (int k = 0; k < kRegTile; ++k) {
			const int i = tid + k * blockDim.x;
			if (i < N) {
				const float xi = x[i];
				reg[k] = xi;  // 静态下标：进寄存器，非 local memory
				mergeOnline(local_m, local_d, xi, 1.0f);
			} else {
				reg[k] = 0.0f;  // 越界槽位初始化（写回遍同样以 i < N 守卫，不参与结果）
			}
		}
		blockReduceOnline(local_m, local_d);  // 两级 warp shuffle 合并出整行 (m, d)
		const float row_max = local_m;
		const float inv_sum = 1.0f / local_d;

#pragma unroll
		for (int k = 0; k < kRegTile; ++k) {
			const int i = tid + k * blockDim.x;
			if (i < N) y[i] = expf(reg[k] - row_max) * inv_sum;  // 取寄存器 x，免第 2 遍全局读
		}
	} else {
		// 回退路径：每线程元素数超过 kRegTile（列宽过大）→ 不缓存，第 2 遍重读全局
		for (int i = tid; i < N; i += blockDim.x) {
			mergeOnline(local_m, local_d, x[i], 1.0f);
		}
		blockReduceOnline(local_m, local_d);
		const float row_max = local_m;
		const float inv_sum = 1.0f / local_d;

		for (int i = tid; i < N; i += blockDim.x) {
			y[i] = expf(x[i] - row_max) * inv_sum;
		}
	}
}

// ---------------------------------------------------------------------------
// online_softmax_v3_false —— v3 的「假寄存器」反面对照（运行期下标 → local memory）
// ---------------------------------------------------------------------------
// 行映射 / 归约 / 启动约束见 online_softmax.cuh 的 v3_false 声明，与 v3 逐项同构，唯一
// 区别：缓存本线程列时用运行期下标 `reg_cache[count++]`，而非 v3 的编译期常量下标 `reg[k]`。
// 寄存器不可被运行期索引，ptxas 只能把整个数组放进 local memory（物理是显存、仅靠 L1
// 缓存），故第 1 遍的每次写入、写回遍的每次读取都是 local memory 往返。
// 本版**不做列宽分派、无回退路径** —— 始终缓存，契约要求 ceil(N / blockDim.x) <= kRegTile
// （block = 256 时 N <= 4096）；超过则 reg_cache 越界写入（未定义行为），由调用方保证。
// 去掉分支是为了与 v3 形成最干净的对照（无分派开销）；该版本只用于量化「local memory 缓存
// vs 真寄存器缓存 vs 重读全局」的代价，非性能候选。
__global__ void online_softmax_v3_false(const float* input, float* output, const int M,
                                        const int N) {
	const int row = blockIdx.x;  // 每 block 处理一行
	if (row >= M) return;        // 空矩阵 / 超配 grid：越界行空转
	const int tid = threadIdx.x;

	const float* x = input + row * N;
	float* y = output + row * N;

	float local_m = -INFINITY;
	float local_d = 0.0f;

	// 定长数组 + 运行期下标 → 数组被降级为 local memory。循环上界 N 是运行期值、循环体
	// 带自增计数，编译器无法把下标常量化。
	float reg_cache[kRegTile];
	int count = 0;
	for (int i = tid; i < N; i += blockDim.x) {
		const float xi = x[i];
		reg_cache[count++] = xi;  // 运行期下标：local memory 写（非寄存器）
		mergeOnline(local_m, local_d, xi, 1.0f);
	}
	blockReduceOnline(local_m, local_d);  // 两级 warp shuffle 合并出整行 (m, d)
	const float row_max = local_m;
	const float inv_sum = 1.0f / local_d;

	count = 0;
	for (int i = tid; i < N; i += blockDim.x) {
		// 运行期下标：local memory 读，省掉第 2 遍全局读但代价是 L1/显存往返
		y[i] = expf(reg_cache[count++] - row_max) * inv_sum;
	}
}

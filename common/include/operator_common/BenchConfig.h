#pragma once
// ============================================================================
// BenchConfig.h —— 公共基准采样配置：严格档采样量按设备算力分档
//   各算子若把严格档的固定次数写死在测试驱动里，换 GPU 就要改代码：强 GPU 上
//   采样偏稀，弱 GPU 上总时长失控。这里按「当前设备 SM 数」相对基准机
//   （docs/benchmark-methodology.md 的口径标定机 RTX 4060 Laptop，24 SM）分档
//   放大采样量，使总内核调用次数与基准机同量级，同时不牺牲分位分辨率。
//   引用方式：#include "operator_common/BenchConfig.h"
// ============================================================================
#include <cuda_runtime.h>

#include "operator_common/cuda_check.h"

// 严格档采样参数（各字段含义见 MakeStrictBenchConfig）。
struct StrictBenchConfig {
	int warmup_iterations;  // 计时前的预热次数（不参与统计）
	int sample_count;       // 采样组数，决定 P5/P95 的分位分辨率
	int iterations;         // 每组内的连续内核调用次数
};

// 基准机（口径标定用）的 SM 数：RTX 4060 Laptop。
constexpr int kBenchRefSmCount = 24;
// 分档倍率上限：GPU 再强也不无限拉长单次严格基准的总时长。
constexpr int kBenchMaxTier = 10;

// 当前设备的 SM 数；查询失败时回退到基准机 SM 数并清除错误状态，保证严格基准
// 不因设备查询失败而中断。
inline int BenchDeviceSmCount() {
	int dev = 0;
	if (cudaGetDevice(&dev) != cudaSuccess) {
		cudaGetLastError();
		return kBenchRefSmCount;
	}
	int sm_count = 0;
	if (cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) {
		cudaGetLastError();
		return kBenchRefSmCount;
	}
	return sm_count > 0 ? sm_count : kBenchRefSmCount;
}

// 由设备算力分档得到严格档采样参数：
//   tier         = clamp(round(SM 数 / 24), 1, 10)
//   warmup       = 100  × tier
//   iterations   = 1000 × tier
//   sample_count = 21（固定，保证 P5/P95 的分位分辨率）
// 基准机（24 SM）即文档口径的「100 次预热 + 21 组 × 1000 次」；更强 GPU 单次内核
// 更快，用更多采样换统计密度，总调用次数与基准机同量级。
// 性能代价：调用次数随 tier 线性增长，慢内核（如 softmax v0）在大 GPU 上仍是单
// 场景秒级到分钟级；如需更稀的采样，以开发档（strict_benchmark = false）运行即可。
inline StrictBenchConfig MakeStrictBenchConfig() {
	const int tier_raw = (BenchDeviceSmCount() + kBenchRefSmCount / 2) / kBenchRefSmCount;
	const int tier = tier_raw < 1 ? 1 : (tier_raw > kBenchMaxTier ? kBenchMaxTier : tier_raw);
	return StrictBenchConfig{100 * tier, 21, 1000 * tier};
}

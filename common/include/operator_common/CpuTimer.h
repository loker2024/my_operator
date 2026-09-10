#pragma once
// CPU 侧计时工具：基于 std::chrono::steady_clock，只依赖 C++ 标准库，
// 不依赖 CUDA，可在纯 CPU 工程中直接使用（与 GpuTimer 接口对齐）。
// 引用方式：#include "operator_common/CpuTimer.h"
#include <chrono>

class CpuTimer {
   public:
	// 记录起始点。
	void Start() { start_ = std::chrono::steady_clock::now(); }

	// 返回自 Start 起的耗时（毫秒）。仅读取一次时钟，不阻塞、不重置。
	float StopMs() {
		return std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start_)
		    .count();
	}

   private:
	std::chrono::steady_clock::time_point start_{};
};

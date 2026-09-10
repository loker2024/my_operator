#pragma once
// 主机侧 GPU 计时工具：基于 CUDA event，误差可控且能隐式同步设备。
// 引用方式：#include "operator_common/GpuTimer.h"
#include <cuda_runtime.h>

#include <cstdint>

#include "operator_common/cuda_check.h"

class GpuTimer {
   public:
	GpuTimer() {
		CUDA_CHECK(cudaEventCreate(&start_));
		CUDA_CHECK(cudaEventCreate(&stop_));
	}
	~GpuTimer() {
		cudaEventDestroy(start_);
		cudaEventDestroy(stop_);
	}

	// 记录起始点（不隐式同步）。
	void Start() { CUDA_CHECK(cudaEventRecord(start_)); }

	// 记录结束点并同步，返回耗时（毫秒）。
	float StopMs() {
		CUDA_CHECK(cudaEventRecord(stop_));
		CUDA_CHECK(cudaEventSynchronize(stop_));
		float ms = 0.0f;
		CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
		return ms;
	}

   private:
	cudaEvent_t start_ = nullptr;
	cudaEvent_t stop_ = nullptr;
};

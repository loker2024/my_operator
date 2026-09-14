// softmax_cudnn.cu —— cuDNN ACCURATE + MODE_INSTANCE 的逐行 Softmax 对照。

#include "include/softmax_cudnn.cuh"

#ifdef SOFTMAX_WITH_CUDNN
#include <cudnn.h>

#include <cstdio>
#include <cstdlib>

namespace {
struct CudnnCtx {
	cudnnHandle_t handle = nullptr;
	cudnnTensorDescriptor_t desc = nullptr;
	int rows = -1;
	int cols = -1;
};

#define SOFTMAX_CUDNN_CHECK(expr)                                                             \
	do {                                                                                      \
		const cudnnStatus_t status__ = (expr);                                                \
		if (status__ != CUDNN_STATUS_SUCCESS) {                                               \
			std::fprintf(stderr, "cuDNN error %s (%d) at %s:%d in %s: %s\\n",                 \
			             cudnnGetErrorString(status__), static_cast<int>(status__), __FILE__, \
			             __LINE__, __func__, #expr);                                          \
			std::abort();                                                                     \
		}                                                                                     \
	} while (0)
}  // namespace

void softmax_cudnn(const float* input, float* output, int M, int N) {
	if (M <= 0 || N <= 0) return;
	static CudnnCtx context;
	if (context.handle == nullptr) {
		SOFTMAX_CUDNN_CHECK(cudnnCreate(&context.handle));
		SOFTMAX_CUDNN_CHECK(cudnnCreateTensorDescriptor(&context.desc));
		SOFTMAX_CUDNN_CHECK(cudnnSetStream(context.handle, nullptr));
	}
	if (context.rows != M || context.cols != N) {
		SOFTMAX_CUDNN_CHECK(
		    cudnnSetTensor4dDescriptorEx(context.desc, CUDNN_DATA_FLOAT, M, 1, 1, N, N, N, N, 1));
		context.rows = M;
		context.cols = N;
	}
	const float alpha = 1.0f;
	const float beta = 0.0f;
	SOFTMAX_CUDNN_CHECK(cudnnSoftmaxForward(context.handle, CUDNN_SOFTMAX_ACCURATE,
	                                        CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, context.desc,
	                                        input, &beta, context.desc, output));
}
#endif

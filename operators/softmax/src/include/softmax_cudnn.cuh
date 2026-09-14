#pragma once
// softmax_cudnn.cuh —— 可选 cuDNN MODE_INSTANCE 行 softmax 对照接口。

#ifdef SOFTMAX_WITH_CUDNN
void softmax_cudnn(const float* input, float* output, int M, int N);
#endif

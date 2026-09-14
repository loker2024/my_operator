#pragma once
// softmax_reference.cuh —— Softmax 测试共用的函数类型与主机 double 参考实现接口。

using SoftmaxKernel = void (*)(const float* input, float* output, int M, int N);
using SoftmaxHostKernel = void (*)(const float* input, float* output, int M, int N);

void softmax_cpu(const float* input, float* output, int M, int N);

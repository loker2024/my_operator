// 临时测试：softmax_v3 在 N 不被 4 整除时的行为
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

__inline__ __device__ float warpReduceMaxShuffle(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffffu, val, offset));
    return val;
}

__inline__ __device__ float warpReduceSumShuffle(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffffu, val, offset);
    return val;
}

__inline__ __device__ float blockReduceMaxShuffle(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    val = warpReduceMaxShuffle(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    int nWarps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < nWarps) ? shared[lane] : -INFINITY;
    if (wid == 0) val = warpReduceMaxShuffle(val);
    if (threadIdx.x == 0) shared[0] = val;
    __syncthreads();
    return shared[0];
}

__inline__ __device__ float blockReduceSumShuffle(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    val = warpReduceSumShuffle(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    int nWarps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < nWarps) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSumShuffle(val);
    if (threadIdx.x == 0) shared[0] = val;
    __syncthreads();
    return shared[0];
}

// ===== 用户内核，原样保留 =====
__global__ void softmax_v3(float* input, float* output, int M, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float* x = input  + row * N;
    float* y = output + row * N;

    // 向量化指针
    float4* x4 = reinterpret_cast<float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);
    int N4 = N / 4;  // float4 元素数

    // Pass 1: 向量化求最大值
    float local_max = -INFINITY;
    for (int i = tid; i < N4; i += blockDim.x) {
        float4 data = x4[i];
        local_max = fmaxf(local_max, fmaxf(fmaxf(data.x, data.y),
                                            fmaxf(data.z, data.w)));
    }
    // 处理尾部元素（N 不是 4 的倍数时）
    for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }
    float max_val = blockReduceMaxShuffle(local_max);

    // Pass 2: 向量化求指数和
    float local_sum = 0.0f;
    for (int i = tid; i < N4; i += blockDim.x) {
        float4 data = x4[i];
        local_sum += expf(data.x - max_val) + expf(data.y - max_val)
                   + expf(data.z - max_val) + expf(data.w - max_val);
    }
    for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
        local_sum += expf(x[i] - max_val);
    }
    float sum = blockReduceSumShuffle(local_sum);

    // Pass 3: 向量化归一化
    float inv_sum = 1.0f / sum;
    for (int i = tid; i < N4; i += blockDim.x) {
        float4 data = x4[i];
        float4 result;
        result.x = expf(data.x - max_val) * inv_sum;
        result.y = expf(data.y - max_val) * inv_sum;
        result.z = expf(data.z - max_val) * inv_sum;
        result.w = expf(data.w - max_val) * inv_sum;
        y4[i] = result;
    }
    for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
        y[i] = expf(x[i] - max_val) * inv_sum;
    }
}

int main() {
    const int M = 8;
    const int N = 1023;   // 1023 % 4 = 3，非 4 整除
    const int total = M * N;

    float *h_in  = (float*)malloc(total * sizeof(float));
    float *h_out = (float*)malloc(total * sizeof(float));
    for (int i = 0; i < total; i++) h_in[i] = ((i * 37) % 17) * 0.5f - 4.0f;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, total * sizeof(float), cudaMemcpyHostToDevice));

    printf("Launch: M=%d, N=%d (N %% 4 = %d), blockDim=256\n", M, N, N % 4);
    softmax_v3<<<M, 256>>>(d_in, d_out, M, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[launch error] %s\n", cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[sync error]   %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaMemcpy(h_out, d_out, total * sizeof(float), cudaMemcpyDeviceToHost));

    // 行和校验
    int bad_rows = 0;
    for (int r = 0; r < M; r++) {
        float s = 0.0f;
        for (int c = 0; c < N; c++) s += h_out[r * N + c];
        if (fabsf(s - 1.0f) > 1e-3) {
            if (bad_rows < 3) printf("row %d sum = %f (expect 1.0)\n", r, s);
            bad_rows++;
        }
    }
    printf(bad_rows == 0 ? "PASS: all rows normalized\n"
                         : "FAIL: %d/%d rows wrong\n", bad_rows == 0 ? 0 : bad_rows, M);
    printf("Kernel finished without CUDA error.\n");
    return 0;

// (base) loker1@localhost:~/my_operator/operators/softmax/src$ nvcc -o softmax_v3 softmax_v3_test.cu
// nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
// (base) loker1@localhost:~/my_operator/operators/softmax/src$ ./softmax_v3 
// Launch: M=8, N=1023 (N % 4 = 3), blockDim=256
// [sync error]   misaligned address
}

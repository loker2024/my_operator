template<const uint BLOCKSIZE>
__global__ void sgemm_v1(const float* A, const float* B, float* C, const uint M, const uint N, const uint K){
    const int cRow=blockIdx.x*BLOCKSIZE+threadIdx.x/BLOCKSIZE;
    const int cCol=blockIdx.y*BLOCKSIZE+threadIdx.x%BLOCKSIZE;

    if(cRow<M && cCol<N){
        float sum=0.0f;
        for (int k=0;k<K;k++){
            sum+=A[cRow*K+k]*B[k*N+cCol];
        }
        C[cRow*N+cCol]=sum;
    }
}

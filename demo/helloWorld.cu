#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>

using namespace std;

__global__ void helloWorld() {
	printf("Hello World from GPU!\n");
}

int main() {
	helloWorld<<<1, 1>>>();
	cudaDeviceSynchronize();
	return 0;

	// nvcc -arch=sm_89 -o helloWorld demo/helloWorld.cu
	//./helloWorld
}
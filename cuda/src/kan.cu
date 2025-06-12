#include "kan.h"

__global__ void kanKernel(float *y){

	// initialise random state
	curandStatePhilox4_32_10_t localRandState;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // Each thread gets same seed, a different sequence number, no offset
    curand_init(seed, idx, 0, &localRandState);

    y[idx] = curand_uniform(&localRandState);
	// input x is of size batch_size,input_dim
	// output y is of size batch_size, output_dim
	// multiply with harmonics, randomly generated in a certain range

}

int main(){
	int blocks = 10000;
	int threads = 1024;
	int totalThreads = blocks*threads;
	float *hostResults = (float *)calloc(totalThreads, sizeof(float));
	float *devResults = (float *)calloc(totalThreads, sizeof(float));
    /* Allocate space for results on device */
    cudaMalloc((void **)&devResults, totalThreads *sizeof(float));
    cudaMalloc((void **)&devResults, totalThreads *sizeof(float));
    cudaMemset(devResults, 0, totalThreads *sizeof(float));

	kanKernel<<<blocks,threads>>>(devResults);

	cudaMemcpy(hostResults, devResults, totalThreads * sizeof(float), cudaMemcpyDeviceToHost);

	for(int i=0;i<totalThreads;i++){
		std::cout << "Random no: " << hostResults[i] << std::endl;
	}
}

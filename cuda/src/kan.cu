#include "kan.h"

__global__ void randKernel(float *y){

	// initialise random state
	curandStatePhilox4_32_10_t localRandState;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // Each thread gets same seed, a different sequence number, no offset
    curand_init(seed, idx, 0, &localRandState);

    y[idx] = curand_uniform(&localRandState);
	// input x is of size batch_size,input_dim
	// output y is of size batch_size, output_dim

	// first x * k


}

__global__ void kanKernel(
	float ***x,
	float ***fourierCoeffs,
	float **y
	){

	// check id value bounds

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int idz = threadIdx.z + blockIdx.z * blockDim.z;
	// input x is of size batch_size, input_dim
	// output y is of size batch_size, output_dim

	// x*k
	float x_element = x[idz][idy][0];

	__syncthreads();

	x[idz][idy][idx] = x_element * idz;


	float cosTerms[batchSize][inputDim][numHarmonics];
	float sinTerms[batchSize][inputDim][numHarmonics];

	float trigInp = x[idz][idy][idx];
	float cosRes = 0;
	float sinRes = 0;

	sincosf(trigInp,&sinRes,&cosRes);

	cosTerms[idz][idy][idx] = cosRes;
	sinTerms[idz][idy][idx] = sinRes;

	// second bounds check for output dims

	float yCos[batchSize][outputDim];
	float ySin[batchSize][outputDim];

	// optimize
	for(int i=0;i<inputDim;i++){
		for(int j=0;j<numHarmonics;j++){
			yCos[idz][idy] += cosTerms[idz][i][j] * fourierCoeffs[idy][i][j];
			ySin[idz][idy] += sinTerms[idz][i][j] * fourierCoeffs[idy][i][j];
		}
	}

	y[idz][idy] = yCos[idz][idy] + ySin[idz][idy] + bias;
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

	randKernel<<<blocks,threads>>>(devResults);

	cudaMemcpy(hostResults, devResults, totalThreads * sizeof(float), cudaMemcpyDeviceToHost);

	for(int i=0;i<totalThreads;i++){
		std::cout << "Random no: " << hostResults[i] << std::endl;
	}
}

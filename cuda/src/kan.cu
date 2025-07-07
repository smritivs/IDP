#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <iterator>
#include <algorithm>

const unsigned int batchSize = 16;
const unsigned int numHarmonics = 4;
const unsigned int inputDim = 8;
const unsigned int outputDim = 12;
constexpr unsigned int yThreads = (inputDim > outputDim) ? inputDim : outputDim;

const unsigned int xBlocks = 1;
const unsigned int yBlocks = 1;
const unsigned int zBlocks = 1;

const int bias = 0;

#include <ctime>
#include <chrono>

template <typename fp>
__global__ void kanKernel(
	fp *x,
	fp *fourierCoeffsCos,
	fp *fourierCoeffsSin,
	fp *y,
	const int batchSize, const int inputDim, const int outputDim, const int numHarmonics
	){

	// check id value bounds
	// idx -> weights
	// idy -> input dim
	// idz -> batch
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int idz = threadIdx.z + blockIdx.z * blockDim.z;

	// input x is of size batch_size, input_dim
	// output y is of size batch_size, output_dim

	// x*k
	fp x_element = x[idz*inputDim + idy];

	__syncthreads();

	fp trigInp = x_element * (idx + 1);

	fp cosRes = 0;
	fp sinRes = 0;

	sincosf(trigInp,&sinRes,&cosRes);

	__shared__ fp cosTerms[49152/8];
	__shared__ fp sinTerms[49152/8];



	cosTerms[idz*inputDim*numHarmonics + idy*numHarmonics + idx] = cosRes;
	sinTerms[idz*inputDim*numHarmonics + idy*numHarmonics + idx] = sinRes;

	// second bounds check for output dims

	__shared__ fp yCos[49152/8];
	__shared__ fp ySin[49152/8];

	fp yCosSum = 0;
	fp ySinSum = 0;

	// optimize
	for(int i=0;i<inputDim;i++){
		for(int j=0;j<numHarmonics;j++){
			// yCosSum += cosTerms[idz*inputDim + i*numHarmonics + j] * fourierCoeffsCos[idy*outputDim + i*numHarmonics + j];
			// ySinSum += sinTerms[idz*inputDim + i*numHarmonics + j] * fourierCoeffsSin[idy*outputDim + i*numHarmonics + j];
			yCosSum += cosTerms[idz*inputDim*numHarmonics + i*numHarmonics + j] * fourierCoeffsCos[idy*inputDim*numHarmonics + i*numHarmonics + j];
			ySinSum += sinTerms[idz*inputDim*numHarmonics + i*numHarmonics + j] * fourierCoeffsSin[idy*inputDim*numHarmonics + i*numHarmonics + j];


		}
	}

	y[idz*outputDim + idy] = yCosSum + ySinSum + bias;
}


extern "C" void kanGPU(float *a, float *b, float *c, float *res, const int batchSize, const int inputDim, const int outputDim, const int numHarmonics){

	int totalThreads = numHarmonics * inputDim * batchSize;

	// change to appropriate number
	dim3 gridDim(1,1,1);
	dim3 blockDim(numHarmonics,inputDim,batchSize);

	// allocate device and host memory
	float *x;
	float *fcc;
	float *fcs;
	float *y;
	// copy data to GPU
	cudaMalloc((void **)&x, totalThreads*sizeof(float));
	cudaMalloc((void **)&fcc, outputDim*inputDim*numHarmonics*sizeof(float));
	cudaMalloc((void **)&fcs, outputDim*inputDim*numHarmonics*sizeof(float));
	cudaMalloc((void **)&y, totalThreads*sizeof(float));
    cudaMemset(y, 0, batchSize*outputDim*sizeof(float));


	cudaMemcpy(x, a, totalThreads*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(fcc, b, outputDim*inputDim*numHarmonics*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(fcs, c, outputDim*inputDim*numHarmonics*sizeof(float), cudaMemcpyHostToDevice);

	// benchmark

    kanKernel<float><<<gridDim,blockDim>>>(x,fcc,fcs,y,batchSize,inputDim,outputDim,numHarmonics);

    // copy back to CPU
	cudaMemcpy(res, y, batchSize*outputDim*sizeof(float), cudaMemcpyDeviceToHost);


}

// int main(){
// 	float y[batchSize*outputDim] = {0};

// 	// read random test vector from file

// 	std::ifstream xfile("./input_x.txt");
// 	std::istream_iterator<float> xstart(xfile), xend;
// 	std::vector <float> xVec(xstart,xend);

// 	std::ifstream fccfile("./input_fourier_coeffs_cos.txt");
// 	std::istream_iterator<float> fccstart(fccfile), fccend;
// 	std::vector <float> fccVec(fccstart,fccend);

// 	std::ifstream fcsfile("./input_fourier_coeffs_sin.txt");
// 	std::istream_iterator<float> fcsstart(fcsfile), fcsend;
// 	std::vector <float> fcsVec(fcsstart,fcsend);

// 	float x[batchSize*inputDim*numHarmonics] = {0};
// 	float fourierCoeffsCos[outputDim*inputDim*numHarmonics] = {0};
// 	float fourierCoeffsSin[outputDim*inputDim*numHarmonics] = {0};

// 	std::copy(xVec.begin(), xVec.end(), x);
// 	std::copy(fccVec.begin(), fccVec.end(), fourierCoeffsCos);
// 	std::copy(fcsVec.begin(), fcsVec.end(), fourierCoeffsSin);

// 	kanGPU<float>(x,fourierCoeffsCos,fourierCoeffsSin,y);

// 	for(int i=0;i<batchSize*outputDim;i++){
// 		std::cout << y[i] << std::endl;
// 	}
// }

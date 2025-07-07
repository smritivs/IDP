#include "kan.h"
#include <cstdlib>
#include <ctime>
#include <chrono>

template <typename fp>
__global__ void kanKernel(
	fp *x,
	fp *fourierCoeffsCos,
	fp *fourierCoeffsSin,
	fp *y
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
	__syncthreads();
	fp x_element = x[idz*inputDim + idy];

	fp trigInp = x_element * (idx + 1);

	fp cosRes = 0;
	fp sinRes = 0;

	sincosf(trigInp,&sinRes,&cosRes);

	__shared__ fp cosTerms[batchSize*inputDim*numHarmonics];
	__shared__ fp sinTerms[batchSize*inputDim*numHarmonics];

	__syncthreads();
	// bounds check
	if(idy < inputDim){
		cosTerms[idz*inputDim*numHarmonics + idy*numHarmonics + idx] = cosRes;
		sinTerms[idz*inputDim*numHarmonics + idy*numHarmonics + idx] = sinRes;
	}

	fp yCosSum = 0;
	fp ySinSum = 0;
	__shared__ fp partialProdsCosTest[inputDim*numHarmonics];

	// create partial products and store in shared memory
	__shared__ fp partialProdsCos[numHarmonics];
	__shared__ fp partialProdsSin[numHarmonics];

	partialProdsCos[idx] = 0;
	partialProdsSin[idx] = 0;

	// optimize
	for(int i=0;i<inputDim;i++){
		for(int j=0;j<numHarmonics;j++){
			yCosSum += cosTerms[idz*inputDim*numHarmonics + i*numHarmonics + j] * fourierCoeffsCos[idy*inputDim*numHarmonics + i*numHarmonics + j];
			ySinSum += sinTerms[idz*inputDim*numHarmonics + i*numHarmonics + j] * fourierCoeffsSin[idy*inputDim*numHarmonics + i*numHarmonics + j];
			partialProdsCosTest[i*numHarmonics + j] = cosTerms[idz*inputDim*numHarmonics + i*numHarmonics + j] * fourierCoeffsCos[idy*inputDim*numHarmonics + i*numHarmonics + j];

		}
		partialProdsCos[idx] += cosTerms[idz*inputDim*numHarmonics + i*numHarmonics + idx] * fourierCoeffsCos[idy*inputDim*numHarmonics + i*numHarmonics + idx];
		partialProdsSin[idx] += sinTerms[idz*inputDim*numHarmonics + i*numHarmonics + idx] * fourierCoeffsSin[idy*inputDim*numHarmonics + i*numHarmonics + idx];
	}

	__syncthreads();
	// reduce partial prods array
	for(int i=numHarmonics;i>=2;i=i/2){
		if(idx < i/2){
			partialProdsCos[idx] += partialProdsCos[idx + i/2];
			partialProdsSin[idx] += partialProdsSin[idx + i/2];
		}
	}

	__syncthreads();
	y[idz*outputDim + idy] = partialProdsCos[0] + partialProdsSin[0] + bias;
	// y[idz*outputDim + idy] = yCosSum + ySinSum + bias;
}

template<typename tensor>

void kanGPU(tensor *a, tensor *b, tensor *c, tensor *res){

	int totalThreads = numHarmonics * inputDim * batchSize;

	// change to appropriate number
	dim3 gridDim(xBlocks,yBlocks,zBlocks);
	dim3 blockDim(ceil(numHarmonics/xBlocks),ceil(yThreads/yBlocks),ceil(batchSize/zBlocks));

	// allocate device and host memory
	tensor *x;
	tensor *fcc;
	tensor *fcs;
	tensor *y;
	// copy data to GPU
	cudaMalloc((void **)&x, totalThreads*sizeof(tensor));
	cudaMalloc((void **)&fcc, outputDim*inputDim*numHarmonics*sizeof(tensor));
	cudaMalloc((void **)&fcs, outputDim*inputDim*numHarmonics*sizeof(tensor));
	cudaMalloc((void **)&y, totalThreads*sizeof(tensor));
    // cudaMemset(y, 0, batchSize*outputDim*sizeof(tensor));


	cudaMemcpy(x, a, totalThreads*sizeof(tensor), cudaMemcpyHostToDevice);
	cudaMemcpy(fcc, b, outputDim*inputDim*numHarmonics*sizeof(tensor), cudaMemcpyHostToDevice);
	cudaMemcpy(fcs, c, outputDim*inputDim*numHarmonics*sizeof(tensor), cudaMemcpyHostToDevice);

	// benchmark

    kanKernel<tensor><<<gridDim,blockDim>>>(x,fcc,fcs,y);

    // copy back to CPU
	cudaMemcpy(res, y, batchSize*outputDim*sizeof(tensor), cudaMemcpyDeviceToHost);

	cudaFree(x);
	cudaFree(fcc);
	cudaFree(fcs);
	cudaFree(y);
}

int main(){
	float y[batchSize*outputDim] = {0};

	// read random test vector from file

	std::ifstream xfile("./input_x.txt");
	std::istream_iterator<float> xstart(xfile), xend;
	std::vector <float> xVec(xstart,xend);

	std::ifstream fccfile("./input_fourier_coeffs_cos.txt");
	std::istream_iterator<float> fccstart(fccfile), fccend;
	std::vector <float> fccVec(fccstart,fccend);

	std::ifstream fcsfile("./input_fourier_coeffs_sin.txt");
	std::istream_iterator<float> fcsstart(fcsfile), fcsend;
	std::vector <float> fcsVec(fcsstart,fcsend);

	float *x;
	float *fourierCoeffsCos;
	float *fourierCoeffsSin;

	x = &xVec[0];
	fourierCoeffsCos = &fccVec[0];
	fourierCoeffsSin = &fcsVec[0];

	kanGPU<float>(x,fourierCoeffsCos,fourierCoeffsSin,y);

	for(int i=0;i<batchSize*outputDim;i++){
		std::cout << y[i] << std::endl;
	}

	return 0;
}

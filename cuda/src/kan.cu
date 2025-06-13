#include "kan.h"


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
	fp x_element = x[idz*inputDim + idy*numHarmonics + 0];

	__syncthreads();

	fp trigInp = x_element * (idx + 1);

	fp cosRes = 0;
	fp sinRes = 0;

	sincosf(trigInp,&sinRes,&cosRes);

	__shared__ fp cosTerms[batchSize*inputDim*numHarmonics];
	__shared__ fp sinTerms[batchSize*inputDim*numHarmonics];

	

	cosTerms[idz*inputDim + idy*numHarmonics + idx] = cosRes;
	sinTerms[idz*inputDim + idy*numHarmonics + idx] = sinRes;

	// second bounds check for output dims

	__shared__ fp yCos[batchSize*outputDim];
	__shared__ fp ySin[batchSize*outputDim];

	fp yCosSum = 0;
	fp ySinSum = 0;

	__syncthreads();	

	// optimize
	for(int i=0;i<inputDim;i++){
		for(int j=0;j<numHarmonics;j++){
			// yCosSum += cosTerms[idz*inputDim + i*numHarmonics + j] * fourierCoeffsCos[idy*outputDim + i*numHarmonics + j];
			// ySinSum += sinTerms[idz*inputDim + i*numHarmonics + j] * fourierCoeffsSin[idy*outputDim + i*numHarmonics + j];
			yCosSum += cosTerms[idz*inputDim + i*numHarmonics + j] * fourierCoeffsCos[idy*inputDim*numHarmonics + i*numHarmonics + j];
			ySinSum += sinTerms[idz*inputDim + i*numHarmonics + j] * fourierCoeffsSin[idy*inputDim*numHarmonics + i*numHarmonics + j];


		}
	}
	// yCos[idz*batchSize + idy*outputDim] = yCosSum;
	// ySin[idz*batchSize + idy*outputDim] = ySinSum;

	yCos[idz*outputDim + idy] = yCosSum;
	ySin[idz*outputDim + idy] = ySinSum;

	y[idz*outputDim + idy] = yCos[idz*outputDim + idy] + ySin[idz*outputDim + idy] + bias;
}

template<typename tensor>

void kanGPU(tensor *a, tensor *b, tensor *c, tensor *res){

	int totalThreads = numHarmonics * inputDim * batchSize;

	// change to appropriate number
	dim3 gridDim(1,1,1);
	dim3 blockDim(numHarmonics,inputDim,batchSize);

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
    cudaMemset(y, 0, batchSize*outputDim*sizeof(tensor));


	cudaMemcpy(x, a, totalThreads*sizeof(tensor), cudaMemcpyHostToDevice);
	cudaMemcpy(fcc, b, outputDim*inputDim*numHarmonics*sizeof(tensor), cudaMemcpyHostToDevice);
	cudaMemcpy(fcs, c, outputDim*inputDim*numHarmonics*sizeof(tensor), cudaMemcpyHostToDevice);

    kanKernel<float><<<gridDim,blockDim>>>(x,fcc,fcs,y);

    // copy back to CPU
	cudaMemcpy(res, y, batchSize*outputDim*sizeof(tensor), cudaMemcpyDeviceToHost);
	
}

int main(){
	int totalThreads = numHarmonics * inputDim * batchSize;

	float x[batchSize*inputDim*numHarmonics] = {1,0,1,0,1,0,1,0};
	float fourierCoeffsCos[outputDim*inputDim*numHarmonics] = {-0.11764081567525864, 0.26520204544067383,-0.2647557258605957, -0.6589452028274536,-0.26754602789878845, -0.1423449069261551,0.5330049395561218, -0.27065348625183105,-0.5419355034828186, -0.5137631297111511,-0.056270383298397064, 0.05014132708311081,0.23562470078468323, 1.270749807357788,0.0936775878071785, -0.22838330268859863,-0.2776329219341278, -0.1495910882949829,0.3131037950515747, 0.1637701541185379,0.31852877140045166, -0.38924485445022583,-0.4049484431743622, -0.09340766817331314};
	float fourierCoeffsSin[outputDim*inputDim*numHarmonics] = {-0.7642549872398376, -0.3950747549533844,0.5043845772743225, -0.0421271026134491,0.07050096988677979, -0.34597474336624146,-0.3534367084503174, 0.14451274275779724, 0.15364940464496613, 0.22368177771568298,-0.40279123187065125, 0.4280928373336792,0.29595276713371277, -0.8042312860488892,0.5119664669036865, -0.18502190709114075, -0.31381455063819885, 0.635210394859314,0.049303196370601654, -0.07333413511514664,-0.1196710392832756, -0.3811095058917999,0.15586987137794495, 0.5923671126365662};

	float y[batchSize*outputDim] = {0};

	kanGPU<float>(x,fourierCoeffsCos,fourierCoeffsSin,y);

	for(int i=0;i<batchSize*outputDim;i++){
		std::cout << "Res: " << y[i] << std::endl;
	}
}

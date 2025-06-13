#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>

const unsigned int batchSize = 1;
const unsigned int inputDim = 4;
const unsigned int outputDim = 3;
const unsigned int numHarmonics = 2;
const int bias = 0;

const unsigned long long seed = 1234;
curandStatePhilox4_32_10_t *devPHILOXStates;

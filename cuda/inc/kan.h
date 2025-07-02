#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <iterator>
#include <algorithm>

const unsigned int batchSize = 16;
const unsigned int numHarmonics = 5;
const unsigned int inputDim = 10;
const unsigned int outputDim = 10;
const int bias = 0;

const unsigned long long seed = 1234;
curandStatePhilox4_32_10_t *devPHILOXStates;

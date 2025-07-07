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

const unsigned long long seed = 1234;
curandStatePhilox4_32_10_t *devPHILOXStates;

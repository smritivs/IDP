#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>

const unsigned long long seed = 1234;
curandStatePhilox4_32_10_t *devPHILOXStates;

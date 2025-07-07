#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define TWO_PI 6.283185307179586f

// compute sin(2π x ⋅ f) and cos(2π x ⋅ f)
__global__ void fourier_encode_kernel(
    const float* x,        // Input: [N, D]
    const float* freq,     // Frequencies: [D, M]
    float* out,            // Output: [N, 2M]
    int N, int D, int M
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // batch index
    int col = blockIdx.y * blockDim.y + threadIdx.y;  // frequency index

    if (row < N && col < M) {
        float dot = 0.0f;
        for (int i = 0; i < D; ++i) {
            dot += x[row * D + i] * freq[i * M + col];
        }

        float val = TWO_PI * dot;
        float sin_val = sinf(val);
        float cos_val = cosf(val);

        out[row * (2 * M) + col] = sin_val;
        out[row * (2 * M) + col + M] = cos_val;
    }
}

// Wrapper
extern "C" void launch_fourier(
    const float* x,
    const float* freq,
    float* out,
    int N, int D, int M
) {
    dim3 threads(8, 8);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    float *dev_x, *dev_freq, *dev_out;
    cudaMalloc(&dev_x, N * D * sizeof(float));
    cudaMalloc(&dev_freq, D * M * sizeof(float));
    cudaMalloc(&dev_out, N * 2 * M * sizeof(float));

    // data
    cudaMemcpy(dev_x, x, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_freq, freq, D * M * sizeof(float), cudaMemcpyHostToDevice);

    fourier_encode_kernel<<<blocks, threads>>>(dev_x, dev_freq, dev_out, N, D, M);

    cudaMemcpy(out, dev_out, N * 2 * M * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
}

// Remove or comment out the main() function for extension build
/*
int main() {
    const int N = 4;  // batch size
    const int D = 2;  // input dimension
    const int M = 3;  // number of frequencies

    // Sample input: 4 vectors of 2D coordinates
    float host_x[N * D] = {
        0.1f, 0.2f,
        0.5f, 0.4f,
        0.9f, 0.7f,
        1.0f, 0.8f
    };

    // Frequency matrix: shape [D, M]
    float host_freq[D * M] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    float host_out[N * 2 * M] = {0};

    // Allocate GPU memory
    float *dev_x, *dev_freq, *dev_out;
    cudaMalloc(&dev_x, N * D * sizeof(float));
    cudaMalloc(&dev_freq, D * M * sizeof(float));
    cudaMalloc(&dev_out, N * 2 * M * sizeof(float));

    // data
    cudaMemcpy(dev_x, host_x, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_freq, host_freq, D * M * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threads(8, 8);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    launch_fourier(dev_x, dev_freq, dev_out, N, D, M);

    cudaMemcpy(host_out, dev_out, N * 2 * M * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Output (sin/cos features):\n";
    for (int i = 0; i < N; ++i) {
        std::cout << "Row " << i << ": ";
        for (int j = 0; j < 2 * M; ++j) {
            std::cout << host_out[i * 2 * M + j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
*/

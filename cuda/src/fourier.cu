#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <iterator>
#include <algorithm>

constexpr float TWO_PI  = 6.283185307179586;

// compute sin(2π x ⋅ f) and cos(2π x ⋅ f)
__global__ void fourier_encode_kernel(
    const float* x,        // Input: [N, D]
    const float* freq,     // Frequencies: [D, M]
    float* out,            // Output: [N, 2M]
    const int N, const int D, const int M
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
void launch_fourier(
    float* host_x,
    float* host_freq,
    float* host_out,
    const int N, const int D, const int M
) {

    float *dev_x, *dev_freq, *dev_out;
    cudaMalloc(&dev_x, N * D * sizeof(float));
    cudaMalloc(&dev_freq, D * M * sizeof(float));
    cudaMalloc(&dev_out, N * 2 * M * sizeof(float));

    // data
    cudaMemcpy(dev_x, host_x, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_freq, host_freq, D * M * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(8, 8);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    fourier_encode_kernel<<<blocks, threads>>>(dev_x, dev_freq, dev_out, N, D, M);

    cudaMemcpy(host_out, dev_out, N * 2 * M * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_x);
    cudaFree(dev_freq);
    cudaFree(dev_out);

}

int main() {
	constexpr int N = 4;  // batch size
	constexpr int D = 2;  // input dimension
	constexpr int M = 3;  // number of frequencies

    // Sample input: 4 vectors of 2D coordinates
    float host_x[N * D] = {0};

    // Frequency matrix: shape [D, M]
    float host_freq[D * M] = {0};

    float host_out[N * 2 * M] = {0};

	std::ifstream xfile("./input_x.txt");
	std::istream_iterator<float> xstart(xfile), xend;
	std::vector<float> xVec(xstart,xend);

	std::ifstream fqsfile("./input_freqs.txt");
	std::istream_iterator<float> fqsstart(fqsfile), fqsend;
	std::vector<float> fqsVec(fqsstart,fqsend);

	std::copy(xVec.begin(), xVec.end(), host_x);
	std::copy(fqsVec.begin(), fqsVec.end(), host_freq);

    launch_fourier(host_x, host_freq, host_out, N, D, M);

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

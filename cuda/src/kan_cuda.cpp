#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Declare your CUDA kernel launcher (from your .cu)
extern "C" void kanGPU(
    float* x,
    float* fourierCoeffsCos,
    float* fourierCoeffsSin,
    float* y,
    const int batchSize, const int inputDim, const int outputDim, const int numHarmonics
);

// PyTorch binding
at::Tensor kan_forward(at::Tensor x, at::Tensor fourierCoeffsCos, at::Tensor fourierCoeffsSin) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(fourierCoeffsCos.is_cuda(), "cos coeffs must be a CUDA tensor");
    TORCH_CHECK(fourierCoeffsSin.is_cuda(), "sin coeffs must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == at::kFloat, "x must be float32");
    TORCH_CHECK(fourierCoeffsCos.dtype() == at::kFloat, "cos coeffs must be float32");
    TORCH_CHECK(fourierCoeffsCos.dtype() == at::kFloat, "sin coeffs must be float32");
    const int batchSize = x.size(0);
    const int inputDim = x.size(1);
    const int outputDim = fourierCoeffsCos.size(0);
    const int numHarmonics = fourierCoeffsCos.size(2);

    auto out = at::zeros({batchSize * outputDim}, x.options());

    kanGPU(
        x.data_ptr<float>(),
        fourierCoeffsCos.data_ptr<float>(),
        fourierCoeffsSin.data_ptr<float>(),
        out.data_ptr<float>(),
        batchSize,
        inputDim,
        outputDim,
        numHarmonics
    );
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kan_forward", &kan_forward, "KAN CUDA forward");
}

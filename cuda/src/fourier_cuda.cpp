#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Declare your CUDA kernel launcher (from your .cu)
void launch_fourier(
    const float* x,
    const float* freq,
    float* out,
    int N, int D, int M
);

// PyTorch binding
at::Tensor fourier_forward(at::Tensor x, at::Tensor freq) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(freq.is_cuda(), "freq must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == at::kFloat, "x must be float32");
    TORCH_CHECK(freq.dtype() == at::kFloat, "freq must be float32");
    int N = x.size(0);
    int D = x.size(1);
    int M = freq.size(1);

    auto out = at::zeros({N, 2 * M}, x.options());

    launch_fourier(
        x.data_ptr<float>(),
        freq.data_ptr<float>(),
        out.data_ptr<float>(),
        N, D, M
    );
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fourier_forward", &fourier_forward, "Fourier CUDA forward");
}

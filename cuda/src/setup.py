from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fourier_cuda',
    ext_modules=[
        CUDAExtension(
            'fourier_cuda',
            ['fourier_cuda.cpp', 'fourier.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

setup(
    name='kan_cuda',
    ext_modules=[
        CUDAExtension(
            'kan_cuda',
            ['kan_cuda.cpp', 'kan.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

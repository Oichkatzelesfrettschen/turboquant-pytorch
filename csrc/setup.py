"""
Build script for TurboQuant CUDA extensions.

Usage:
    cd csrc && python setup.py install
    # or for development:
    cd csrc && python setup.py develop

The CUDA extension provides fused kernels for:
    - WHT rotation + scalar quantization (single kernel launch)
    - Asymmetric attention score computation from compressed keys

Optimization patterns from steinmarder:
    - Shared memory codebook broadcast
    - 8-wide ILP FFMA accumulator chains
    - Vectorized float4 loads for coalesced access
    - Storage-compute split (INT8 storage, FP32 compute)
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="turboquant_cuda",
    ext_modules=[
        CUDAExtension(
            "turboquant_cuda",
            ["fused_quantize.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_80,code=sm_80",  # Ampere
                    "-gencode=arch=compute_86,code=sm_86",  # GA102
                    "-gencode=arch=compute_89,code=sm_89",  # Ada Lovelace
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)

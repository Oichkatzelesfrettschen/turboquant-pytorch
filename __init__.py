# Core TurboQuant pipeline
from .turboquant import TurboQuantMSE, TurboQuantProd, TurboQuantKVCache
from .lloyd_max import LloydMaxCodebook, solve_lloyd_max
from .compressors import TurboQuantCompressorV2, TurboQuantCompressorMSE

# Cayley-Dickson algebra engine
from .cd_algebra import (
    cd_conjugate, cd_multiply, cd_norm_sq, cd_norm, cd_normalize,
    cd_inverse, cd_associator, cd_associator_norm, cd_random_unit,
)

# Rotation methods
from .rotations import (
    Rotation, HaarRotation, WHTRotation, CDRotation,
    CDMultiLayerRotation, PCARotation, KacRotation,
)

# Lattice vector quantization
from .lattice_vq import (
    VectorQuantizer, ScalarLloydMaxQuantizer,
    E8LatticeQuantizer, Z8PrefixCutQuantizer,
)
from .e8_quantizer import e8_closest_point, generate_e8_roots
from .lattice_codebook import get_codebook, codebook_sizes

# Adaptive precision
from .adaptive import AdaptiveBitAllocator, QuantConfig

# Tensor decomposition
from .tensor_decomposition import svd_compress, joint_rank_bitwidth

# Spectral analysis
from .spectral import distribution_analysis, spectral_bit_allocation, rotation_quality_score

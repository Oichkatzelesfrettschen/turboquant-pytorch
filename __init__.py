# When pytest's internal machinery tries to import __init__.py directly
# (before conftest.py runs), relative imports fail because there is no parent
# package. Guard all imports so this file loads cleanly in that scenario.
# conftest.py then bootstraps the real package via importlib.util.
try:
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

    # CD fidelity metric (phase-geometry preservation)
    from .cd_fidelity import (
        cd_fidelity_ratio, sliding_cd_fidelity, fidelity_summary,
        residual_associator_per_token, distortion_decomposition,
    )

    # Zero-divisor affinity scoring
    from .zd_bias import sedenion_zd_affinity, batch_zd_affinity, zd_quartile_analysis

    # Hierarchical CD tower quantization
    from .hierarchical import (
        tower_levels, allocate_bits_to_levels,
        hierarchical_quantize, compare_hierarchical_vs_uniform,
    )

    # Sign packing (8x QJL memory reduction)
    from .sign_pack import pack_signs, unpack_signs, packed_inner_product

    # 8D K-Means VQ codebook (NSNQuant-style, calibration-free)
    from .kmeans_vq import KMeans8DQuantizer

    # E8-block rotation (136x parameter reduction, KS-validated)
    from .e8_rotation import E8BlockRotation

    # Clifford rotor rotation (RotorQuant synthesis)
    from .clifford_rotor import CliffordRotorRotation

    # NSN pre-processing + KIVI asymmetry + group-wise quantization + precision windows
    from .nsn_preprocess import (
        nsn_preprocess, nsn_restore, adaptive_vq_scale,
        kivi_quantize_keys, kivi_dequantize_keys,
        kivi_quantize_values, kivi_dequantize_values,
        PrecisionWindows,
    )

    # Cariow factorization analysis
    from .cariow import MultCountRecord, mult_count_table, theoretical_speedup_vs_dimension

    # Unified algebraic foundations
    from .algebra_foundations import (
        DIVISION_ALGEBRAS, CD_TOWER, EXCEPTIONAL_ALGEBRAS,
        barnes_wall_connection, enumerate_methods,
    )

    # Hybrid synthesis pipeline (NSN + WHT global + CD block + sign packing)
    from .hybrid_pipeline import (
        HybridWHTCDRotation, AdaptivePerHeadRotation,
        hybrid_compress, hybrid_attention_scores,
    )

    # cuBLAS-accelerated WHT rotation + Triton quantization kernels
    from .triton_kernels import CuBLASWHTRotation

    # CPU/GPU dispatch with architecture detection
    from .gpu_dispatch import detect_gpu, apply_gpu_optimizations, GPUProfile, KernelTier
    from .cpu_dispatch import detect_cpu, CPUProfile, CPUQuantizeDispatch

    # Smart configuration presets
    from .config import TurboQuantConfig, KVCacheConfig

except ImportError:
    # Loaded outside package context (pytest direct import). Skip.
    pass

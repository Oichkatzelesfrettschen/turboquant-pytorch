"""
Zero-divisor affinity scoring for quantization bias detection.

Moreno (1997, arXiv q-alg/9710013) proved that an element a in the sedenion
algebra is a zero-divisor iff the left-multiplication operator L_a has
eigenvalue -2. Reggiani (2024) showed the ZD manifold has topology G_2 x S^1.

For TurboQuant: QJL residual vectors aligned with zero-divisor manifolds
have poor inner-product behavior. Quantization error along these directions
is amplified because the CD multiplication structure provides no
error-correcting redundancy.

Lower affinity = closer to ZD manifold = more quantization-vulnerable.

Ported from open_gororoba/crates/cd_kernel/src/turboquant/zd_bias.rs.
"""

import torch
from torch import Tensor

from .cd_algebra import cd_multiply, cd_normalize


def sedenion_zd_affinity(
    v: Tensor,
    n_samples: int = 200,
    seed: int = 42,
) -> Tensor:
    """
    Compute zero-divisor affinity of vectors in sedenion (16D) space.

    The affinity is the minimum norm of a*b over sampled unit vectors b.
    True zero-divisors have affinity = 0. Vectors far from the ZD manifold
    have high affinity.

    Args:
        v: vectors, shape (..., 16)
        n_samples: number of random unit vectors to test
        seed: random seed

    Returns:
        Affinity scores, shape (...). Lower = more ZD-vulnerable.
    """
    assert v.shape[-1] == 16, f"ZD affinity requires 16D, got {v.shape[-1]}"

    v_unit = cd_normalize(v)
    batch_shape = v.shape[:-1]

    gen = torch.Generator(device=v.device)
    gen.manual_seed(seed)

    min_norms = torch.full(batch_shape, float("inf"), device=v.device)

    for _ in range(n_samples):
        b = torch.randn(*batch_shape, 16, device=v.device, generator=gen)
        b = cd_normalize(b)
        product = cd_multiply(v_unit, b)
        prod_norm = product.norm(dim=-1)
        min_norms = torch.minimum(min_norms, prod_norm)

    return min_norms


def batch_zd_affinity(
    residuals: Tensor,
    n_samples: int = 200,
    seed: int = 42,
) -> Tensor:
    """
    Compute ZD-affinity scores for a batch of vectors.

    For vectors with dimension > 16, projects into sedenion (16D) subspace
    by taking the first 16 components. For dim < 16, pads with zeros.

    Args:
        residuals: shape (n, d) -- QJL residual vectors
        n_samples: number of samples for affinity estimation
        seed: random seed

    Returns:
        Affinity scores, shape (n,). Lower = more ZD-vulnerable.
    """
    n, d = residuals.shape

    if d >= 16:
        sedenion_part = residuals[:, :16]
    else:
        sedenion_part = torch.zeros(n, 16, device=residuals.device)
        sedenion_part[:, :d] = residuals

    return sedenion_zd_affinity(sedenion_part, n_samples, seed)


def zd_quartile_analysis(
    residuals: Tensor,
    quantization_errors: Tensor,
    n_samples: int = 200,
) -> Tensor:
    """
    Bin residuals by ZD-affinity quartile and compute per-quartile MSE.

    Tests the hypothesis: low ZD-affinity residuals have higher quantization
    error (the ZD manifold amplifies quantization noise).

    Args:
        residuals: shape (n, d) -- QJL residual vectors
        quantization_errors: shape (n,) -- per-vector MSE
        n_samples: samples for affinity estimation

    Returns:
        Per-quartile MSE, shape (4,). Index 0 = lowest affinity (most vulnerable).
    """
    affinities = batch_zd_affinity(residuals, n_samples)
    n = affinities.shape[0]

    sorted_indices = affinities.argsort()
    q_size = n // 4
    quartile_mse = torch.zeros(4, device=residuals.device)

    for q in range(4):
        start = q * q_size
        end = n if q == 3 else start + q_size
        indices = sorted_indices[start:end]
        quartile_mse[q] = quantization_errors[indices].mean()

    return quartile_mse

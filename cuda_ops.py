"""
CUDA kernel wrappers with automatic fallback to pure PyTorch.

Four kernels ported from open_gororoba + steinmarder:
    1. turboquant_quantize_boundary: Lloyd-Max boundary-search quantization
    2. turboquant_dequant_dot: fused dequant + dot product (attention hot path)
    3. turboquant_sign_dot: QJL sign-sketch inner product with bit-packed signs
    4. turboquant_fast_jl_rotate: in-place Walsh-Hadamard Transform

JIT-compiles on first use. Falls back to pure PyTorch if CUDA unavailable.

Optimization patterns:
    - Storage-compute split: INT8 indices in memory, FP32 for arithmetic
    - SoA layout: key_indices[coord * n_keys + key_idx] for coalesced reads
    - Codebook in L1: max 16 entries @ 4-bit = 64 bytes (fits in register file)
    - POPC for bit-packed sign inner products (7-8 cyc on Ada)
"""

import math
import os

import torch
from torch import Tensor
from typing import Optional

_cuda_module = None
_cuda_available = False


def _try_load_cuda():
    """Attempt to JIT-compile and load the CUDA extension."""
    global _cuda_module, _cuda_available

    if _cuda_module is not None:
        return _cuda_available

    if not torch.cuda.is_available():
        _cuda_available = False
        return False

    try:
        from torch.utils.cpp_extension import load
        csrc_dir = os.path.join(os.path.dirname(__file__), "csrc")
        _cuda_module = load(
            name="turboquant_cuda",
            sources=[os.path.join(csrc_dir, "fused_quantize.cu")],
            verbose=False,
        )
        _cuda_available = True
    except Exception:
        _cuda_available = False

    return _cuda_available


# ---------------------------------------------------------------------------
# Kernel 1: Boundary-search quantization
# ---------------------------------------------------------------------------

def quantize_boundary(
    values: Tensor,
    boundaries: Tensor,
    *,
    use_cuda: Optional[bool] = None,
) -> Tensor:
    """
    Quantize f32 values using sorted Lloyd-Max boundaries.

    Index = count(boundaries[j] < value). For 3-bit: 7 comparisons.

    Args:
        values: input f32 tensor, any shape
        boundaries: sorted boundary tensor, shape (2^bits - 1,)
        use_cuda: force CUDA/CPU/auto

    Returns:
        Quantization indices, same shape as values, dtype uint8.
    """
    if use_cuda is None:
        use_cuda = values.is_cuda and _try_load_cuda()

    if use_cuda and _cuda_module is not None:
        return _cuda_module.turboquant_quantize_boundary(boundaries, values)

    # Pure PyTorch fallback
    count = (values.unsqueeze(-1) > boundaries).sum(dim=-1)
    return count.to(torch.uint8)


# ---------------------------------------------------------------------------
# Kernel 2: Fused dequant + dot product
# ---------------------------------------------------------------------------

def dequant_dot(
    queries: Tensor,
    key_indices: Tensor,
    centroids: Tensor,
    key_norms: Tensor,
    *,
    use_cuda: Optional[bool] = None,
) -> Tensor:
    """
    Fused dequant + dot product for attention scoring.

    Computes scores[q,k] = key_norms[k] * sum_c(queries[q,c] * centroids[key_indices[c,k]])

    SoA layout: key_indices[coord, key_idx] for coalesced CUDA reads.

    Args:
        queries: (n_queries, d) float
        key_indices: (d, n_keys) uint8 in SoA layout
        centroids: (n_levels,) float codebook
        key_norms: (n_keys,) float per-key scaling

    Returns:
        Attention scores, (n_queries, n_keys) float.
    """
    if use_cuda is None:
        use_cuda = queries.is_cuda and _try_load_cuda()

    if use_cuda and _cuda_module is not None:
        return _cuda_module.turboquant_dequant_dot(
            queries, key_indices, centroids, key_norms
        )

    # Pure PyTorch fallback: dequantize then matmul
    keys_recon = centroids[key_indices.long()]  # (d, n_keys)
    scores = queries @ keys_recon.float()  # (n_queries, n_keys)
    return scores * key_norms.unsqueeze(0)


# ---------------------------------------------------------------------------
# Kernel 3: QJL sign-sketch inner product
# ---------------------------------------------------------------------------

def sign_dot(
    s_matrix: Tensor,
    query: Tensor,
    packed_signs: Tensor,
    residual_norms: Tensor,
    correction_scale: float,
    *,
    use_cuda: Optional[bool] = None,
) -> Tensor:
    """
    QJL correction: ||r|| * sqrt(pi/2)/m * <S@q, signs>.

    Signs are bit-packed as int32 words (32 signs per word).

    Args:
        s_matrix: (m, d) QJL projection matrix
        query: (d,) single query vector
        packed_signs: (n_keys, n_words) int32 packed sign bits
        residual_norms: (n_keys,) float
        correction_scale: sqrt(pi/2) / m

    Returns:
        QJL correction terms, (n_keys,) float.
    """
    if use_cuda is None:
        use_cuda = query.is_cuda and _try_load_cuda()

    if use_cuda and _cuda_module is not None:
        return _cuda_module.turboquant_sign_dot(
            s_matrix, query, packed_signs, residual_norms, correction_scale
        )

    # Pure PyTorch fallback
    projected = s_matrix @ query  # (m,)
    # Unpack signs from int32 words
    m = s_matrix.shape[0]
    n_keys = packed_signs.shape[0]
    signs = torch.zeros(n_keys, m, device=query.device)
    for w in range(packed_signs.shape[1]):
        for b in range(32):
            j = w * 32 + b
            if j >= m:
                break
            bit = (packed_signs[:, w] >> b) & 1
            signs[:, j] = bit.float() * 2 - 1

    qjl_ip = signs @ projected  # (n_keys,)
    return residual_norms * correction_scale * qjl_ip


# ---------------------------------------------------------------------------
# Kernel 4: Fast Walsh-Hadamard Transform
# ---------------------------------------------------------------------------

def fast_jl_rotate(
    data: Tensor,
    d1: Tensor,
    d2: Tensor,
    *,
    use_cuda: Optional[bool] = None,
    inplace: bool = False,
) -> Tensor:
    """
    Fast JL rotation: y = D1 * WHT * D2 * x (in-place on CUDA).

    Args:
        data: (n_vectors, d) float
        d1: (d,) Rademacher diagonal 1
        d2: (d,) Rademacher diagonal 2
        use_cuda: force CUDA/CPU/auto
        inplace: if True, modify data in-place (CUDA only)

    Returns:
        Rotated data, same shape.
    """
    if use_cuda is None:
        use_cuda = data.is_cuda and _try_load_cuda()

    if use_cuda and _cuda_module is not None:
        if not inplace:
            data = data.clone()
        _cuda_module.turboquant_fast_jl_rotate(data, d1, d2)
        return data

    # Pure PyTorch fallback
    from .rotations import _fast_hadamard
    y = data * d2
    y = _fast_hadamard(y)
    y = y * d1
    return y

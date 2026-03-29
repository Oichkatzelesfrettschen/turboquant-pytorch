"""
CUDA kernel wrappers with automatic fallback to pure PyTorch.

Attempts to JIT-compile the CUDA kernels in csrc/ on first use.
If CUDA is unavailable or compilation fails, all functions transparently
fall back to the equivalent pure PyTorch implementations.

Optimization patterns from steinmarder applied:
    - Storage-compute split: INT8 indices in memory, FP32 for arithmetic
    - Shared memory codebook: broadcast to all threads in a warp
    - 8-wide ILP: independent FFMA accumulator chains
    - Vectorized loads: float4 coalesced access
"""

import os
import math
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


def fused_wht_quantize(
    x: Tensor,
    d1: Tensor,
    d2: Tensor,
    centroids: Tensor,
    *,
    use_cuda: Optional[bool] = None,
) -> Tensor:
    """
    Fused WHT rotation + scalar quantization.

    Applies D1 @ H_d @ D2 @ x, then quantizes each coordinate to the
    nearest centroid. Fuses three kernel launches into one.

    Args:
        x: input vectors, shape (n, d)
        d1: first sign vector, shape (d,)
        d2: second sign vector, shape (d,)
        centroids: Lloyd-Max centroids, shape (n_centroids,)
        use_cuda: force CUDA (True), force CPU (False), or auto-detect (None)

    Returns:
        Quantization indices, shape (n, d), int8.
    """
    if use_cuda is None:
        use_cuda = x.is_cuda and _try_load_cuda()

    if use_cuda and _cuda_module is not None:
        return _cuda_module.fused_wht_quantize(x, d1, d2, centroids)

    # Pure PyTorch fallback
    from .rotations import _fast_hadamard
    y = x * d2
    y = _fast_hadamard(y)
    y = y * d1
    diffs = y.unsqueeze(-1) - centroids
    indices = diffs.abs().argmin(dim=-1).to(torch.int8)
    return indices


def fused_asymmetric_attention(
    queries: Tensor,
    k_mse: Tensor,
    qjl_signs: Tensor,
    r_norms: Tensor,
    S: Tensor,
    correction_scale: float,
    *,
    use_cuda: Optional[bool] = None,
) -> Tensor:
    """
    Fused asymmetric attention score computation.

    scores[q,k] = <Q[q], K_mse[k]> + ||r[k]|| * C * <S@Q[q], signs[k]>

    Args:
        queries: query vectors, shape (n_q, d)
        k_mse: MSE-reconstructed keys, shape (n_k, d)
        qjl_signs: QJL sign bits, shape (n_k, d), int8
        r_norms: residual norms, shape (n_k,)
        S: QJL projection matrix, shape (d, d)
        correction_scale: sqrt(pi/2) / m
        use_cuda: force CUDA/CPU/auto

    Returns:
        Attention scores, shape (n_q, n_k).
    """
    if use_cuda is None:
        use_cuda = queries.is_cuda and _try_load_cuda()

    if use_cuda and _cuda_module is not None:
        return _cuda_module.fused_asymmetric_attention(
            queries, k_mse, qjl_signs, r_norms, S, correction_scale
        )

    # Pure PyTorch fallback
    term1 = queries @ k_mse.T
    q_projected = queries @ S.T
    signs_float = qjl_signs.float()
    qjl_ip = q_projected @ signs_float.T
    term2 = correction_scale * qjl_ip * r_norms.unsqueeze(0)
    return term1 + term2

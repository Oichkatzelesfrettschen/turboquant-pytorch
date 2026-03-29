"""
CD fidelity metric: phase-geometry preservation through quantization.

The Cayley-Dickson associator ||[a,b,c]|| measures three-way phase coupling
in non-associative algebra. When computed on direction-normalized vectors
(unit vectors), it isolates angular/phase relationships from magnitude.

Key insight: the 3.1% cosine distortion from TurboQuant decomposes into:
    ~3.08% magnitude distortion (norm changes)
    ~0.02% phase-geometry distortion (angular relationships)

Attention weights depend primarily on angular relationships, explaining
why TurboQuant achieves 99.5% attention fidelity despite mediocre MSE.

The CD fidelity ratio = A_post / A_pre measures how well quantization
preserves this phase-geometry structure.

Novel applications:
    1. Per-token diagnostic: high residual associator identifies tokens
       where sign projections capture structure poorly -> adaptive bit allocation
    2. Distortion decomposition: separate magnitude vs phase-geometry error

Ported from open_gororoba/crates/cd_kernel/src/turboquant/cd_fidelity.rs.
"""

import torch
from torch import Tensor
from typing import Dict, Tuple
from dataclasses import dataclass

from .cd_algebra import cd_associator_norm, cd_normalize


def cd_fidelity_ratio(
    a: Tensor, b: Tensor, c: Tensor,
    a_q: Tensor, b_q: Tensor, c_q: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute CD fidelity ratio for quantized triplets.

    Both original (a,b,c) and quantized (a_q, b_q, c_q) are direction-normalized
    before computing the associator, isolating phase-geometry from magnitude.

    Args:
        a, b, c: original vectors, shape (..., 2^k)
        a_q, b_q, c_q: quantized vectors, same shape

    Returns:
        (ratio, a_pre, a_post) where:
            ratio: fidelity ratio A_post / A_pre (1.0 = perfect preservation)
            a_pre: pre-quantization associator norm
            a_post: post-quantization associator norm
    """
    norm_a = cd_normalize(a)
    norm_b = cd_normalize(b)
    norm_c = cd_normalize(c)
    norm_aq = cd_normalize(a_q)
    norm_bq = cd_normalize(b_q)
    norm_cq = cd_normalize(c_q)

    a_pre = cd_associator_norm(norm_a, norm_b, norm_c)
    a_post = cd_associator_norm(norm_aq, norm_bq, norm_cq)

    ratio = torch.where(
        a_pre.abs() < 1e-15,
        torch.ones_like(a_pre),
        a_post / a_pre,
    )
    return ratio, a_pre, a_post


def sliding_cd_fidelity(
    original: Tensor,
    quantized: Tensor,
) -> Tensor:
    """
    Compute CD fidelity across a sequence using sliding window of 3.

    For each consecutive triplet (v[i], v[i+1], v[i+2]), computes the
    fidelity ratio between original and quantized sequences.

    Args:
        original: shape (n, d) -- original vectors
        quantized: shape (n, d) -- quantized vectors

    Returns:
        Fidelity ratios, shape (n-2,).
    """
    n = original.shape[0]
    if n < 3:
        return torch.tensor([])

    ratios, _, _ = cd_fidelity_ratio(
        original[:-2], original[1:-1], original[2:],
        quantized[:-2], quantized[1:-1], quantized[2:],
    )
    return ratios


@dataclass
class FidelitySummary:
    """Summary statistics for CD fidelity across a sequence."""
    mean_ratio: float
    min_ratio: float
    max_ratio: float
    std_ratio: float
    mean_a_pre: float
    mean_a_post: float
    n_triplets: int


def fidelity_summary(
    original: Tensor,
    quantized: Tensor,
) -> FidelitySummary:
    """Compute summary statistics for CD fidelity."""
    n = original.shape[0]
    if n < 3:
        return FidelitySummary(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0)

    ratios, a_pres, a_posts = cd_fidelity_ratio(
        original[:-2], original[1:-1], original[2:],
        quantized[:-2], quantized[1:-1], quantized[2:],
    )

    return FidelitySummary(
        mean_ratio=ratios.mean().item(),
        min_ratio=ratios.min().item(),
        max_ratio=ratios.max().item(),
        std_ratio=ratios.std().item(),
        mean_a_pre=a_pres.mean().item(),
        mean_a_post=a_posts.mean().item(),
        n_triplets=len(ratios),
    )


def residual_associator_per_token(
    residuals: Tensor,
) -> Tensor:
    """
    Per-token residual associator scores for adaptive bit allocation.

    For each token t, computes the average associator norm from triplets
    containing that token: (r[t], r[t+1], r[t+2]) in CD algebra.

    Tokens with high scores have phase-coupling structure that sign
    projections capture poorly -> these need more bits.

    Args:
        residuals: shape (n, d) -- QJL residual vectors

    Returns:
        Per-token scores, shape (n,). Higher = needs more bits.
    """
    n = residuals.shape[0]
    if n < 3:
        return torch.zeros(n, device=residuals.device)

    # Compute associator norms for all consecutive triplets
    triplet_norms = cd_associator_norm(
        residuals[:-2], residuals[1:-1], residuals[2:]
    )  # shape (n-2,)

    # Each interior token participates in up to 3 triplets
    scores = torch.zeros(n, device=residuals.device)
    scores[:-2] += triplet_norms
    scores[1:-1] += triplet_norms
    scores[2:] += triplet_norms

    # Normalize by participation count
    counts = torch.ones(n, device=residuals.device) * 3.0
    counts[0] = 1.0
    counts[1] = 2.0
    counts[-1] = 1.0
    counts[-2] = 2.0
    scores /= counts

    return scores


def distortion_decomposition(
    original: Tensor,
    quantized: Tensor,
) -> Dict[str, Tensor]:
    """
    Decompose quantization distortion into magnitude and phase-geometry.

    For each vector pair:
        total = 1 - cosine_similarity(original, quantized)
        magnitude = ((||original|| - ||quantized||) / ||original||)^2
        phase = 1 - cos(angle between unit directions)

    Args:
        original: shape (n, d)
        quantized: shape (n, d)

    Returns:
        Dict with "total", "magnitude", "phase" tensors, each shape (n,).
    """
    norm_o = original.norm(dim=-1)
    norm_q = quantized.norm(dim=-1)
    dot = (original * quantized).sum(dim=-1)

    cos_sim = dot / (norm_o * norm_q + 1e-15)
    total = 1.0 - cos_sim

    magnitude = ((norm_o - norm_q) / (norm_o + 1e-15)) ** 2

    dir_o = original / (norm_o.unsqueeze(-1) + 1e-15)
    dir_q = quantized / (norm_q.unsqueeze(-1) + 1e-15)
    dir_dot = (dir_o * dir_q).sum(dim=-1).clamp(-1.0, 1.0)
    phase = 1.0 - dir_dot

    return {"total": total, "magnitude": magnitude, "phase": phase}

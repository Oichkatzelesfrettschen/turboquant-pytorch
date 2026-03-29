"""
Hierarchical CD tower quantization: per-level bit allocation.

The CD tower provides a natural multi-scale decomposition of d-dimensional
vectors matching the Cayley-Dickson construction:

    Level 0: coords [0..4)    quaternion core      (fully associative, most robust)
    Level 1: coords [4..8)    octonion extension   (alternative, robust)
    Level 2: coords [8..16)   sedenion extension   (84+ ZD pairs, moderately vulnerable)
    Level 3: coords [16..32)  pathion extension     (proliferating ZDs, most vulnerable)
    Level 4: coords [32..64)  chingon extension
    Level 5: coords [64..128) routon extension      (most fragile)

MERA-CD duality (Novel Connection N1):
    The CD tower doubling (4D -> 128D) is the algebraic dual of MERA
    coarse-graining (128D -> 4D). Each level adds a "scale" of phase
    structure. Quantization is truncation in the tower sense: coarser
    levels are robust (need fewer bits), finer levels are fragile (need
    more bits).

Algorithm:
    1. Decompose d-dimensional vector into tower-level components
    2. Allocate bits per level based on fragility heuristic
    3. Quantize each level independently at its assigned bit-width
    4. Reconstruct by concatenating quantized components

The decomposition is lossless: concatenation of components = original vector.

Ported from open_gororoba/crates/cd_kernel/src/turboquant/hierarchical.rs.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .lloyd_max import LloydMaxCodebook
from .rotations import Rotation, WHTRotation, HaarRotation


@dataclass
class TowerLevel:
    """Specification for one level of the CD tower."""
    dim: int      # number of coordinates at this level
    bits: int     # quantization bit-width
    start: int    # start index in full vector
    end: int      # end index (exclusive)


def tower_levels(d: int) -> List[TowerLevel]:
    """
    Decompose a d-dimensional vector into CD tower levels.

    The CD doubling construction builds d = 2^k as nested doublings.
    At each level, the new coordinates are the "detail" coefficients
    from the doubling -- analogous to wavelet detail coefficients.

    Args:
        d: vector dimension (must be power of 2, >= 4)

    Returns:
        List of TowerLevel, from most robust (quaternion core) to most fragile.
    """
    assert d >= 4 and (d & (d - 1)) == 0, f"d must be power of 2 >= 4, got {d}"
    n_levels = int(math.log2(d)) - 1  # 128 -> 6 levels

    levels = []
    # Level 0: quaternion core
    levels.append(TowerLevel(dim=4, bits=0, start=0, end=4))

    start = 4
    level_dim = 4
    for _ in range(1, n_levels):
        end = min(start + level_dim, d)
        levels.append(TowerLevel(dim=end - start, bits=0, start=start, end=end))
        start = end
        level_dim *= 2

    if start < d:
        levels.append(TowerLevel(dim=d - start, bits=0, start=start, end=d))

    return levels


def allocate_bits_to_levels(
    levels: List[TowerLevel],
    budget_bits: int,
    min_bits: int = 2,
    max_bits: int = 5,
) -> List[TowerLevel]:
    """
    Assign bits per tower level to minimize total MSE at a fixed budget.

    Greedy approach: start all levels at min_bits, then promote the level
    with highest fragility score until the budget is exhausted.

    Higher tower levels (larger start index) are more fragile due to
    proliferating zero divisors and weaker algebraic structure.

    Args:
        levels: from tower_levels()
        budget_bits: total bits for the full vector (e.g., 3 * 128 = 384)
        min_bits: minimum bits per level
        max_bits: maximum bits per level

    Returns:
        New list of TowerLevel with bits assigned.
    """
    result = [TowerLevel(l.dim, min_bits, l.start, l.end) for l in levels]

    def current_total():
        return sum(l.dim * l.bits for l in result)

    while current_total() < budget_bits:
        best_idx = None
        best_score = -1.0

        for i, level in enumerate(result):
            if level.bits >= max_bits:
                continue
            if current_total() + level.dim > budget_bits:
                continue
            # Fragility heuristic: higher tower levels are more fragile
            fragility = math.log(level.start + 1.0)
            score = fragility * level.dim
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            break
        result[best_idx].bits += 1

    return result


def hierarchical_quantize(
    x: Tensor,
    levels: List[TowerLevel],
    seed: int = 42,
    rotation: str = "wht",
) -> Tuple[List[dict], Tensor]:
    """
    Quantize a batch of vectors using hierarchical tower decomposition.

    Each tower level is quantized independently at its assigned bit-width
    using its own rotation + Lloyd-Max quantizer.

    Args:
        x: input vectors, shape (n, d)
        levels: from allocate_bits_to_levels()
        seed: base random seed (each level gets seed + level_index)
        rotation: "wht" or "haar" for per-level rotation

    Returns:
        (per_level_states, reconstructed) where:
            per_level_states: list of quantization state dicts, one per level
            reconstructed: shape (n, d), the reconstructed vectors
    """
    n, d = x.shape
    reconstructed = torch.zeros_like(x)
    per_level_states = []

    for li, level in enumerate(levels):
        if level.bits == 0:
            per_level_states.append(None)
            continue

        component = x[:, level.start:level.end]  # (n, level.dim)

        # Create per-level rotation and codebook
        level_seed = seed + li * 1000
        if level.dim >= 4 and (level.dim & (level.dim - 1)) == 0:
            if rotation == "wht":
                rot = WHTRotation(level.dim, seed=level_seed, device=x.device)
            else:
                rot = HaarRotation(level.dim, seed=level_seed, device=x.device)
        else:
            rot = HaarRotation(level.dim, seed=level_seed, device=x.device)

        # Rotate
        rotated = rot.rotate(component)

        # Quantize with Lloyd-Max
        codebook = LloydMaxCodebook(level.dim, level.bits)
        centroids = codebook.centroids.to(x.device)
        diffs = rotated.unsqueeze(-1) - centroids
        indices = diffs.abs().argmin(dim=-1)

        # Dequantize
        recon_rotated = centroids[indices]
        recon = rot.unrotate(recon_rotated)
        reconstructed[:, level.start:level.end] = recon

        per_level_states.append({
            "indices": indices,
            "level": li,
            "bits": level.bits,
            "start": level.start,
            "end": level.end,
        })

    return per_level_states, reconstructed


def compare_hierarchical_vs_uniform(
    x: Tensor,
    uniform_bits: int = 3,
    min_level_bits: int = 2,
    max_level_bits: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Compare hierarchical vs uniform quantization at the same total budget.

    Args:
        x: input vectors, shape (n, d)
        uniform_bits: bits for uniform quantization
        min_level_bits: minimum bits per tower level
        max_level_bits: maximum bits per tower level
        seed: random seed

    Returns:
        Dict with hierarchical_mse, uniform_mse, improvement_pct, and per_level_mse.
    """
    n, d = x.shape
    budget = d * uniform_bits

    # Hierarchical
    levels = tower_levels(d)
    levels = allocate_bits_to_levels(levels, budget, min_level_bits, max_level_bits)
    _, recon_hier = hierarchical_quantize(x, levels, seed=seed)
    hier_mse = ((x - recon_hier) ** 2).mean().item()

    # Per-level MSE
    per_level_mse = []
    for level in levels:
        comp_orig = x[:, level.start:level.end]
        comp_recon = recon_hier[:, level.start:level.end]
        level_mse = ((comp_orig - comp_recon) ** 2).mean().item()
        per_level_mse.append(level_mse)

    # Uniform
    from .turboquant import TurboQuantMSE
    tq_uniform = TurboQuantMSE(d, uniform_bits, seed=seed, rotation="wht")
    recon_uniform, _ = tq_uniform(x)
    uniform_mse = ((x - recon_uniform) ** 2).mean().item()

    improvement = (uniform_mse - hier_mse) / (uniform_mse + 1e-15) * 100

    return {
        "hierarchical_mse": hier_mse,
        "uniform_mse": uniform_mse,
        "improvement_pct": improvement,
        "per_level_mse": per_level_mse,
        "level_bits": [l.bits for l in levels],
        "level_dims": [l.dim for l in levels],
    }

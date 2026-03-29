"""
Prefix-cut lattice codebook enumerator for CD tower filtration.

The CD tower has nested codebooks Lambda_N in Z^8 with trinary
coordinates {-1, 0, +1}. The codebook sizes 2048, 1024, 512, 256, 32
are determined by successive prefix-cut (trie-like) exclusion rules.

The codebook structure is NOT a ball, lattice shell, or symmetry orbit.
It is a trie-like decision cut: points are excluded by matching specific
early-coordinate patterns (prefixes). This is consistent with a hierarchical
encoding/generation procedure, not a geometric one.

Base universe S_base = { l in {-1,0,1}^8 : l[0] != +1,
                         sum(l_i) == 0 mod 2,
                         #{i : l_i != 0} == 0 mod 2 }
|S_base| = 2187 = 3^7

Filtration hierarchy:
    S_base (2187) -> Lambda_2048 -> Lambda_1024 -> Lambda_512 -> Lambda_256 -> Lambda_32

Each level removes points by matching forbidden prefixes.
Bit rates: Lambda_2048 = 1.375 bits/dim, Lambda_32 = 0.625 bits/dim.

Ported from open_gororoba/crates/cd_kernel/src/lattice_codebook.rs.
"""

import torch
from torch import Tensor
from typing import Dict, Optional
import itertools


def _in_base_universe(l: tuple) -> bool:
    """Check if a point is in S_base."""
    if l[0] == 1:
        return False
    if sum(l) % 2 != 0:
        return False
    nonzero_count = sum(1 for x in l if x != 0)
    return nonzero_count % 2 == 0


def _in_lambda_2048(l: tuple) -> bool:
    """Lambda_2048 = S_base minus forbidden prefixes."""
    if not _in_base_universe(l):
        return False
    if l[0] == 0 and l[1] == 1 and l[2] == 1:
        return False
    if l[0] == 0 and l[1] == 1 and l[2] == 0 and l[3] == 1 and l[4] == 1:
        return False
    if l[0] == 0 and l[1] == 1 and l[2] == 0 and l[3] == 1 and l[4] == 0 and l[5] == 1:
        return False
    return True


def _in_lambda_1024(l: tuple) -> bool:
    """Lambda_1024 = Lambda_2048 intersect {l[0]=-1} minus additional exclusions."""
    if not _in_lambda_2048(l):
        return False
    if l[0] != -1:
        return False
    if l[1] == 1 and l[2] == 1 and l[3] == 1:
        return False
    if l[1] == 1 and l[2] == 1 and l[3] == 0 and l[4] == 0:
        return False
    if l[1] == 1 and l[2] == 1 and l[3] == 0 and l[4] == 1:
        return False
    return True


def _in_lambda_512(l: tuple) -> bool:
    """Lambda_512 = Lambda_1024 minus additional exclusions."""
    if not _in_lambda_1024(l):
        return False
    if l[1] == 1:
        return False
    if l[1] == 0 and l[2] == 1:
        return False
    if l[1] == 0 and l[2] == 0 and l[3] == 0:
        return False
    if l[1] == 0 and l[2] == 0 and l[3] == 1:
        return False
    if l[1] == 0 and l[2] == 0 and l[3] == -1 and l[4] == 1:
        return False
    if l[1] == 0 and l[2] == 0 and l[3] == -1 and l[4] == 0 and l[5] == 1 and l[6] == 1:
        return False
    return True


def _in_lambda_256(l: tuple) -> bool:
    """Lambda_256 = Lambda_512 minus additional exclusions."""
    if not _in_lambda_512(l):
        return False
    if l[0] == -1 and l[1] == 0:
        return False
    if l[0] == -1 and l[1] == -1 and l[2] == 1 and l[3] == 1:
        return False
    if l[0] == -1 and l[1] == -1 and l[2] == 1 and l[3] == 0:
        return False
    if l[0] == -1 and l[1] == -1 and l[2] == 1 and l[3] == -1 and l[4] == 1:
        return False
    if l[0] == -1 and l[1] == -1 and l[2] == 1 and l[3] == -1 and l[4] == 0:
        return False
    if l == (-1, -1, 1, -1, -1, 1, 1, 1):
        return False
    return True


def _in_lambda_32(l: tuple) -> bool:
    """Lambda_32 = Lambda_256 with prefix (-1,-1,-1,-1) and additional constraint."""
    if not _in_lambda_256(l):
        return False
    if l[0] != -1 or l[1] != -1 or l[2] != -1 or l[3] != -1:
        return False
    if l[4] == 1 and l[5] != -1:
        return False
    return True


# Map level names to predicate functions
_PREDICATES = {
    "base": _in_base_universe,
    "2048": _in_lambda_2048,
    "1024": _in_lambda_1024,
    "512": _in_lambda_512,
    "256": _in_lambda_256,
    "32": _in_lambda_32,
}


def enumerate_codebook(level: str = "2048") -> Tensor:
    """
    Enumerate all points in the given codebook level.

    Args:
        level: one of "base", "2048", "1024", "512", "256", "32"

    Returns:
        Tensor of shape (N, 8) with int8 values in {-1, 0, +1}.
    """
    predicate = _PREDICATES[level]
    vals = (-1, 0, 1)
    points = []
    for l in itertools.product(vals, repeat=8):
        if predicate(l):
            points.append(l)
    return torch.tensor(points, dtype=torch.int8)


# Cache for precomputed codebooks
_CODEBOOK_CACHE: Dict[str, Tensor] = {}


def get_codebook(level: str = "2048", device: str = "cpu") -> Tensor:
    """
    Get the codebook for a given level, with caching.

    Args:
        level: one of "base", "2048", "1024", "512", "256", "32"
        device: torch device

    Returns:
        Tensor of shape (N, 8) with float32 values in {-1.0, 0.0, +1.0}.
    """
    if level not in _CODEBOOK_CACHE:
        _CODEBOOK_CACHE[level] = enumerate_codebook(level).float()
    return _CODEBOOK_CACHE[level].to(device)


def codebook_sizes() -> Dict[str, int]:
    """Return the sizes of all codebook levels."""
    return {level: len(enumerate_codebook(level)) for level in _PREDICATES}


def nearest_neighbor(
    x: Tensor,
    codebook: Tensor,
) -> Tensor:
    """
    Find the nearest codebook point for each input vector via L2 distance.

    Args:
        x: input vectors, shape (..., 8), float
        codebook: codebook points, shape (N, 8), float

    Returns:
        Indices of nearest codebook points, shape (...).
    """
    # x: (..., 8) -> (..., 1, 8)
    # codebook: (N, 8) -> (1, ..., 1, N, 8)
    x_expanded = x.unsqueeze(-2)  # (..., 1, 8)
    diffs = x_expanded - codebook  # (..., N, 8)
    dists = (diffs * diffs).sum(dim=-1)  # (..., N)
    return dists.argmin(dim=-1)  # (...)


def quantize_blocks(
    x: Tensor,
    level: str = "2048",
    scale: Optional[float] = None,
) -> Tensor:
    """
    Quantize 8D blocks using the prefix-cut lattice codebook.

    Args:
        x: input vectors, shape (..., 8), float
        level: codebook level
        scale: scaling factor to match codebook radius. If None, auto-computed
               to minimize MSE.

    Returns:
        Codebook indices, shape (...).
    """
    codebook = get_codebook(level, device=x.device)

    if scale is None:
        # Auto-scale: match the RMS of x to the RMS of the codebook
        x_rms = x.pow(2).mean().sqrt()
        cb_rms = codebook.pow(2).mean().sqrt()
        scale = x_rms / (cb_rms + 1e-8)

    x_scaled = x / (scale + 1e-8)
    return nearest_neighbor(x_scaled, codebook)


def dequantize_blocks(
    indices: Tensor,
    level: str = "2048",
    scale: float = 1.0,
    device: str = "cpu",
) -> Tensor:
    """
    Dequantize indices back to 8D vectors.

    Args:
        indices: codebook indices, shape (...)
        level: codebook level
        scale: same scaling factor used in quantize_blocks
        device: torch device

    Returns:
        Reconstructed vectors, shape (..., 8), float.
    """
    codebook = get_codebook(level, device=device)
    return codebook[indices] * scale


def bits_per_dim(level: str) -> float:
    """Return the bits per dimension for a given codebook level."""
    import math
    n = len(enumerate_codebook(level))
    return math.log2(n) / 8

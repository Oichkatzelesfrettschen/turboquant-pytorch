"""
DEPRECATED: This module is superseded by rotations.WHTRotation.

The Walsh-Hadamard Transform rotation is now part of the unified rotation
interface in rotations.py, with a vectorized butterfly implementation that
eliminates the Python inner loops from the original version.

Usage migration:
    # Old (this file):
    from wht_rotation import generate_wht_rotation, wht_rotate, wht_unrotate
    rot = generate_wht_rotation(d, seed=42)
    y = wht_rotate(x, rot)
    x_rec = wht_unrotate(y, rot)

    # New (rotations.py):
    from rotations import WHTRotation
    rot = WHTRotation(d, seed=42)
    y = rot.rotate(x)
    x_rec = rot.unrotate(y)

The key mathematical insight preserved in rotations.py:
    The WHT butterfly structure IS the Cayley-Dickson doubling butterfly.
    Each level of the CD tower (4D -> 8D -> 16D -> ...) maps to one
    WHT butterfly level. The operation (a,b) -> (a+b, a-b) at each
    level is the CD doubling formula for scalar multiplication.

Reference: Ailon & Chazelle 2006, "Fast Johnson-Lindenstrauss transform".
"""

import warnings
from typing import Optional

import torch
from torch import Tensor

# Re-export the vectorized implementation for backward compatibility
from .rotations import _fast_hadamard as hadamard_transform, WHTRotation


def generate_wht_rotation(d: int, seed: Optional[int] = None, device: str = "cpu") -> dict:
    """Deprecated. Use WHTRotation(d, seed, device) instead."""
    warnings.warn(
        "generate_wht_rotation is deprecated. Use rotations.WHTRotation instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    rot = WHTRotation(d, seed=seed, device=device)
    return {"d1": rot.d1, "d2": rot.d2}


def wht_rotate(x: Tensor, rotation: dict) -> Tensor:
    """Deprecated. Use WHTRotation.rotate() instead."""
    y = x * rotation["d2"]
    y = hadamard_transform(y)
    y = y * rotation["d1"]
    return y


def wht_unrotate(y: Tensor, rotation: dict) -> Tensor:
    """Deprecated. Use WHTRotation.unrotate() instead."""
    x = y * rotation["d1"]
    x = hadamard_transform(x)
    x = x * rotation["d2"]
    return x

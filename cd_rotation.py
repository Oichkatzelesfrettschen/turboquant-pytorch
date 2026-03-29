"""
Cayley-Dickson structured block rotations for vector quantization.

Uses CD algebra multiplication as a structured linear map to decorrelate
coordinates before quantization. The key insight: left-multiplication by a
unit CD element x -> a*x is a linear map represented by the matrix L_a.

For composition algebras (dim <= 8), unit multiplication preserves norms:
||a*x|| = ||a|| * ||x|| = ||x||, making it an exact isometry (rotation or
reflection). This gives O(d) storage and O(d log d) application cost vs
O(d^2) for a dense Haar rotation.

For dim >= 16 (sedenions+), zero divisors break norm multiplicativity.
The rotation is approximate -- some directions get compressed or expanded.
This is novel research territory: does decorrelation quality matter more
than exact isometry for quantization?

The block rotation strategy: partition a d-dimensional vector into blocks
of size block_dim, apply independent CD rotations to each block. For
d=128, block_dim=8: 16 independent octonion rotations.
"""

import torch
from torch import Tensor
from typing import Optional

from .cd_algebra import (
    cd_multiply,
    cd_conjugate,
    cd_inverse,
    cd_normalize,
    cd_norm,
    cd_random_unit,
)


def quaternion_sandwich(x: Tensor, q: Tensor) -> Tensor:
    """
    Quaternion sandwich product: x -> q * x * q^{-1}.

    For unit quaternion q, this is a rotation in R^3 that fixes the real axis.
    The rotation angle is 2*arccos(q[0]) around the axis (q[1], q[2], q[3]).

    This is the standard quaternion rotation formula used in 3D graphics,
    robotics, and attitude representation.

    Args:
        x: vectors to rotate, shape (..., 4)
        q: unit quaternion(s), shape (..., 4) or (4,) for broadcast

    Returns:
        Rotated vectors, shape (..., 4).
    """
    q_inv = cd_conjugate(q)  # for unit quaternions, q* = q^{-1}
    return cd_multiply(cd_multiply(q, x), q_inv)


def quaternion_sandwich_inverse(y: Tensor, q: Tensor) -> Tensor:
    """
    Inverse of quaternion sandwich: y -> q^{-1} * y * q.

    Args:
        y: rotated vectors, shape (..., 4)
        q: unit quaternion(s), shape (..., 4) or (4,) for broadcast

    Returns:
        Original vectors, shape (..., 4).
    """
    q_inv = cd_conjugate(q)
    return cd_multiply(cd_multiply(q_inv, y), q)


def cd_left_rotate(x: Tensor, a: Tensor) -> Tensor:
    """
    Left-multiplication rotation: x -> a * x.

    For unit octonions (dim=8), this is an isometry in R^8 because
    octonions are a composition algebra: ||a*x|| = ||a||*||x|| = ||x||.

    For dim >= 16, this is NOT an isometry due to zero divisors.
    The distortion can be measured via cd_associator_norm.

    Args:
        x: vectors to rotate, shape (..., 2^k)
        a: unit CD element(s), shape (..., 2^k) or (2^k,) for broadcast

    Returns:
        Rotated vectors, shape (..., 2^k).
    """
    return cd_multiply(a, x)


def cd_left_unrotate(y: Tensor, a: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Inverse of left-multiplication: y -> a^{-1} * y.

    For composition algebras (dim <= 8), this is exact.
    For dim >= 16, this is approximate due to zero divisors --
    a^{-1} * (a * x) != x in general when associativity fails.

    Args:
        y: rotated vectors, shape (..., 2^k)
        a: unit CD element(s), shape (..., 2^k) or (2^k,) for broadcast
        eps: regularization for inverse computation

    Returns:
        Approximately unrotated vectors, shape (..., 2^k).
    """
    a_inv = cd_inverse(a, eps=eps)
    return cd_multiply(a_inv, y)


def cd_block_rotate(
    x: Tensor,
    elements: Tensor,
    block_dim: int,
    use_sandwich: bool = False,
) -> Tensor:
    """
    Block CD rotation: partition x into blocks and apply independent CD rotations.

    For d=128, block_dim=8: reshapes to (batch, 16, 8), applies 16 independent
    octonion left-multiplications, reshapes back to (batch, 128).

    Args:
        x: input vectors, shape (..., d) where d is divisible by block_dim
        elements: unit CD elements for each block, shape (n_blocks, block_dim)
        block_dim: size of each block (4 for quaternion, 8 for octonion, etc.)
        use_sandwich: if True, use q*x*q^{-1} for dim=4 (true rotation in R^3).
                      if False, use left-multiply a*x (rotation in R^{block_dim}).

    Returns:
        Rotated vectors, same shape as x.
    """
    d = x.shape[-1]
    if d % block_dim != 0:
        raise ValueError(
            f"Dimension {d} not divisible by block_dim {block_dim}"
        )

    n_blocks = d // block_dim
    if elements.shape[0] != n_blocks:
        raise ValueError(
            f"Expected {n_blocks} elements, got {elements.shape[0]}"
        )

    batch_shape = x.shape[:-1]
    blocks = x.reshape(*batch_shape, n_blocks, block_dim)

    if use_sandwich and block_dim == 4:
        rotated = quaternion_sandwich(blocks, elements)
    else:
        # Left-multiply each block by its corresponding unit element
        # elements: (n_blocks, block_dim) -> broadcast with (batch, n_blocks, block_dim)
        rotated = cd_left_rotate(blocks, elements)

    return rotated.reshape(*batch_shape, d)


def cd_block_unrotate(
    y: Tensor,
    elements: Tensor,
    block_dim: int,
    use_sandwich: bool = False,
    eps: float = 1e-8,
) -> Tensor:
    """
    Inverse of cd_block_rotate.

    For composition algebras (dim <= 8): exact inverse.
    For dim >= 16: approximate inverse (associativity failure).

    Args:
        y: rotated vectors, shape (..., d)
        elements: same unit CD elements used for rotation
        block_dim: same block size used for rotation
        use_sandwich: must match the value used in cd_block_rotate
        eps: regularization for inverse computation

    Returns:
        Approximately unrotated vectors, same shape as y.
    """
    d = y.shape[-1]
    n_blocks = d // block_dim
    batch_shape = y.shape[:-1]
    blocks = y.reshape(*batch_shape, n_blocks, block_dim)

    if use_sandwich and block_dim == 4:
        unrotated = quaternion_sandwich_inverse(blocks, elements)
    else:
        unrotated = cd_left_unrotate(blocks, elements, eps=eps)

    return unrotated.reshape(*batch_shape, d)


def generate_cd_block_elements(
    d: int,
    block_dim: int,
    seed: Optional[int] = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Generate random unit CD elements for block rotation.

    Storage cost: n_blocks * block_dim floats = d floats total.
    Compare: Haar rotation stores d*d floats.

    Args:
        d: total vector dimension
        block_dim: block size (4, 8, 16, 32, ...)
        seed: optional random seed
        device: torch device
        dtype: tensor dtype

    Returns:
        Unit CD elements, shape (n_blocks, block_dim).
    """
    n_blocks = d // block_dim
    return cd_random_unit(n_blocks, dim=block_dim, seed=seed, device=device, dtype=dtype)


def cd_multi_layer_rotate(
    x: Tensor,
    elements_list: list,
    block_dims: list,
) -> Tensor:
    """
    Multi-layer CD rotation: apply block rotations at multiple scales.

    Layer 1: block_dim=4 (quaternion rotations within 4-element groups)
    Layer 2: block_dim=8 (octonion rotations across pairs of quat groups)
    Layer 3: block_dim=16 (sedenion rotations across pairs of oct groups)

    This mirrors the CD tower structure: each layer doubles the block size
    and mixes across the previous layer's blocks.

    Args:
        x: input vectors, shape (..., d)
        elements_list: list of element tensors, one per layer
        block_dims: list of block dimensions, one per layer

    Returns:
        Rotated vectors, same shape as x.
    """
    result = x
    for elements, block_dim in zip(elements_list, block_dims):
        result = cd_block_rotate(result, elements, block_dim)
    return result


def cd_multi_layer_unrotate(
    y: Tensor,
    elements_list: list,
    block_dims: list,
    eps: float = 1e-8,
) -> Tensor:
    """
    Inverse of cd_multi_layer_rotate: unrotate in reverse layer order.

    Args:
        y: rotated vectors, shape (..., d)
        elements_list: same elements used for rotation
        block_dims: same block dimensions used for rotation
        eps: regularization for inverse computation

    Returns:
        Approximately unrotated vectors, same shape as y.
    """
    result = y
    for elements, block_dim in reversed(list(zip(elements_list, block_dims))):
        result = cd_block_unrotate(result, elements, block_dim, eps=eps)
    return result


def measure_rotation_quality(
    x: Tensor,
    rotate_fn,
    unrotate_fn,
) -> dict:
    """
    Measure the quality of a rotation method for quantization.

    Args:
        x: test vectors, shape (n, d)
        rotate_fn: callable x -> y
        unrotate_fn: callable y -> x_recovered

    Returns:
        Dict with metrics:
            invertibility_error: ||x - unrotate(rotate(x))|| / ||x||
            coord_variance: mean per-coordinate variance (target: 1/d for unit vectors)
            max_cross_corr: max absolute cross-correlation between adjacent coords
            isometry_error: mean |( ||rotate(x)|| / ||x|| ) - 1|
    """
    d = x.shape[-1]
    y = rotate_fn(x)
    x_recovered = unrotate_fn(y)

    invertibility_error = (x - x_recovered).norm() / x.norm()

    coord_var = y.var(dim=0).mean()

    cc_vals = []
    for i in range(d - 1):
        cc = torch.corrcoef(y[:, i:i + 2].T)[0, 1]
        cc_vals.append(cc.abs())
    max_cross_corr = torch.stack(cc_vals).max() if cc_vals else torch.tensor(0.0)

    x_norms = x.norm(dim=-1)
    y_norms = y.norm(dim=-1)
    isometry_error = ((y_norms / (x_norms + 1e-8)) - 1.0).abs().mean()

    return {
        "invertibility_error": invertibility_error.item(),
        "coord_variance": coord_var.item(),
        "max_cross_corr": max_cross_corr.item(),
        "isometry_error": isometry_error.item(),
    }

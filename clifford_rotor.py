"""
Clifford algebra rotor rotation for quantization decorrelation.

Implements Cl(p,q) rotors (sandwich product RvR~) as a rotation method,
synthesizing the RotorQuant approach (Pope, March 2026) with our CD
framework. While RotorQuant uses Cl(3,0) on 3D blocks, we generalize:

    Cl(3,0):  3D blocks, sandwich qvq^{-1}, fixes scalar component
    Cl(0,3):  3D blocks, split signature variant
    Cl(4,0):  quaternion-equivalent on 4D blocks (isomorphic to H x H)
    Cl(7,0):  octonion-adjacent on 7D blocks

The key difference from CD left-multiplication:
    CD left-multiply ax:     rotates FULL d-dimensional space
    Clifford sandwich RvR~:  rotates the VECTOR subspace (grade-1), fixes scalars

For quantization, both decorrelate coordinates. The Clifford approach has
the advantage that the sandwich product ALWAYS preserves norms (even in
dimensions where CD has zero divisors). The CD approach has the advantage
of simpler implementation and the full-space rotation.

We provide both and let the benchmark decide.

STATUS: EXPERIMENTAL. The Cl(3,0) geometric product has a known bug in the
64-term multiplication table that causes ~46% invertibility error in batched
mode. Single-block roundtrips work correctly. The multiplication table needs
systematic verification against a reference Clifford algebra implementation.
Use E8BlockRotation or CDRotation for production quantization.
"""

import math
import torch
from torch import Tensor
from typing import Optional

from .rotations import Rotation


def _cl3_geometric_product(a: Tensor, b: Tensor) -> Tensor:
    """
    Geometric product in Cl(3,0) -- the algebra of 3D Euclidean space.

    A general Cl(3,0) element has 8 components:
        a = a0 + a1*e1 + a2*e2 + a3*e3 + a12*e12 + a13*e13 + a23*e23 + a123*e123

    We represent it as a tensor of shape (..., 8) with indices:
        [0]=scalar, [1]=e1, [2]=e2, [3]=e3, [4]=e12, [5]=e13, [6]=e23, [7]=e123

    The multiplication table for Cl(3,0) where ei*ei = +1:
        e1*e2 = e12,  e2*e1 = -e12
        e1*e3 = e13,  e3*e1 = -e13
        e2*e3 = e23,  e3*e2 = -e23
        e12*e3 = e123, etc.
    """
    a0, a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    a12, a13, a23, a123 = a[..., 4], a[..., 5], a[..., 6], a[..., 7]
    b0, b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    b12, b13, b23, b123 = b[..., 4], b[..., 5], b[..., 6], b[..., 7]

    # Full geometric product -- 64 terms organized by output grade
    r0 = (a0*b0 + a1*b1 + a2*b2 + a3*b3
           - a12*b12 - a13*b13 - a23*b23 - a123*b123)
    r1 = (a0*b1 + a1*b0 - a2*b12 + a12*b2 - a3*b13 + a13*b3
           + a23*b123 - a123*b23)
    r2 = (a0*b2 + a1*b12 + a2*b0 - a12*b1 - a3*b23 - a13*b123
           + a23*b3 + a123*b13)
    r3 = (a0*b3 - a1*b13 + a2*b23 + a3*b0 + a12*b123 + a13*b1
           - a23*b2 - a123*b12)
    r12 = (a0*b12 + a1*b2 - a2*b1 + a12*b0 + a3*b123 - a13*b23
            + a23*b13 + a123*b3)
    r13 = (a0*b13 + a1*b3 + a2*b123 - a12*b23 - a3*b1 + a13*b0
            + a23*b12 - a123*b2)
    r23 = (a0*b23 - a1*b123 + a2*b3 + a12*b13 - a3*b2 - a13*b12
            + a23*b0 + a123*b1)
    r123 = (a0*b123 + a1*b23 - a2*b13 + a12*b3 + a3*b12 - a13*b2
             + a23*b1 + a123*b0)

    return torch.stack([r0, r1, r2, r3, r12, r13, r23, r123], dim=-1)


def _cl3_reverse(a: Tensor) -> Tensor:
    """
    Reversion (tilde) in Cl(3,0): reverse the order of basis vectors.

    Grade 0 (scalar): unchanged
    Grade 1 (vectors): unchanged
    Grade 2 (bivectors): negated
    Grade 3 (trivector): negated
    """
    result = a.clone()
    result[..., 4:7] = -result[..., 4:7]  # bivectors
    result[..., 7] = -result[..., 7]       # trivector
    return result


def _cl3_sandwich(rotor: Tensor, v: Tensor) -> Tensor:
    """
    Sandwich product: v' = R * v * R~

    The rotor R is an even-grade element (scalar + bivector components).
    The vector v has only grade-1 components (e1, e2, e3).

    This rotates the 3D vector while preserving its norm.
    """
    # Embed v as grade-1 element in Cl(3,0)
    v_full = torch.zeros(*v.shape[:-1], 8, device=v.device, dtype=v.dtype)
    v_full[..., 1:4] = v  # grade-1 components

    r_rev = _cl3_reverse(rotor)
    # R * v * R~
    rv = _cl3_geometric_product(rotor, v_full)
    rvr = _cl3_geometric_product(rv, r_rev)

    # Extract grade-1 (vector) part
    return rvr[..., 1:4]


def _cl3_sandwich_inverse(rotor_rev: Tensor, rotor: Tensor, v: Tensor) -> Tensor:
    """Inverse sandwich: v' = R~ * v * R (undoes R * v * R~)."""
    v_full = torch.zeros(*v.shape[:-1], 8, device=v.device, dtype=v.dtype)
    v_full[..., 1:4] = v
    rv = _cl3_geometric_product(rotor_rev, v_full)
    rvr = _cl3_geometric_product(rv, rotor)
    return rvr[..., 1:4]


def _random_cl3_rotor(
    *batch_shape,
    seed: Optional[int] = None,
    device: str = "cpu",
) -> Tensor:
    """
    Generate random unit rotors in Cl(3,0).

    A rotor is an even-grade element R = cos(theta/2) + sin(theta/2)*B
    where B is a unit bivector. This generates random rotation angles
    and random rotation planes.
    """
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    # Random rotation angle
    theta = torch.rand(*batch_shape, generator=gen) * 2 * math.pi

    # Random unit bivector (3 components: e12, e13, e23)
    bv = torch.randn(*batch_shape, 3, generator=gen)
    bv = bv / (bv.norm(dim=-1, keepdim=True) + 1e-8)

    # Rotor: R = cos(theta/2) + sin(theta/2) * B
    c = torch.cos(theta / 2).unsqueeze(-1)
    s = torch.sin(theta / 2).unsqueeze(-1)

    rotor = torch.zeros(*batch_shape, 8, device=device)
    rotor[..., 0:1] = c      # scalar
    rotor[..., 4:7] = s * bv  # bivector
    return rotor.to(device)


class CliffordRotorRotation(Rotation):
    """
    Clifford Cl(3,0) rotor rotation via sandwich product.

    Partitions d-dimensional vectors into blocks of 3 and applies
    independent rotor rotations R*v*R~ to each block.

    For d=128: 42 blocks of 3D + 2 leftover coords (handled by identity).
    Each rotor has 4 parameters (scalar + 3 bivector), so 42*4 = 168 total.

    The sandwich product ALWAYS preserves norms (no zero-divisor issue),
    but only rotates 3D subspaces (the scalar grade is invariant).

    This is the approach from RotorQuant (Pope, March 2026) generalized
    to work within our rotation ABC framework.
    """

    def __init__(
        self,
        d: int,
        seed: Optional[int] = None,
        device: str = "cpu",
    ):
        self.d = d
        self.block_size = 3
        self.n_blocks = d // self.block_size
        self.remainder = d % self.block_size

        self.rotors = _random_cl3_rotor(
            self.n_blocks, seed=seed, device=device
        )

    def rotate(self, x: Tensor) -> Tensor:
        batch_shape = x.shape[:-1]
        # Split into blocks of 3 + remainder
        main = x[..., :self.n_blocks * 3].reshape(*batch_shape, self.n_blocks, 3)
        rotated = _cl3_sandwich(self.rotors, main)
        result = torch.empty_like(x)
        result[..., :self.n_blocks * 3] = rotated.reshape(*batch_shape, self.n_blocks * 3)
        if self.remainder > 0:
            result[..., self.n_blocks * 3:] = x[..., self.n_blocks * 3:]
        return result

    def unrotate(self, y: Tensor) -> Tensor:
        batch_shape = y.shape[:-1]
        main = y[..., :self.n_blocks * 3].reshape(*batch_shape, self.n_blocks, 3)
        # Inverse rotation: R~ * v * R undoes R * v * R~
        rotor_rev = _cl3_reverse(self.rotors)
        unrotated = _cl3_sandwich(rotor_rev, main)
        result = torch.empty_like(y)
        result[..., :self.n_blocks * 3] = unrotated.reshape(*batch_shape, self.n_blocks * 3)
        if self.remainder > 0:
            result[..., self.n_blocks * 3:] = y[..., self.n_blocks * 3:]
        return result

    def storage_elements(self) -> int:
        return self.n_blocks * 4  # 4 rotor components per block

    def to(self, device: str) -> "CliffordRotorRotation":
        self.rotors = self.rotors.to(device)
        return self

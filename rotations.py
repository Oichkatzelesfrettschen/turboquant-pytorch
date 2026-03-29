"""
Unified rotation interface for TurboQuant.

Provides a common ABC for all rotation methods, enabling plug-and-play
substitution in the quantization pipeline. Rotation decorrelates coordinates
so that per-coordinate (or per-block) quantization approaches optimality.

Implementations:
    HaarRotation:   Dense random orthogonal matrix. O(d^2) storage, O(d^2) apply.
                    Gold standard for decorrelation quality.
    WHTRotation:    Walsh-Hadamard with random signs. O(d) storage, O(d log d) apply.
                    Near-Haar quality at fraction of cost. Ailon-Chazelle 2006.
    CDBlockRotation: Cayley-Dickson algebra multiplication. O(d) storage, O(d log d) apply.
                    Algebraically structured; isometric for block_dim <= 8.
    PCARotation:    Data-dependent eigenvector rotation. O(d^2) storage, requires calibration.
                    Optimal decorrelation for the calibration distribution.
    KacRotation:    Random Givens rotations. O(k) storage for k rotations. Tunable
                    quality-speed tradeoff.
"""

import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .cd_rotation import (
    cd_block_rotate,
    cd_block_unrotate,
    generate_cd_block_elements,
    cd_multi_layer_rotate,
    cd_multi_layer_unrotate,
)


class Rotation(ABC):
    """Abstract base class for rotation methods."""

    @abstractmethod
    def rotate(self, x: Tensor) -> Tensor:
        """Apply rotation to decorrelate coordinates."""

    @abstractmethod
    def unrotate(self, y: Tensor) -> Tensor:
        """Undo rotation to recover original coordinate frame."""

    @abstractmethod
    def storage_elements(self) -> int:
        """Number of float elements stored for this rotation."""

    @abstractmethod
    def to(self, device: str) -> "Rotation":
        """Move rotation parameters to the given device."""


class HaarRotation(Rotation):
    """
    Dense Haar-distributed random orthogonal rotation via QR decomposition.

    The gold standard: uniform over SO(d), optimal decorrelation.
    Cost: O(d^2) storage, O(d^2) matmul per application.
    """

    def __init__(self, d: int, seed: Optional[int] = None, device: str = "cpu"):
        self.d = d
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        G = torch.randn(d, d, generator=gen)
        Q, R = torch.linalg.qr(G)
        diag_sign = torch.sign(torch.diag(R))
        diag_sign[diag_sign == 0] = 1.0
        self.Pi = (Q * diag_sign.unsqueeze(0)).to(device)

    def rotate(self, x: Tensor) -> Tensor:
        return x @ self.Pi.T

    def unrotate(self, y: Tensor) -> Tensor:
        return y @ self.Pi

    def storage_elements(self) -> int:
        return self.d * self.d

    def to(self, device: str) -> "HaarRotation":
        self.Pi = self.Pi.to(device)
        return self


class WHTRotation(Rotation):
    """
    Fast Walsh-Hadamard rotation: Pi = D1 @ H_d @ D2.

    D1, D2 are random diagonal sign matrices (+1/-1).
    H_d is the normalized Hadamard matrix applied via butterfly.

    Storage: O(d) -- just two sign vectors.
    Application: O(d log d) -- butterfly structure.

    The butterfly at each level IS the Cayley-Dickson doubling:
    (a,b) -> (a+b, a-b) matches the CD formula for scalar multiplication.

    Reference: Ailon & Chazelle 2006, "Fast Johnson-Lindenstrauss transform".
    """

    def __init__(self, d: int, seed: Optional[int] = None, device: str = "cpu"):
        if d == 0 or (d & (d - 1)) != 0:
            raise ValueError(f"d must be a power of 2, got {d}")
        self.d = d
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        self.d1 = torch.sign(torch.randn(d, generator=gen)).to(device)
        self.d1[self.d1 == 0] = 1.0
        self.d2 = torch.sign(torch.randn(d, generator=gen)).to(device)
        self.d2[self.d2 == 0] = 1.0

    def rotate(self, x: Tensor) -> Tensor:
        y = x * self.d2
        y = _fast_hadamard(y)
        y = y * self.d1
        return y

    def unrotate(self, y: Tensor) -> Tensor:
        # H is self-inverse (up to scaling), D matrices are self-inverse
        x = y * self.d1
        x = _fast_hadamard(x)
        x = x * self.d2
        return x

    def storage_elements(self) -> int:
        return 2 * self.d

    def to(self, device: str) -> "WHTRotation":
        self.d1 = self.d1.to(device)
        self.d2 = self.d2.to(device)
        return self


class CDRotation(Rotation):
    """
    Cayley-Dickson block rotation using algebraic multiplication.

    Partitions d-dimensional vectors into blocks of block_dim and applies
    independent CD left-multiplications by random unit elements.

    block_dim=4:  Quaternion rotation. Isometric. 32 blocks for d=128.
    block_dim=8:  Octonion rotation. Isometric (composition algebra). 16 blocks for d=128.
    block_dim=16: Sedenion rotation. NOT isometric (zero divisors). 8 blocks for d=128.
    block_dim=32+: Higher tower. Experimental.

    For quantization, the key question is decorrelation quality, not exact isometry.
    Sedenion rotations may decorrelate better due to richer algebraic structure,
    even though they distort norms slightly.

    Storage: O(d) -- one unit element per block.
    Application: O(d log(block_dim)) -- recursive CD multiply per block.
    """

    def __init__(
        self,
        d: int,
        block_dim: int = 8,
        seed: Optional[int] = None,
        device: str = "cpu",
        use_sandwich: bool = False,
    ):
        if d % block_dim != 0:
            raise ValueError(f"d={d} not divisible by block_dim={block_dim}")
        if block_dim == 0 or (block_dim & (block_dim - 1)) != 0:
            raise ValueError(f"block_dim must be a power of 2, got {block_dim}")
        self.d = d
        self.block_dim = block_dim
        self.use_sandwich = use_sandwich
        self.elements = generate_cd_block_elements(
            d, block_dim, seed=seed, device=device
        )

    def rotate(self, x: Tensor) -> Tensor:
        return cd_block_rotate(x, self.elements, self.block_dim, self.use_sandwich)

    def unrotate(self, y: Tensor) -> Tensor:
        return cd_block_unrotate(y, self.elements, self.block_dim, self.use_sandwich)

    def storage_elements(self) -> int:
        return self.elements.numel()

    def to(self, device: str) -> "CDRotation":
        self.elements = self.elements.to(device)
        return self


class CDMultiLayerRotation(Rotation):
    """
    Multi-layer CD rotation at increasing block sizes.

    Applies CD rotations at multiple scales, mirroring the CD tower:
    Layer 1: quaternion (4D blocks) -- local mixing
    Layer 2: octonion (8D blocks) -- medium-range mixing
    Layer 3: sedenion (16D blocks) -- long-range mixing

    This gives better decorrelation than single-layer because each layer
    mixes across the previous layer's block boundaries.

    Storage: O(d * n_layers).
    Application: O(d * n_layers * log(max_block_dim)).
    """

    def __init__(
        self,
        d: int,
        block_dims: list = None,
        seed: Optional[int] = None,
        device: str = "cpu",
    ):
        self.d = d
        if block_dims is None:
            # Default: quaternion + octonion layers (both isometric)
            block_dims = [b for b in [4, 8] if d % b == 0]
        self.block_dims = block_dims
        self.elements_list = []
        for i, bd in enumerate(block_dims):
            s = seed + i * 1000 if seed is not None else None
            elems = generate_cd_block_elements(d, bd, seed=s, device=device)
            self.elements_list.append(elems)

    def rotate(self, x: Tensor) -> Tensor:
        return cd_multi_layer_rotate(x, self.elements_list, self.block_dims)

    def unrotate(self, y: Tensor) -> Tensor:
        return cd_multi_layer_unrotate(y, self.elements_list, self.block_dims)

    def storage_elements(self) -> int:
        return sum(e.numel() for e in self.elements_list)

    def to(self, device: str) -> "CDMultiLayerRotation":
        self.elements_list = [e.to(device) for e in self.elements_list]
        return self


class PCARotation(Rotation):
    """
    Data-dependent PCA rotation using eigenvectors of the covariance matrix.

    Requires a calibration set. Optimal decorrelation for the calibration
    distribution but may not generalize to other inputs.

    SliceGPT (Ashkboos et al., 2024) uses this approach for weight matrices.
    For KV cache, the covariance is computed over a calibration corpus.

    Storage: O(d^2) -- eigenvector matrix.
    Application: O(d^2) -- matmul.
    """

    def __init__(self, d: int, device: str = "cpu"):
        self.d = d
        self.Pi = torch.eye(d, device=device)  # identity until calibrated
        self._calibrated = False

    def calibrate(self, data: Tensor):
        """
        Compute PCA rotation from calibration data.

        Args:
            data: calibration vectors, shape (n, d)
        """
        data = data - data.mean(dim=0, keepdim=True)
        cov = (data.T @ data) / (data.shape[0] - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        # Sort by descending eigenvalue
        idx = eigenvalues.argsort(descending=True)
        self.Pi = eigenvectors[:, idx].T.to(data.device)
        self._calibrated = True

    def rotate(self, x: Tensor) -> Tensor:
        return x @ self.Pi.T

    def unrotate(self, y: Tensor) -> Tensor:
        return y @ self.Pi

    def storage_elements(self) -> int:
        return self.d * self.d

    def to(self, device: str) -> "PCARotation":
        self.Pi = self.Pi.to(device)
        return self


class KacRotation(Rotation):
    """
    Random Givens (Kac) rotation: product of random 2D rotations.

    Each Givens rotation mixes two coordinates with a random angle.
    After k rotations, coordinates become increasingly decorrelated.
    At k = O(d log d), quality approaches Haar.

    Tunable quality-speed tradeoff: more rotations = better decorrelation
    but slower application.

    Implementation: rotations are grouped into non-conflicting "rounds"
    where no coordinate appears twice. Each round applies up to d/2
    Givens rotations simultaneously via vectorized gather/scatter. For
    d=128, this reduces ~896 Python iterations to ~30 vectorized rounds.

    Storage: O(k) -- k (index_pair, angle) triples.
    Application: O(n_rounds) tensor operations instead of O(k) Python iterations.
    """

    def __init__(
        self,
        d: int,
        n_rotations: Optional[int] = None,
        seed: Optional[int] = None,
        device: str = "cpu",
    ):
        self.d = d
        if n_rotations is None:
            n_rotations = d * int(math.log2(d))
        self.n_rotations = n_rotations

        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)

        indices = torch.randint(0, d, (n_rotations, 2), generator=gen)
        mask = indices[:, 0] == indices[:, 1]
        indices[mask, 1] = (indices[mask, 1] + 1) % d

        angles = torch.rand(n_rotations, generator=gen) * 2 * math.pi
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)

        # Group rotations into non-conflicting rounds for vectorized application.
        # Two rotations conflict if they share any coordinate index.
        self.rounds = _build_kac_rounds(indices, cos_a, sin_a, d)
        self.rounds = [(r_i.to(device), r_j.to(device), r_c.to(device), r_s.to(device))
                       for r_i, r_j, r_c, r_s in self.rounds]
        self._n_rotations_actual = n_rotations

    def rotate(self, x: Tensor) -> Tensor:
        y = x.clone()
        for r_i, r_j, r_c, r_s in self.rounds:
            yi = y[..., r_i]  # (..., round_size)
            yj = y[..., r_j]  # (..., round_size)
            y[..., r_i] = yi * r_c - yj * r_s
            y[..., r_j] = yi * r_s + yj * r_c
        return y

    def unrotate(self, y: Tensor) -> Tensor:
        x = y.clone()
        for r_i, r_j, r_c, r_s in reversed(self.rounds):
            xi = x[..., r_i]
            xj = x[..., r_j]
            x[..., r_i] = xi * r_c + xj * r_s   # negated angle: sin -> -sin
            x[..., r_j] = -xi * r_s + xj * r_c
        return x

    def storage_elements(self) -> int:
        return self._n_rotations_actual * 3

    def to(self, device: str) -> "KacRotation":
        self.rounds = [(r_i.to(device), r_j.to(device), r_c.to(device), r_s.to(device))
                       for r_i, r_j, r_c, r_s in self.rounds]
        return self


def _build_kac_rounds(indices, cos_a, sin_a, d):
    """
    Partition Givens rotations into non-conflicting rounds.

    Two rotations conflict if they touch any shared coordinate.
    Each round contains rotations that can be applied simultaneously.

    Greedy coloring: for each rotation, assign it to the first round
    where neither of its coordinates is already used.
    """
    rounds_data = []  # list of (indices_i, indices_j, cos, sin) per round
    rounds_used = []  # set of used coordinates per round

    n = indices.shape[0]
    for k in range(n):
        i_val = indices[k, 0].item()
        j_val = indices[k, 1].item()

        placed = False
        for r_idx, used in enumerate(rounds_used):
            if i_val not in used and j_val not in used:
                used.add(i_val)
                used.add(j_val)
                rounds_data[r_idx][0].append(i_val)
                rounds_data[r_idx][1].append(j_val)
                rounds_data[r_idx][2].append(cos_a[k].item())
                rounds_data[r_idx][3].append(sin_a[k].item())
                placed = True
                break

        if not placed:
            rounds_used.append({i_val, j_val})
            rounds_data.append(([i_val], [j_val], [cos_a[k].item()], [sin_a[k].item()]))

    # Convert to tensors
    result = []
    for r_i_list, r_j_list, r_c_list, r_s_list in rounds_data:
        result.append((
            torch.tensor(r_i_list, dtype=torch.long),
            torch.tensor(r_j_list, dtype=torch.long),
            torch.tensor(r_c_list, dtype=torch.float32),
            torch.tensor(r_s_list, dtype=torch.float32),
        ))
    return result


# ---------------------------------------------------------------------------
# Vectorized Walsh-Hadamard Transform (no Python loops in the butterfly)
# ---------------------------------------------------------------------------

def _fast_hadamard(x: Tensor) -> Tensor:
    """
    Fast Walsh-Hadamard Transform via vectorized butterfly operations.
    O(d log d) without Python inner loops.

    Uses torch.compile for fusion when available (2x speedup on GPU).

    Args:
        x: tensor of shape (..., d) where d is a power of 2.

    Returns:
        H_d @ x / sqrt(d), same shape as x.
    """
    return _fast_hadamard_impl(x)


def _fast_hadamard_impl(x: Tensor) -> Tensor:
    d = x.shape[-1]
    result = x.clone()

    h = 1
    while h < d:
        shape = result.shape[:-1]
        n_pairs = d // (2 * h)
        result = result.reshape(*shape, n_pairs, 2, h)
        a = result[..., 0, :].clone()
        b = result[..., 1, :].clone()
        result[..., 0, :] = a + b
        result[..., 1, :] = a - b
        result = result.reshape(*shape, d)
        h *= 2

    return result / math.sqrt(d)


# torch.compile can accelerate the WHT butterfly on GPU but adds
# significant warmup cost on first call. Enable via environment variable:
#   TURBOQUANT_COMPILE=1 python ...
import os as _os
if _os.environ.get("TURBOQUANT_COMPILE", "0") == "1":
    try:
        _fast_hadamard_impl = torch.compile(_fast_hadamard_impl, mode="reduce-overhead")
    except Exception:
        pass

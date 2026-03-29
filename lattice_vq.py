"""
Unified lattice vector quantization interface.

Provides a common ABC for quantization methods, enabling plug-and-play
substitution in the TurboQuant pipeline. All quantizers operate on the
rotated coordinate space.

Implementations:
    ScalarLloydMaxQuantizer: Per-coordinate optimal scalar quantization (original TurboQuant).
    E8LatticeQuantizer:      8D block quantization via E8 closest-lattice-point.
    Z8PrefixCutQuantizer:    8D block quantization via CD tower prefix-cut codebooks.
"""

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor

from .lloyd_max import LloydMaxCodebook
from .e8_quantizer import e8_closest_point, e8_auto_scale
from .lattice_codebook import get_codebook, nearest_neighbor, bits_per_dim


class VectorQuantizer(ABC):
    """Abstract base class for vector quantizers."""

    @abstractmethod
    def quantize(self, x: Tensor) -> dict:
        """
        Quantize input vectors.

        Args:
            x: input vectors, shape (..., d)

        Returns:
            Dict containing quantization state needed for dequantization.
        """

    @abstractmethod
    def dequantize(self, state: dict) -> Tensor:
        """
        Dequantize from quantization state.

        Args:
            state: dict from quantize()

        Returns:
            Reconstructed vectors, shape (..., d).
        """

    @abstractmethod
    def bits_per_dimension(self) -> float:
        """Return the effective bits per dimension."""


class ScalarLloydMaxQuantizer(VectorQuantizer):
    """
    Per-coordinate Lloyd-Max optimal scalar quantization (original TurboQuant).

    Each coordinate is independently quantized to the nearest centroid from
    a precomputed codebook that minimizes MSE for the post-rotation distribution.

    Bits/dim: exactly the configured bit-width.
    """

    def __init__(self, d: int, bits: int, device: str = "cpu"):
        self.d = d
        self.bits = bits
        self.codebook = LloydMaxCodebook(d, bits)
        self.centroids = self.codebook.centroids.to(device)
        # Pre-compute sorted boundaries for searchsorted (5.5x faster on CPU)
        c = self.codebook.centroids
        self._boundaries = ((c[:-1] + c[1:]) / 2).to(device)
        self.device = device

    def quantize(self, x: Tensor) -> dict:
        # searchsorted: O(log n_levels) binary search per element.
        # 5.5x faster than argmin on CPU, comparable on GPU.
        indices = torch.searchsorted(self._boundaries, x)
        return {"indices": indices.to(torch.int8), "type": "scalar_lloyd_max"}

    def dequantize(self, state: dict) -> Tensor:
        return self.centroids[state["indices"].long()]

    def bits_per_dimension(self) -> float:
        return float(self.bits)

    def storage_bytes(self, n_elements: int) -> int:
        """Actual storage bytes for n_elements quantized values."""
        return n_elements  # int8: 1 byte per element

    def to(self, device: str) -> "ScalarLloydMaxQuantizer":
        self.centroids = self.centroids.to(device)
        self._boundaries = self._boundaries.to(device)
        self.device = device
        return self


class E8LatticeQuantizer(VectorQuantizer):
    """
    8D block quantization via E8 closest-lattice-point algorithm.

    Input vectors of dimension d are reshaped into blocks of 8, each block
    is quantized to its nearest E8 lattice point. Requires d % 8 == 0.

    The E8 lattice achieves G(E8) = 0.07168 vs G(Z) = 0.08333, giving
    ~14% MSE reduction over scalar quantization at the same bit rate.

    Bits/dim: depends on the number of E8 shells used (typically ~1 bit/dim
    for the first shell of 240 points).
    """

    def __init__(self, d: int, device: str = "cpu", scale_method: str = "grid_search"):
        if d % 8 != 0:
            raise ValueError(f"E8 quantizer requires d % 8 == 0, got d={d}")
        self.d = d
        self.n_blocks = d // 8
        self.device = device
        self._scale = None
        self._scale_method = scale_method  # grid_search is 16x better than heuristic

    def quantize(self, x: Tensor) -> dict:
        batch_shape = x.shape[:-1]
        blocks = x.reshape(*batch_shape, self.n_blocks, 8)

        # Auto-scale if not set
        if self._scale is None:
            self._scale = e8_auto_scale(blocks, method=self._scale_method)

        # Quantize each block to nearest E8 point
        scaled = blocks / (self._scale + 1e-8)
        lattice_points = e8_closest_point(scaled)

        return {
            "lattice_points": lattice_points,
            "scale": self._scale,
            "batch_shape": batch_shape,
            "type": "e8_lattice",
        }

    def dequantize(self, state: dict) -> Tensor:
        points = state["lattice_points"]
        scale = state["scale"]
        batch_shape = state["batch_shape"]
        reconstructed = points * scale
        return reconstructed.reshape(*batch_shape, self.d)

    def bits_per_dimension(self) -> float:
        # E8 with 240 roots + origin = 241 points
        # log2(241) / 8 ~ 0.99 bits/dim
        return math.log2(241) / 8

    def calibrate(self, data: Tensor):
        """Set the scale from calibration data using grid search for best MSE."""
        blocks = data.reshape(-1, 8)
        self._scale = e8_auto_scale(blocks, method="grid_search")

    def to(self, device: str) -> "E8LatticeQuantizer":
        self.device = device
        return self


class Z8PrefixCutQuantizer(VectorQuantizer):
    """
    8D block quantization via CD tower prefix-cut lattice codebooks.

    Input vectors are reshaped into blocks of 8, each block is quantized
    to its nearest point in the prefix-cut codebook at the selected level.

    Available levels and bit rates:
        "2048": 1.375 bits/dim (2048 codewords)
        "1024": 1.25 bits/dim  (1024 codewords, approximate)
        "512":  1.125 bits/dim (512 codewords)
        "256":  1.0 bits/dim   (256 codewords)
        "32":   0.625 bits/dim (32 codewords)
    """

    def __init__(self, d: int, level: str = "256", device: str = "cpu"):
        if d % 8 != 0:
            raise ValueError(f"Z8 quantizer requires d % 8 == 0, got d={d}")
        self.d = d
        self.n_blocks = d // 8
        self.level = level
        self.device = device
        self._scale = None
        # Preload the codebook
        self._codebook = get_codebook(level, device=device)

    def quantize(self, x: Tensor) -> dict:
        batch_shape = x.shape[:-1]
        blocks = x.reshape(*batch_shape, self.n_blocks, 8)

        # Auto-scale
        if self._scale is None:
            x_rms = blocks.pow(2).mean().sqrt().item()
            cb_rms = self._codebook.pow(2).mean().sqrt().item()
            self._scale = x_rms / (cb_rms + 1e-8)

        scaled = blocks / (self._scale + 1e-8)
        indices = nearest_neighbor(scaled, self._codebook)

        return {
            "indices": indices,
            "scale": self._scale,
            "batch_shape": batch_shape,
            "type": "z8_prefix_cut",
        }

    def dequantize(self, state: dict) -> Tensor:
        codebook = self._codebook
        points = codebook[state["indices"]]
        scale = state["scale"]
        batch_shape = state["batch_shape"]
        reconstructed = points * scale
        return reconstructed.reshape(*batch_shape, self.d)

    def bits_per_dimension(self) -> float:
        return bits_per_dim(self.level)

    def calibrate(self, data: Tensor):
        """Set the scale from calibration data."""
        blocks = data.reshape(-1, 8)
        x_rms = blocks.pow(2).mean().sqrt().item()
        cb_rms = self._codebook.pow(2).mean().sqrt().item()
        self._scale = x_rms / (cb_rms + 1e-8)

    def to(self, device: str) -> "Z8PrefixCutQuantizer":
        self.device = device
        self._codebook = self._codebook.to(device)
        return self

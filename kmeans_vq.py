"""
8D K-Means vector quantization codebook (NSNQuant-style).

After NSN pre-processing + rotation, coordinates are approximately N(0, 1/d).
This means a single codebook trained on synthetic standard normal data works
universally -- no calibration data needed from the target model.

The codebook is trained once at import time (or lazily on first use) and
reused for all quantization calls. This is the approach from NSNQuant
(arXiv 2505.18231) adapted for our pipeline.

Comparison with our other quantizers:
    ScalarLloydMax: 1D, analytically optimal for Gaussian, ~0.08333 NSM
    E8Lattice:      8D, densest packing, G(E8)=0.07168, ~14% better than scalar
    Z8PrefixCut:    8D, CD tower filtration, discrete codebook
    KMeans8D:       8D, data-driven (trained on N(0,1)), adapts to distribution

The K-Means codebook should achieve rate-distortion between E8 lattice
(geometrically optimal) and scalar Lloyd-Max (1D projection loss).
"""

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from .lattice_vq import VectorQuantizer


def train_kmeans_codebook(
    dim: int = 8,
    n_codewords: int = 256,
    n_train: int = 10000,
    n_iters: int = 30,
    seed: int = 42,
) -> Tensor:
    """
    Train a K-Means codebook on synthetic N(0, 1) data.

    This produces a universal codebook that works for any input
    after NSN pre-processing + rotation (which produces N(0, 1/d)).

    Args:
        dim: codebook vector dimension (8 for 8D VQ)
        n_codewords: number of codewords (256 for ~1 bit/dim)
        n_train: number of synthetic training vectors
        n_iters: K-Means iterations
        seed: random seed

    Returns:
        Codebook, shape (n_codewords, dim).
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    # Generate training data: standard normal (matches post-NSN+rotation distribution)
    data = torch.randn(n_train, dim, generator=gen)

    # Initialize centroids: K-Means++ style
    centroids = _kmeans_pp_init(data, n_codewords, gen)

    # K-Means iterations
    for _ in range(n_iters):
        # Assign each point to nearest centroid
        dists = torch.cdist(data, centroids)  # (n_train, n_codewords)
        assignments = dists.argmin(dim=1)     # (n_train,)

        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(n_codewords)
        for k in range(n_codewords):
            mask = assignments == k
            if mask.any():
                new_centroids[k] = data[mask].mean(dim=0)
                counts[k] = mask.sum()
            else:
                # Dead centroid: reinitialize to random point
                idx = torch.randint(0, n_train, (1,), generator=gen).item()
                new_centroids[k] = data[idx]
                counts[k] = 1

        centroids = new_centroids

    return centroids


def _kmeans_pp_init(data: Tensor, k: int, gen: torch.Generator) -> Tensor:
    """K-Means++ initialization: spread initial centroids far apart."""
    n, d = data.shape
    centroids = torch.zeros(k, d)

    # First centroid: random
    idx = torch.randint(0, n, (1,), generator=gen).item()
    centroids[0] = data[idx]

    for i in range(1, k):
        # Distance to nearest existing centroid
        dists = torch.cdist(data, centroids[:i]).min(dim=1).values  # (n,)
        probs = dists / (dists.sum() + 1e-15)
        # Sample proportional to distance
        idx = torch.multinomial(probs, 1, generator=gen).item()
        centroids[i] = data[idx]

    return centroids


# Cached codebooks by (dim, n_codewords)
_KMEANS_CACHE = {}


def get_kmeans_codebook(
    dim: int = 8,
    n_codewords: int = 256,
    device: str = "cpu",
) -> Tensor:
    """Get or train a K-Means codebook (cached)."""
    key = (dim, n_codewords)
    if key not in _KMEANS_CACHE:
        _KMEANS_CACHE[key] = train_kmeans_codebook(dim, n_codewords)
    return _KMEANS_CACHE[key].to(device)


class KMeans8DQuantizer(VectorQuantizer):
    """
    8D block quantization using a K-Means codebook trained on N(0,1).

    Input vectors of dimension d are reshaped into blocks of 8, each
    block quantized to the nearest codeword. Requires d % 8 == 0.

    The codebook is calibration-free: trained on synthetic Gaussian data,
    it works universally after NSN pre-processing + rotation.

    Bit rate: log2(n_codewords) / 8 bits per dimension.
        256 codewords: 1.0 bits/dim
        512 codewords: 1.125 bits/dim
        1024 codewords: 1.25 bits/dim
    """

    def __init__(
        self,
        d: int,
        n_codewords: int = 256,
        device: str = "cpu",
    ):
        if d % 8 != 0:
            raise ValueError(f"KMeans8D requires d % 8 == 0, got d={d}")
        self.d = d
        self.n_blocks = d // 8
        self.n_codewords = n_codewords
        self.device = device
        self._scale = None
        self._codebook = get_kmeans_codebook(8, n_codewords, device=device)

    def quantize(self, x: Tensor) -> dict:
        batch_shape = x.shape[:-1]
        blocks = x.reshape(*batch_shape, self.n_blocks, 8)

        if self._scale is None:
            self._scale = blocks.pow(2).mean().sqrt().item()

        scaled = blocks / (self._scale + 1e-15)
        # Nearest codeword via L2 distance
        flat_blocks = scaled.reshape(-1, 8)
        dists = torch.cdist(flat_blocks, self._codebook)
        indices = dists.argmin(dim=1)

        return {
            "indices": indices.reshape(*batch_shape, self.n_blocks),
            "scale": self._scale,
            "batch_shape": batch_shape,
            "type": "kmeans_8d",
        }

    def dequantize(self, state: dict) -> Tensor:
        indices = state["indices"].reshape(-1)
        scale = state["scale"]
        batch_shape = state["batch_shape"]
        codewords = self._codebook[indices]
        reconstructed = codewords.reshape(*batch_shape, self.n_blocks, 8) * scale
        return reconstructed.reshape(*batch_shape, self.d)

    def bits_per_dimension(self) -> float:
        return math.log2(self.n_codewords) / 8

    def calibrate(self, data: Tensor):
        """Set scale from calibration data."""
        blocks = data.reshape(-1, 8)
        self._scale = blocks.pow(2).mean().sqrt().item()

    def to(self, device: str) -> "KMeans8DQuantizer":
        self.device = device
        self._codebook = self._codebook.to(device)
        return self

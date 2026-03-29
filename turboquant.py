"""
TurboQuant: Two-stage vector quantization with near-optimal distortion.

Stage 1 (MSE): Random rotation + per-coordinate Lloyd-Max quantization
Stage 2 (QJL): 1-bit Quantized Johnson-Lindenstrauss on residuals for unbiased inner products

Reference: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (ICLR 2026)
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Union

from .lloyd_max import LloydMaxCodebook
from .rotations import Rotation, HaarRotation, WHTRotation, CDRotation
from .lattice_vq import VectorQuantizer, ScalarLloydMaxQuantizer


def _make_rotation(
    d: int,
    rotation: Union[None, str, Rotation],
    seed: int,
    device: str,
) -> Rotation:
    """
    Construct a Rotation instance from a flexible specification.

    Args:
        d: vector dimension
        rotation: None/"haar" for Haar, "wht" for Walsh-Hadamard,
                  "cd4"/"cd8"/"cd16"/... for CD block rotation,
                  or a pre-built Rotation instance.
        seed: random seed
        device: torch device

    Returns:
        Rotation instance ready to use.
    """
    if isinstance(rotation, Rotation):
        return rotation.to(device)
    if rotation is None or rotation == "haar":
        return HaarRotation(d, seed=seed, device=device)
    if rotation == "wht":
        return WHTRotation(d, seed=seed, device=device)
    if isinstance(rotation, str) and rotation.startswith("cd"):
        block_dim = int(rotation[2:])
        return CDRotation(d, block_dim=block_dim, seed=seed, device=device)
    raise ValueError(
        f"Unknown rotation: {rotation!r}. "
        f"Use None, 'haar', 'wht', 'cd4', 'cd8', 'cd16', or a Rotation instance."
    )


def _make_quantizer(
    d: int,
    bits: int,
    quantizer: Union[None, str, VectorQuantizer],
    device: str,
) -> VectorQuantizer:
    """
    Construct a VectorQuantizer from a flexible specification.

    Args:
        d: vector dimension
        bits: bit budget per dimension (used for scalar Lloyd-Max)
        quantizer: None/"scalar" for Lloyd-Max, "e8" for E8 lattice,
                   "z8_2048"/"z8_1024"/"z8_512"/"z8_256"/"z8_32" for prefix-cut,
                   or a pre-built VectorQuantizer instance.
        device: torch device

    Returns:
        VectorQuantizer instance.
    """
    if isinstance(quantizer, VectorQuantizer):
        return quantizer.to(device)
    if quantizer is None or quantizer == "scalar":
        return ScalarLloydMaxQuantizer(d, bits, device=device)
    if quantizer == "e8":
        from .lattice_vq import E8LatticeQuantizer
        return E8LatticeQuantizer(d, device=device)
    if isinstance(quantizer, str) and quantizer.startswith("z8"):
        from .lattice_vq import Z8PrefixCutQuantizer
        level = quantizer.split("_")[1] if "_" in quantizer else "256"
        return Z8PrefixCutQuantizer(d, level=level, device=device)
    raise ValueError(
        f"Unknown quantizer: {quantizer!r}. "
        f"Use None, 'scalar', 'e8', 'z8_256', etc., or a VectorQuantizer instance."
    )


def generate_rotation_matrix(d: int, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    """
    Generate a random orthogonal rotation matrix via QR decomposition of Gaussian matrix.
    This is the Haar-distributed random rotation used in TurboQuant.

    Kept for backward compatibility. New code should use rotations.HaarRotation.
    """
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen)
    Q, R = torch.linalg.qr(G)
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)
    return Q.to(device)


def generate_qjl_matrix(d: int, m: Optional[int] = None, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    """
    Generate the random projection matrix S for QJL.
    S has i.i.d. N(0,1) entries, shape (m, d).
    Default m = d (same dimensionality).
    """
    if m is None:
        m = d
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    S = torch.randn(m, d, generator=gen)
    return S.to(device)


class TurboQuantMSE(nn.Module):
    """
    Stage 1: MSE-optimal quantizer.
    Randomly rotates, then applies per-coordinate Lloyd-Max quantization.

    Supports pluggable rotation methods via the `rotation` parameter:
        None or "haar": Dense Haar rotation (original TurboQuant, backward compatible)
        "wht": Walsh-Hadamard rotation (O(d log d), requires d = power of 2)
        "cd4"/"cd8"/"cd16": CD block rotation with quaternion/octonion/sedenion blocks
        Rotation instance: any Rotation subclass from rotations.py

    Supports pluggable quantization methods via the `quantizer` parameter:
        None or "scalar": Per-coordinate Lloyd-Max (original TurboQuant)
        "e8": E8 lattice vector quantization (14% MSE improvement, requires d % 8 == 0)
        "z8_2048"/"z8_256"/"z8_32": CD tower prefix-cut codebooks (requires d % 8 == 0)
        VectorQuantizer instance: any VectorQuantizer subclass from lattice_vq.py
    """

    def __init__(
        self,
        d: int,
        bits: int,
        seed: int = 42,
        device: str = "cpu",
        rotation: Union[None, str, Rotation] = None,
        quantizer: Union[None, str, VectorQuantizer] = None,
    ):
        super().__init__()
        self.d = d
        self.bits = bits
        self.device = device

        # Set up rotation
        self._rotation = _make_rotation(d, rotation, seed, device)

        # Legacy buffer for backward compatibility with state_dict loading
        if isinstance(self._rotation, HaarRotation):
            self.register_buffer("Pi", self._rotation.Pi)
        else:
            self.register_buffer("Pi", torch.empty(0))

        # Set up quantizer
        self._quantizer = _make_quantizer(d, bits, quantizer, device)

        # Keep Lloyd-Max codebook for backward compatibility
        if isinstance(self._quantizer, ScalarLloydMaxQuantizer):
            self.codebook = LloydMaxCodebook(d, bits)
            self.register_buffer("centroids", self.codebook.centroids.to(device))
            self.register_buffer("boundaries", self.codebook.boundaries.to(device))
        else:
            self.codebook = None
            self.register_buffer("centroids", torch.empty(0))
            self.register_buffer("boundaries", torch.empty(0))

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotation to decorrelate coordinates."""
        return self._rotation.rotate(x)

    def unrotate(self, y: torch.Tensor) -> torch.Tensor:
        """Undo rotation to recover original coordinate frame."""
        return self._rotation.unrotate(y)

    def quantize(self, x: torch.Tensor) -> dict:
        """
        Quantize vectors. Returns quantization state dict.

        For backward compatibility with code expecting integer indices,
        scalar Lloyd-Max returns {"indices": Tensor, "type": "scalar_lloyd_max"}.
        """
        y = self.rotate(x)
        return self._quantizer.quantize(y)

    def dequantize(self, state) -> torch.Tensor:
        """
        Dequantize from quantization state.

        Args:
            state: dict from quantize(), or a plain Tensor of indices
                   (backward compatibility with scalar Lloyd-Max).
        """
        if isinstance(state, torch.Tensor):
            # Backward compatibility: plain indices tensor means scalar Lloyd-Max
            if self.codebook is not None:
                y_hat = self.centroids[state]
            else:
                y_hat = self._quantizer.dequantize({"indices": state})
            return self.unrotate(y_hat)
        y_hat = self._quantizer.dequantize(state)
        return self.unrotate(y_hat)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Full quantize-dequantize cycle.
        Returns: (reconstructed_x, quantization_state)
        """
        state = self.quantize(x)
        x_hat = self.dequantize(state)
        return x_hat, state


class TurboQuantProd(nn.Module):
    """
    Stage 1 + Stage 2: Unbiased inner product quantizer.
    Uses (b-1)-bit MSE quantizer + 1-bit QJL on residuals.

    Total storage per vector: (b-1)*d bits for MSE indices + d bits for QJL signs + 16 bits for residual norm
    Effective: ~b bits per dimension (the QJL bit replaces one MSE bit)
    """

    def __init__(
        self,
        d: int,
        bits: int,
        qjl_dim: Optional[int] = None,
        seed: int = 42,
        device: str = "cpu",
        rotation: Union[None, str, Rotation] = None,
        quantizer: Union[None, str, VectorQuantizer] = None,
    ):
        """
        Args:
            d: vector dimension
            bits: total bit budget per coordinate (MSE uses bits-1, QJL uses 1)
            qjl_dim: projection dimension for QJL (default = d)
            seed: random seed for reproducibility
            device: torch device
            rotation: rotation method (None/"haar"/"wht"/"cd4"/"cd8"/"cd16" or Rotation instance)
            quantizer: quantization method (None/"scalar"/"e8"/"z8_256" or VectorQuantizer instance)
        """
        super().__init__()
        self.d = d
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)
        self.qjl_dim = qjl_dim or d
        self.device = device

        # Stage 1: MSE quantizer with (bits-1) bits
        self.mse = TurboQuantMSE(
            d, self.mse_bits, seed=seed, device=device,
            rotation=rotation, quantizer=quantizer,
        )

        # Stage 2: QJL projection matrix
        self.register_buffer("S", generate_qjl_matrix(d, m=self.qjl_dim, seed=seed + 1, device=device))

    def quantize(self, x: torch.Tensor) -> dict:
        """
        Full TurboQuant quantization.

        Returns dict with:
            - 'mse_state': quantization state from MSE stage (dict)
            - 'qjl_signs': (batch, qjl_dim) sign bits of QJL-projected residual
            - 'residual_norm': (batch,) L2 norm of residual
        """
        # Stage 1: MSE quantize
        x_hat, mse_state = self.mse(x)

        # Compute residual
        residual = x - x_hat
        residual_norm = torch.norm(residual, dim=-1, keepdim=True)  # (batch, 1)

        # Stage 2: QJL - project residual and take sign
        projected = residual @ self.S.T  # (batch, qjl_dim)
        qjl_signs = torch.sign(projected)  # (batch, qjl_dim)
        qjl_signs[qjl_signs == 0] = 1.0  # map zeros to +1

        return {
            "mse_state": mse_state,
            "qjl_signs": qjl_signs,
            "residual_norm": residual_norm.squeeze(-1),
        }

    def dequantize(self, compressed: dict) -> torch.Tensor:
        """Dequantize MSE component (for reconstruction)."""
        return self.mse.dequantize(compressed["mse_state"])

    def inner_product(self, y: torch.Tensor, compressed: dict) -> torch.Tensor:
        """
        Compute unbiased inner product estimate: <y, x> using compressed representation of x.

        The estimator is:
            <y, x_mse> + ||r|| * sqrt(pi/2) / m * <S @ y, qjl_signs>

        Args:
            y: query vectors (batch, d) or (d,)
            compressed: dict from quantize()

        Returns:
            Estimated inner products (batch,)
        """
        # Term 1: inner product with MSE reconstruction
        x_mse = self.mse.dequantize(compressed["mse_state"])
        term1 = (y * x_mse).sum(dim=-1)

        # Term 2: QJL correction
        # Project query with same S matrix (but don't quantize query)
        y_projected = y @ self.S.T  # (batch, qjl_dim)
        qjl_ip = (y_projected * compressed["qjl_signs"]).sum(dim=-1)

        m = self.qjl_dim
        correction_scale = math.sqrt(math.pi / 2) / m
        term2 = compressed["residual_norm"] * correction_scale * qjl_ip

        return term1 + term2

    def forward(self, x: torch.Tensor) -> dict:
        """Quantize input vectors."""
        return self.quantize(x)


class TurboQuantKVCache:
    """
    KV cache wrapper that uses TurboQuant to compress keys and values.
    Drop-in replacement concept for a standard KV cache.
    """

    def __init__(
        self,
        d_key: int,
        d_value: int,
        bits: int = 3,
        seed: int = 42,
        device: str = "cpu",
        rotation: Union[None, str, Rotation] = None,
        quantizer: Union[None, str, VectorQuantizer] = None,
    ):
        self.d_key = d_key
        self.d_value = d_value
        self.bits = bits
        self.device = device

        # Use TurboQuantProd for keys (need inner products for attention)
        self.key_quantizer = TurboQuantProd(
            d_key, bits, seed=seed, device=device,
            rotation=rotation, quantizer=quantizer,
        )
        # Use TurboQuantMSE for values (need MSE reconstruction, not inner products)
        self.value_quantizer = TurboQuantMSE(
            d_value, bits, seed=seed + 100, device=device,
            rotation=rotation, quantizer=quantizer,
        )

        # Storage
        self.key_cache = []    # list of compressed key dicts
        self.value_cache = []  # list of quantization state dicts

    def append(self, keys: torch.Tensor, values: torch.Tensor):
        """
        Append new key-value pairs to cache.
        keys: (batch, seq_len, d_key) or (seq_len, d_key)
        values: (batch, seq_len, d_value) or (seq_len, d_value)
        """
        orig_shape = keys.shape
        flat_keys = keys.reshape(-1, self.d_key)
        flat_values = values.reshape(-1, self.d_value)

        compressed_keys = self.key_quantizer.quantize(flat_keys)
        value_state = self.value_quantizer.quantize(flat_values)

        compressed_keys["shape"] = orig_shape
        self.key_cache.append(compressed_keys)
        self.value_cache.append({
            "state": value_state,
            "shape": values.shape,
        })

    def attention_scores(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Compute attention scores between queries and all cached keys.
        Uses unbiased inner product estimation via TurboQuant.

        queries: (batch, d_key) or (d_key,)
        Returns: scores for each cached position
        """
        scores = []
        for cached in self.key_cache:
            s = self.key_quantizer.inner_product(queries, cached)
            scores.append(s)
        return torch.cat(scores, dim=-1) if scores else torch.tensor([])

    def get_values(self) -> torch.Tensor:
        """Reconstruct all cached values."""
        values = []
        for cached in self.value_cache:
            v = self.value_quantizer.dequantize(cached["state"])
            values.append(v)
        return torch.cat(values, dim=0) if values else torch.tensor([])

    def memory_usage_bits(self) -> dict:
        """Estimate memory usage in bits."""
        n_qjl = sum(c["qjl_signs"].numel() for c in self.key_cache) if self.key_cache else 0
        n_norms = sum(c["residual_norm"].numel() for c in self.key_cache) if self.key_cache else 0

        # Estimate key bits based on quantizer type
        key_quant_bits = self.key_quantizer.mse.d * self.key_quantizer.mse._quantizer.bits_per_dimension()
        n_key_vecs = sum(c["qjl_signs"].shape[0] for c in self.key_cache) if self.key_cache else 0
        key_bits = int(n_key_vecs * key_quant_bits) + n_qjl * 1 + n_norms * 16

        value_quant_bits = self.value_quantizer.d * self.value_quantizer._quantizer.bits_per_dimension()
        n_val_vecs = sum(1 for c in self.value_cache for _ in [c])  # count entries
        value_bits = int(n_val_vecs * value_quant_bits) if self.value_cache else 0

        total_elements = (n_key_vecs + n_val_vecs) * self.d_key  # approximate
        fp16_equivalent = total_elements * 16

        total = key_bits + value_bits
        return {
            "key_bits": key_bits,
            "value_bits": value_bits,
            "total_bits": total,
            "fp16_bits": fp16_equivalent,
            "compression_ratio": fp16_equivalent / total if total > 0 else 0,
        }

    def __len__(self):
        return sum(c["qjl_signs"].shape[0] for c in self.key_cache) if self.key_cache else 0

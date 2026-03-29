"""
NSN (Normalize-Shift-Normalize) pre-processing for quantization.

NSNQuant (arXiv 2505.18231) achieves calibration-free quantization via:
    1. Token-wise normalization: scale each vector to unit norm
    2. Channel-wise centering: subtract per-channel mean
    3. Second token-wise normalization: re-normalize to unit norm

After NSN, coordinates are approximately standard normal (KL 0.02 from
N(0,1) with Hadamard), which is exactly what Lloyd-Max codebooks are
optimized for. This makes NSN the ideal pre-processing step before
TurboQuant's rotation + quantization pipeline.

The full pipeline becomes: NSN -> Rotation -> Quantize -> Dequantize -> Unrotate -> Undo-NSN

Also includes:
    - KIVI-style key/value asymmetric quantization (per-channel keys, per-token values)
    - Group-wise quantization with hybrid symmetric/asymmetric per group
    - FP16 precision windows (recent + sink tokens at full precision)
    - Adaptive post-VQ scaling (angular preservation)

Ported from open_gororoba baselines/nsnquant.rs, baselines/kivi.rs, grouping.rs.
"""

import torch
from torch import Tensor
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
import math


# ---------------------------------------------------------------------------
# NSN Pre-processing
# ---------------------------------------------------------------------------

@dataclass
class NSNState:
    """State from NSN pre-processing needed for reconstruction."""
    norms_1: Tensor     # (n,) first normalization norms
    channel_means: Tensor  # (d,) per-channel means
    norms_2: Tensor     # (n,) second normalization norms


def nsn_preprocess(x: Tensor) -> Tuple[Tensor, NSNState]:
    """
    Apply Normalize-Shift-Normalize pre-processing.

    After NSN, each vector has unit norm and approximately zero per-channel
    mean. Combining with WHT rotation produces coordinates with KL divergence
    0.02 from standard normal -- ideal for Lloyd-Max quantization.

    Args:
        x: input vectors, shape (n, d)

    Returns:
        (preprocessed, state) where preprocessed has shape (n, d) and
        state contains metadata needed for nsn_restore().
    """
    # Step 1: Token-wise normalization
    norms_1 = x.norm(dim=-1, keepdim=True)  # (n, 1)
    x_n = x / (norms_1 + 1e-15)

    # Step 2: Channel-wise centering
    channel_means = x_n.mean(dim=0, keepdim=True)  # (1, d)
    x_ns = x_n - channel_means

    # Step 3: Second token-wise normalization
    norms_2 = x_ns.norm(dim=-1, keepdim=True)  # (n, 1)
    x_nsn = x_ns / (norms_2 + 1e-15)

    state = NSNState(
        norms_1=norms_1.squeeze(-1),
        channel_means=channel_means.squeeze(0),
        norms_2=norms_2.squeeze(-1),
    )
    return x_nsn, state


def nsn_restore(x_nsn: Tensor, state: NSNState) -> Tensor:
    """
    Reverse NSN pre-processing to recover original-scale vectors.

    Args:
        x_nsn: NSN-preprocessed vectors, shape (n, d)
        state: from nsn_preprocess()

    Returns:
        Restored vectors, shape (n, d).
    """
    # Reverse step 3: denormalize by norms_2
    x_ns = x_nsn * state.norms_2.unsqueeze(-1)
    # Reverse step 2: add channel means
    x_n = x_ns + state.channel_means.unsqueeze(0)
    # Reverse step 1: denormalize by norms_1
    x = x_n * state.norms_1.unsqueeze(-1)
    return x


# ---------------------------------------------------------------------------
# Adaptive Post-VQ Scaling (angular preservation)
# ---------------------------------------------------------------------------

def adaptive_vq_scale(x_original: Tensor, x_quantized: Tensor) -> Tensor:
    """
    Adaptive post-VQ scaling from NSNQuant.

    Corrects the quantized vector to preserve the component parallel to the
    original: v_Q = (||v||^2 / (v . v_Q)) * v_Q

    This preserves the projection onto the original direction, reducing
    angular error at the cost of slightly higher MSE.

    Args:
        x_original: original vectors, shape (..., d)
        x_quantized: quantized vectors, shape (..., d)

    Returns:
        Scaled quantized vectors, shape (..., d).
    """
    orig_norm_sq = (x_original ** 2).sum(dim=-1, keepdim=True)
    dot = (x_original * x_quantized).sum(dim=-1, keepdim=True)
    scale = orig_norm_sq / (dot + 1e-15)
    return x_quantized * scale


# ---------------------------------------------------------------------------
# KIVI Key/Value Asymmetric Quantization
# ---------------------------------------------------------------------------

def kivi_quantize_keys(
    keys: Tensor,
    bits: int = 2,
) -> Dict[str, Tensor]:
    """
    Per-channel key quantization (KIVI pattern).

    For each channel c, find min/max across all tokens, then quantize
    that channel uniformly. This handles persistent per-channel outliers.

    Args:
        keys: shape (seq_len, d) or (batch, heads, seq_len, d)
        bits: quantization bit-width

    Returns:
        Dict with indices, scales, zero_points for dequantization.
    """
    orig_shape = keys.shape
    if keys.dim() == 4:
        B, H, S, D = keys.shape
        keys_flat = keys.reshape(-1, D)
    else:
        keys_flat = keys
        S, D = keys.shape

    n_levels = 2 ** bits

    # Per-channel min/max (across all tokens)
    ch_min = keys_flat.min(dim=0).values  # (D,)
    ch_max = keys_flat.max(dim=0).values  # (D,)
    ch_range = ch_max - ch_min
    scales = ch_range / (n_levels - 1)
    scales = scales.clamp(min=1e-15)

    # Quantize
    indices = ((keys_flat - ch_min.unsqueeze(0)) / scales.unsqueeze(0)).round()
    indices = indices.clamp(0, n_levels - 1).to(torch.uint8)

    return {
        "indices": indices.reshape(orig_shape),
        "scales": scales,        # (D,) per-channel
        "zero_points": ch_min,   # (D,) per-channel
        "bits": bits,
        "axis": "per_channel",
    }


def kivi_dequantize_keys(compressed: Dict[str, Tensor]) -> Tensor:
    """Dequantize per-channel quantized keys."""
    indices = compressed["indices"].float()
    scales = compressed["scales"]
    zp = compressed["zero_points"]

    if indices.dim() == 4:
        # (B, H, S, D) -- broadcast scales (D,)
        return indices * scales + zp
    return indices * scales.unsqueeze(0) + zp.unsqueeze(0)


def kivi_quantize_values(
    values: Tensor,
    bits: int = 2,
) -> Dict[str, Tensor]:
    """
    Per-token value quantization (KIVI pattern).

    For each token t, find min/max across all dimensions, then quantize
    that token uniformly. Confines errors to individual tokens.

    Args:
        values: shape (seq_len, d) or (batch, heads, seq_len, d)
        bits: quantization bit-width

    Returns:
        Dict with indices, scales, zero_points for dequantization.
    """
    orig_shape = values.shape
    n_levels = 2 ** bits

    # Per-token min/max (across all dimensions)
    tok_min = values.min(dim=-1).values  # (..., seq_len) or (seq_len,)
    tok_max = values.max(dim=-1).values
    tok_range = tok_max - tok_min
    scales = tok_range / (n_levels - 1)
    scales = scales.clamp(min=1e-15)

    # Quantize
    indices = ((values - tok_min.unsqueeze(-1)) / scales.unsqueeze(-1)).round()
    indices = indices.clamp(0, n_levels - 1).to(torch.uint8)

    return {
        "indices": indices,
        "scales": scales,        # per-token
        "zero_points": tok_min,  # per-token
        "bits": bits,
        "axis": "per_token",
    }


def kivi_dequantize_values(compressed: Dict[str, Tensor]) -> Tensor:
    """Dequantize per-token quantized values."""
    indices = compressed["indices"].float()
    scales = compressed["scales"]
    zp = compressed["zero_points"]
    return indices * scales.unsqueeze(-1) + zp.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Group-wise Quantization
# ---------------------------------------------------------------------------

@dataclass
class GroupQuantParams:
    """Per-group quantization parameters."""
    scales: Tensor       # (n_groups,)
    zero_points: Tensor  # (n_groups,)
    symmetric: Tensor    # (n_groups,) bool
    group_size: int
    n_groups: int


def compute_group_params(
    values: Tensor,
    group_size: int = 128,
    bits: int = 4,
    symmetry_threshold: float = 0.1,
) -> GroupQuantParams:
    """
    Compute per-group quantization parameters with hybrid symmetric/asymmetric.

    For each group of consecutive coordinates:
        - If distribution is centered (|center| < threshold * range): symmetric
        - Otherwise: asymmetric with zero_point = min

    Args:
        values: shape (d,) -- single vector to parameterize
        group_size: coordinates per group
        bits: quantization bit-width
        symmetry_threshold: center/range threshold for symmetric mode

    Returns:
        GroupQuantParams with per-group scales and zero-points.
    """
    d = values.shape[-1]
    n_groups = (d + group_size - 1) // group_size
    n_levels = 2 ** bits

    scales = torch.zeros(n_groups, device=values.device)
    zero_points = torch.zeros(n_groups, device=values.device)
    symmetric = torch.zeros(n_groups, dtype=torch.bool, device=values.device)

    for g in range(n_groups):
        start = g * group_size
        end = min(start + group_size, d)
        group = values[..., start:end]

        g_min = group.min().item()
        g_max = group.max().item()
        g_range = g_max - g_min
        center = (g_min + g_max) / 2.0

        is_sym = abs(center) < symmetry_threshold * max(g_range, 1e-15)

        if is_sym:
            abs_max = max(abs(g_min), abs(g_max))
            scales[g] = max(2.0 * abs_max / (n_levels - 1), 1e-15)
            zero_points[g] = 0.0
            symmetric[g] = True
        else:
            scales[g] = max(g_range / (n_levels - 1), 1e-15)
            zero_points[g] = g_min
            symmetric[g] = False

    return GroupQuantParams(scales, zero_points, symmetric, group_size, n_groups)


def group_quantize(values: Tensor, params: GroupQuantParams, bits: int) -> Tensor:
    """Quantize a vector using group-wise parameters."""
    d = values.shape[-1]
    max_idx = (2 ** bits) - 1
    indices = torch.zeros(d, dtype=torch.uint8, device=values.device)

    for g in range(params.n_groups):
        start = g * params.group_size
        end = min(start + params.group_size, d)
        v = values[..., start:end]
        s = params.scales[g].item()
        zp = params.zero_points[g].item()

        if params.symmetric[g]:
            half = max_idx / 2.0
            idx = ((v / s) + half).round().clamp(0, max_idx)
        else:
            idx = ((v - zp) / s).round().clamp(0, max_idx)
        indices[start:end] = idx.to(torch.uint8)

    return indices


def group_dequantize(indices: Tensor, params: GroupQuantParams, bits: int) -> Tensor:
    """Dequantize from group-wise parameters."""
    d = indices.shape[-1]
    max_idx = (2 ** bits) - 1
    values = torch.zeros(d, device=indices.device, dtype=torch.float32)

    for g in range(params.n_groups):
        start = g * params.group_size
        end = min(start + params.group_size, d)
        idx = indices[start:end].float()
        s = params.scales[g].item()
        zp = params.zero_points[g].item()

        if params.symmetric[g]:
            half = max_idx / 2.0
            values[start:end] = (idx - half) * s
        else:
            values[start:end] = idx * s + zp

    return values


# ---------------------------------------------------------------------------
# FP16 Precision Windows
# ---------------------------------------------------------------------------

@dataclass
class PrecisionWindows:
    """
    High-precision token windows for KV cache.

    Sink tokens (initial positions) and recent tokens are kept at full
    FP16 precision. Middle tokens are quantized. This preserves quality
    for the tokens that most influence attention.

    Default: 128 recent + 4 sink (from InnerQ/KIVI).
    """
    recent_window: int = 128
    sink_window: int = 4

    def should_quantize(self, pos: int, seq_len: int) -> bool:
        """Whether a token at position pos should be quantized."""
        if pos < self.sink_window:
            return False
        if seq_len > self.recent_window and pos >= seq_len - self.recent_window:
            return False
        return True

    def quantize_mask(self, seq_len: int, device: str = "cpu") -> Tensor:
        """
        Boolean mask: True for positions that should be quantized.

        Args:
            seq_len: total sequence length
            device: torch device

        Returns:
            Boolean tensor, shape (seq_len,).
        """
        mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        mask[:self.sink_window] = False
        if seq_len > self.recent_window:
            mask[seq_len - self.recent_window:] = False
        return mask

    def count_quantized(self, seq_len: int) -> Tuple[int, int]:
        """Return (n_quantized, n_full_precision)."""
        mask = self.quantize_mask(seq_len)
        n_quant = mask.sum().item()
        return n_quant, seq_len - n_quant

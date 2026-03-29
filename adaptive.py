"""
Adaptive per-head bit allocation for TurboQuant.

Inspired by steinmarder's 24-precision LBM tier selection, where each
precision tier is chosen based on the dynamic range and sensitivity of
the physical quantity being simulated.

For LLM attention, different heads and layers have different sensitivity
to quantization. High-entropy heads (uniform attention) are less sensitive;
concentrated attention patterns require higher precision to preserve the
sharp peaks that drive generation quality.

Strategy:
    1. Profile: run a calibration pass to measure per-head sensitivity
    2. Allocate: greedily assign bits from a budget to maximize quality
    3. Configure: assign (bits, rotation_type, quantizer_type) per head/layer
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class QuantConfig:
    """Quantization configuration for a single head/layer."""
    bits: int = 3
    rotation: str = "haar"
    quantizer: str = "scalar"


@dataclass
class HeadSensitivity:
    """Sensitivity metrics for a single attention head."""
    layer: int
    head: int
    attention_entropy: float  # higher = more uniform = less sensitive
    weight_magnitude: float   # higher = more important
    sensitivity_score: float  # combined score (higher = needs more bits)


class AdaptiveBitAllocator:
    """
    Assigns per-head quantization configurations from a fixed bit budget.

    Usage:
        allocator = AdaptiveBitAllocator(total_budget_bits=3.0)
        sensitivities = allocator.profile(model, calibration_data)
        configs = allocator.allocate(sensitivities)
        # configs[("layer_0", "head_3")] -> QuantConfig(bits=4, ...)
    """

    def __init__(
        self,
        total_budget_bits: float = 3.0,
        min_bits: int = 1,
        max_bits: int = 4,
    ):
        """
        Args:
            total_budget_bits: average bits per dimension across all heads
            min_bits: minimum bits for any head
            max_bits: maximum bits for any head
        """
        self.total_budget = total_budget_bits
        self.min_bits = min_bits
        self.max_bits = max_bits

    @torch.no_grad()
    def profile_attention(
        self,
        attention_weights: Tensor,
    ) -> List[HeadSensitivity]:
        """
        Profile attention sensitivity from precomputed attention weights.

        Args:
            attention_weights: shape (n_layers, n_heads, seq_q, seq_k)

        Returns:
            List of HeadSensitivity, sorted by sensitivity (highest first).
        """
        n_layers, n_heads = attention_weights.shape[:2]
        sensitivities = []

        for layer in range(n_layers):
            for head in range(n_heads):
                attn = attention_weights[layer, head]  # (seq_q, seq_k)

                # Entropy: H = -sum(p * log(p))
                # Clamp to avoid log(0)
                p = attn.clamp(min=1e-10)
                entropy = -(p * p.log()).sum(dim=-1).mean().item()

                # Normalize entropy by log(seq_k) to get [0, 1] range
                max_entropy = math.log(attn.shape[-1])
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

                # Weight magnitude proxy: max attention value
                peak_attention = attn.max(dim=-1).values.mean().item()

                # Sensitivity: low entropy (concentrated) + high peaks = sensitive
                # Score: higher = needs more bits
                sensitivity_score = (1.0 - normalized_entropy) * peak_attention

                sensitivities.append(HeadSensitivity(
                    layer=layer,
                    head=head,
                    attention_entropy=normalized_entropy,
                    weight_magnitude=peak_attention,
                    sensitivity_score=sensitivity_score,
                ))

        sensitivities.sort(key=lambda s: s.sensitivity_score, reverse=True)
        return sensitivities

    def allocate(
        self,
        sensitivities: List[HeadSensitivity],
    ) -> Dict[Tuple[int, int], QuantConfig]:
        """
        Greedy bit allocation: assign more bits to more sensitive heads.

        Args:
            sensitivities: list from profile_attention()

        Returns:
            Dict mapping (layer, head) -> QuantConfig.
        """
        n_heads = len(sensitivities)
        if n_heads == 0:
            return {}

        # Start everyone at min_bits
        assignments = {
            (s.layer, s.head): self.min_bits for s in sensitivities
        }

        # Total budget = n_heads * total_budget_bits
        total_budget = int(n_heads * self.total_budget)
        current_total = n_heads * self.min_bits

        # Greedily add bits to the most sensitive heads
        sorted_by_sensitivity = sorted(
            sensitivities, key=lambda s: s.sensitivity_score, reverse=True
        )

        for s in sorted_by_sensitivity:
            key = (s.layer, s.head)
            while assignments[key] < self.max_bits and current_total < total_budget:
                assignments[key] += 1
                current_total += 1

        # Convert to QuantConfig with mixed quantizer selection
        configs = {}
        for (layer, head), bits in assignments.items():
            if bits >= 4:
                config = QuantConfig(bits=bits, rotation="haar", quantizer="scalar")
            elif bits == 3:
                config = QuantConfig(bits=bits, rotation="wht", quantizer="e8")
            elif bits == 2:
                config = QuantConfig(bits=bits, rotation="wht", quantizer="z8_256")
            else:
                config = QuantConfig(bits=bits, rotation="wht", quantizer="z8_32")
            configs[(layer, head)] = config

        return configs

    def summary(
        self,
        configs: Dict[Tuple[int, int], QuantConfig],
    ) -> Dict[str, float]:
        """Compute summary statistics for an allocation."""
        if not configs:
            return {}
        bits_list = [c.bits for c in configs.values()]
        return {
            "n_heads": len(configs),
            "mean_bits": sum(bits_list) / len(bits_list),
            "min_bits": min(bits_list),
            "max_bits": max(bits_list),
            "bits_histogram": {
                b: bits_list.count(b) for b in sorted(set(bits_list))
            },
        }


def allocate_per_layer_bits(
    cache,
    total_budget: float = 3.0,
    min_bits: int = 2,
    max_bits: int = 4,
) -> List[int]:
    """
    Calibration-free per-layer bit allocation based on key variance.

    Layers with low key variance are harder to quantize (proportionally
    more noise) and get more bits. Layers with high key variance have
    more room for quantization error and get fewer bits.

    Derived from per-layer analysis on Qwen2.5-3B (36 layers):
    - L22 (worst cosine 0.954): lowest variance (2.64)
    - L0 (best cosine 0.976): highest variance (236)
    - Correlation(cosine, 1/variance) = -0.558

    Args:
        cache: DynamicCache from model forward pass
        total_budget: average bits across all layers
        min_bits: floor
        max_bits: ceiling

    Returns:
        List of per-layer bit allocations.
    """
    n_layers = len(cache.layers) if hasattr(cache, 'layers') else len(cache)
    variances = []

    for li in range(n_layers):
        if hasattr(cache, 'layers'):
            keys = cache.layers[li].keys.float()
        else:
            keys = cache[li][0].float()
        variances.append(keys.var().item())

    # Lower variance = harder to quantize = needs more bits
    # Score = 1/variance (normalized)
    inv_var = [1.0 / (v + 1e-8) for v in variances]
    total_inv = sum(inv_var)

    # Allocate: distribute budget proportional to difficulty
    # Start everyone at min_bits, then distribute extra budget
    bits = [min_bits] * n_layers
    extra_budget = int(total_budget * n_layers) - min_bits * n_layers

    # Sort layers by difficulty (hardest first)
    difficulty_order = sorted(range(n_layers), key=lambda i: inv_var[i], reverse=True)

    for li in difficulty_order:
        while bits[li] < max_bits and extra_budget > 0:
            bits[li] += 1
            extra_budget -= 1

    return bits

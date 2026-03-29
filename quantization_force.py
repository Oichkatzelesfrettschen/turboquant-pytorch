"""
Quantization Force: unified bit allocation via rate-distortion Lagrange optimization.

Inspired by dimensional analysis from the Pais Superforce (S_F = c^4/G):
Einstein's field equation couples geometry to energy via a single constant.
Analogously, quantization couples information to distortion via a single
"quantization force" that defines the natural scale for bit allocation.

    Einstein:      G_uv = (1/S_F) * T_uv
    Quantization:  I(x;x_hat) = (1/F_Q) * D(b, sigma)

For Lloyd-Max on a Gaussian source with variance sigma^2 at b bits:
    D(b, sigma^2) = sigma^2 * C * 4^{-b}     (C = pi*sqrt(3)/2)
    F_Q = b * ln(2) * 4^b / (sigma^2 * C)    (grows exponentially with bits)

The Lagrange-optimal bit allocation across regions:
    b_r = b_mean + (1/(2*ln(4))) * ln(sigma_r^2 / sigma_mean^2)

High-variance regions get more bits, but only logarithmically.
This is the closed-form reverse water-filling solution from
rate-distortion theory, replacing ad-hoc greedy heuristics.

Optional CD structure factor for Cayley-Dickson tower levels:
    S_r = 1 + alpha * ||associator_r|| / ||x_r||
    (corrects for zero-divisor noise amplification at dim >= 16)
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor


# Lloyd-Max distortion constant for Gaussian source
_LLOYD_MAX_C = math.pi * math.sqrt(3) / 2
_INV_2LN4 = 1.0 / (2 * math.log(4))


@dataclass
class RegionStats:
    """Statistics for one quantization region (layer, tower level, coordinate group)."""
    index: int
    n_elements: int     # number of scalar elements in this region
    variance: float     # per-element variance (sigma^2)
    structure_factor: float = 1.0  # CD correction (1.0 = no correction)


def lloyd_max_distortion(bits: float, variance: float) -> float:
    """Lloyd-Max MSE distortion for b bits on Gaussian source with given variance."""
    return variance * _LLOYD_MAX_C * (4.0 ** (-bits))


def quantization_force(bits: float, variance: float) -> float:
    """
    Compute the quantization force F_Q = I / D.

    By analogy with Planck force S_F = c^4/G (the natural scale in Einstein's
    equation), F_Q is the natural scale in rate-distortion theory. It measures
    the "force" with which a quantizer preserves information against distortion.

    F_Q grows exponentially with bits: each additional bit provides
    exponentially more preservation force, just as the Planck force
    emerges from the extreme ratio of c^4 to G.

    Args:
        bits: quantization bit-width
        variance: per-element source variance

    Returns:
        F_Q in units of [information / distortion] = [bits / MSE].
    """
    if bits <= 0 or variance <= 0:
        return 0.0
    info = bits * math.log(2)
    distortion = lloyd_max_distortion(bits, variance)
    return info / distortion


def lagrange_optimal_allocation(
    regions: list[RegionStats],
    total_budget: float,
    min_bits: int = 2,
    max_bits: int = 8,
) -> list[int]:
    """
    Closed-form Lagrange-optimal bit allocation across regions.

    Minimizes total distortion sum_r D_r(b_r) subject to the
    constraint sum_r n_r * b_r <= total_budget * sum_r n_r.

    The first-order KKT condition gives:
        b_r = b_mean + (1/(2*ln(4))) * ln(v_eff_r / v_eff_mean)

    where v_eff_r = structure_factor_r * variance_r.

    After computing the continuous optimal, we round to integers
    and iteratively redistribute residual budget from clipping.

    Args:
        regions: list of RegionStats describing each region
        total_budget: average bits per element across all regions
        min_bits: minimum bits for any region
        max_bits: maximum bits for any region

    Returns:
        List of integer bit allocations, one per region.
    """
    n = len(regions)
    if n == 0:
        return []

    # Effective variance = structure_factor * variance
    v_eff = [max(r.structure_factor * r.variance, 1e-30) for r in regions]

    # Weighted geometric mean (weighted by n_elements)
    total_elements = sum(r.n_elements for r in regions)
    if total_elements == 0:
        return [min_bits] * n

    # Log-space mean: ln(v_mean) = sum(n_r * ln(v_r)) / sum(n_r)
    log_v_mean = sum(r.n_elements * math.log(v) for r, v in zip(regions, v_eff)) / total_elements

    # Continuous optimal allocation
    b_continuous = []
    for r, v in zip(regions, v_eff):
        b_r = total_budget + _INV_2LN4 * (math.log(v) - log_v_mean)
        b_continuous.append(b_r)

    # Round and clip
    bits = [max(min_bits, min(max_bits, round(b))) for b in b_continuous]

    # Redistribute residual budget from clipping (max 3 iterations)
    for _ in range(3):
        used = sum(r.n_elements * b for r, b in zip(regions, bits))
        target = int(total_budget * total_elements)
        residual = target - used

        if residual == 0:
            break

        # Find unconstrained regions (not at min or max)
        adjustable = [
            i for i in range(n)
            if (residual > 0 and bits[i] < max_bits)
            or (residual < 0 and bits[i] > min_bits)
        ]

        if not adjustable:
            break

        # Sort by how far each region is from its continuous optimum
        # (prioritize regions that were most distorted by rounding)
        adjustable.sort(
            key=lambda i: abs(b_continuous[i] - bits[i]) * regions[i].n_elements,
            reverse=True,
        )

        for i in adjustable:
            if residual == 0:
                break
            step = 1 if residual > 0 else -1
            new_b = bits[i] + step
            if min_bits <= new_b <= max_bits:
                cost = regions[i].n_elements * step
                if abs(residual) >= abs(cost):
                    bits[i] = new_b
                    residual -= cost

    return bits


def compute_structure_factor(
    data: Tensor,
    block_dim: int = 16,
    n_triplets: int = 100,
    alpha: float = 1.0,
) -> float:
    """
    Compute the CD structure factor for a data block.

    For block_dim >= 16 (sedenions and beyond), zero divisors amplify
    quantization noise. The structure factor S > 1 increases the
    effective variance, causing the Lagrange allocator to assign
    more bits to these regions.

    For block_dim <= 8 (composition algebras), returns 1.0.

    Args:
        data: tensor of shape (n, d) where d >= block_dim
        block_dim: CD algebra dimension (4, 8, 16, 32, ...)
        n_triplets: number of random triplets to sample
        alpha: scaling coefficient for associator contribution

    Returns:
        Structure factor S >= 1.0.
    """
    if block_dim <= 8:
        return 1.0  # composition algebras are isometric, no ZD noise

    from .cd_algebra import cd_associator_norm

    n, d = data.shape
    if d < block_dim or n < 3:
        return 1.0

    # Sample random triplets from the data projected to block_dim
    blocks = data[:, :block_dim]
    idx = torch.randint(0, n, (n_triplets, 3))
    a = blocks[idx[:, 0]]
    b = blocks[idx[:, 1]]
    c = blocks[idx[:, 2]]

    assoc_norms = cd_associator_norm(a, b, c)
    data_norms = blocks.norm(dim=-1).mean()

    mean_assoc = assoc_norms.mean().item()
    mean_norm = max(data_norms.item(), 1e-8)

    return 1.0 + alpha * mean_assoc / mean_norm


def unified_bit_allocation(
    variances: list[float],
    total_budget: float,
    n_elements: list[int] | None = None,
    structure_factors: list[float] | None = None,
    min_bits: int = 2,
    max_bits: int = 8,
) -> list[int]:
    """
    Convenience wrapper: Lagrange-optimal allocation from variance list.

    Replaces:
    - adaptive.py allocate_per_layer_bits() (per-layer variance heuristic)
    - hierarchical.py allocate_bits_to_levels() (greedy marginal)
    - spectral.py spectral_bit_allocation() (log-energy proportional)

    All three are unified under the same Lagrange variational principle.

    Args:
        variances: per-region variance (sigma^2)
        total_budget: average bits per element
        n_elements: elements per region (default: all equal = 1)
        structure_factors: CD correction per region (default: all 1.0)
        min_bits: floor
        max_bits: ceiling

    Returns:
        Integer bit allocation per region.
    """
    n = len(variances)
    if n_elements is None:
        n_elements = [1] * n
    if structure_factors is None:
        structure_factors = [1.0] * n

    regions = [
        RegionStats(
            index=i,
            n_elements=n_elements[i],
            variance=variances[i],
            structure_factor=structure_factors[i],
        )
        for i in range(n)
    ]

    return lagrange_optimal_allocation(regions, total_budget, min_bits, max_bits)

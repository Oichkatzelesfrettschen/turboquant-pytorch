"""
Spectral analysis tools for quantization.

Provides distribution analysis and frequency-domain quantization strategies
informed by open_gororoba's spectral_core module. After rotation, coordinates
should be approximately i.i.d. Gaussian -- deviations from this indicate the
rotation isn't fully decorrelating, and adaptive bit allocation helps.

Functions:
    coordinate_spectrum:     FFT of per-coordinate distributions
    spectral_bit_allocation: JPEG-style frequency-adaptive bit assignment
    distribution_analysis:   Per-coordinate variance, kurtosis, entropy
"""

import math
from typing import Dict

import torch
from torch import Tensor


def distribution_analysis(x: Tensor) -> Dict[str, Tensor]:
    """
    Compute per-coordinate distribution statistics.

    After an ideal rotation, all coordinates should have:
        variance = 1/d (for unit vectors)
        kurtosis = 3.0 (Gaussian)
        entropy ~ 0.5 * log(2*pi*e*sigma^2)

    Deviations indicate imperfect decorrelation or non-Gaussian tails.

    Args:
        x: rotated vectors, shape (n, d)

    Returns:
        Dict with:
            variance: per-coordinate variance, shape (d,)
            kurtosis: per-coordinate excess kurtosis, shape (d,)
            skewness: per-coordinate skewness, shape (d,)
            entropy_estimate: per-coordinate Gaussian entropy estimate, shape (d,)
            cross_correlation: adjacent coordinate correlation, shape (d-1,)
    """
    n, d = x.shape
    mean = x.mean(dim=0)
    centered = x - mean

    variance = centered.var(dim=0)
    std = variance.sqrt().clamp(min=1e-10)
    normalized = centered / std

    skewness = (normalized ** 3).mean(dim=0)
    kurtosis = (normalized ** 4).mean(dim=0) - 3.0  # excess kurtosis

    # Gaussian entropy: 0.5 * log(2*pi*e*sigma^2)
    entropy_estimate = 0.5 * torch.log(2 * math.pi * math.e * variance.clamp(min=1e-20))

    # Adjacent coordinate correlation
    cross_corr = torch.zeros(d - 1, device=x.device)
    for i in range(d - 1):
        corr_matrix = torch.corrcoef(x[:, i:i + 2].T)
        cross_corr[i] = corr_matrix[0, 1].abs()

    return {
        "variance": variance,
        "kurtosis": kurtosis,
        "skewness": skewness,
        "entropy_estimate": entropy_estimate,
        "cross_correlation": cross_corr,
    }


def coordinate_spectrum(x: Tensor) -> Tensor:
    """
    Compute the FFT power spectrum of per-coordinate distributions.

    Each coordinate's marginal distribution is treated as a signal.
    Periodic patterns in the spectrum indicate structured (non-random)
    dependencies between samples, which may affect quantization.

    Args:
        x: rotated vectors, shape (n, d)

    Returns:
        Power spectrum, shape (d, n//2 + 1).
    """
    # FFT along the sample dimension for each coordinate
    X = torch.fft.rfft(x, dim=0)
    power = (X.real ** 2 + X.imag ** 2) / x.shape[0]
    return power.T  # (d, n//2 + 1)


def spectral_energy(x: Tensor) -> Tensor:
    """
    Compute the spectral energy per coordinate after block-FFT.

    Treats groups of coordinates as frequency bins (like JPEG).
    High-energy coordinates carry more information and should get more bits.

    Args:
        x: rotated vectors, shape (n, d)

    Returns:
        Per-coordinate energy, shape (d,).
    """
    return (x ** 2).mean(dim=0)


def spectral_bit_allocation(
    x: Tensor,
    total_bits: int,
    min_bits: int = 1,
    max_bits: int = 8,
) -> Tensor:
    """
    Per-coordinate bit allocation via Lagrange-optimal rate-distortion.

    Uses the quantization force framework: the closed-form Lagrange solution
    b_r = b_mean + (1/(2*ln(4))) * ln(sigma_r^2 / sigma_mean^2) replaces
    the previous ad-hoc log-energy proportional heuristic.

    For independent Gaussian coordinates (post-rotation), this IS the
    optimal reverse water-filling solution from rate-distortion theory.

    Args:
        x: rotated vectors, shape (n, d)
        total_bits: total bit budget across all dimensions (= d * avg_bits)
        min_bits: minimum bits for any coordinate
        max_bits: maximum bits for any coordinate

    Returns:
        Per-coordinate bit allocation, shape (d,), integer values.
    """
    from .quantization_force import unified_bit_allocation

    d = x.shape[-1]
    energy = spectral_energy(x)
    variances = energy.tolist()  # per-coord variance = per-coord energy for mean-0

    avg_bits = total_bits / d
    bits_list = unified_bit_allocation(
        variances, avg_bits, min_bits=min_bits, max_bits=max_bits,
    )

    return torch.tensor(bits_list, dtype=torch.int32, device=x.device)


def rotation_quality_score(x_original: Tensor, x_rotated: Tensor) -> Dict[str, float]:
    """
    Score the quality of a rotation for quantization purposes.

    A perfect rotation produces coordinates that are:
        1. Uniform variance (low variance ratio)
        2. Zero cross-correlation
        3. Gaussian (zero excess kurtosis)
        4. Norm-preserving (isometry)

    Args:
        x_original: original vectors, shape (n, d)
        x_rotated: rotated vectors, shape (n, d)

    Returns:
        Dict with quality scores (lower = better for all metrics).
    """
    d = x_rotated.shape[-1]
    stats = distribution_analysis(x_rotated)

    # Variance uniformity: std of per-coordinate variances (ideal: 0)
    var_uniformity = stats["variance"].std().item()

    # Cross-correlation: mean absolute adjacent correlation (ideal: 0)
    mean_cross_corr = stats["cross_correlation"].mean().item()

    # Gaussianity: mean absolute excess kurtosis (ideal: 0)
    mean_kurtosis = stats["kurtosis"].abs().mean().item()

    # Isometry: relative norm change (ideal: 0)
    orig_norms = x_original.norm(dim=-1)
    rot_norms = x_rotated.norm(dim=-1)
    isometry_error = ((rot_norms / (orig_norms + 1e-8)) - 1.0).abs().mean().item()

    # Combined score (weighted geometric mean of quality metrics)
    combined = (
        var_uniformity * 100 +
        mean_cross_corr * 100 +
        mean_kurtosis * 10 +
        isometry_error * 1000
    )

    return {
        "variance_uniformity": var_uniformity,
        "mean_cross_correlation": mean_cross_corr,
        "mean_excess_kurtosis": mean_kurtosis,
        "isometry_error": isometry_error,
        "combined_score": combined,
    }

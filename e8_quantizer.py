"""
E8 lattice quantizer using the Conway-Sloane closest-lattice-point algorithm.

The E8 lattice is the unique even unimodular lattice in 8 dimensions and
achieves the densest sphere packing in R^8 (Viazovska 2016). As a quantizer,
it has normalized second moment G(E8) = 0.07168, compared to G(Z) = 1/12 =
0.08333 for scalar quantization -- a 14% MSE improvement at the same bit rate.

Construction: E8 = D8 union (D8 + h), where:
    D8 = { x in Z^8 : sum(x_i) is even }
    h  = (1/2, 1/2, ..., 1/2)

The closest-point algorithm:
    1. Round input to nearest D8 point (nearest integer coords with even sum)
    2. Round input to nearest D8+h point (nearest half-integer coords with even sum)
    3. Return whichever is closer

This is O(d) = O(8) per vector -- constant time, far faster than brute-force
codebook search.

The E8 lattice is the same lattice used by QuIP# (Tseng et al., 2024) for
weight quantization. Here we adapt it for KV cache vector quantization.

Reference implementation: open_gororoba/crates/gororoba_algebra/src/lie/e8_lattice.rs.
"""

import math
import torch
from torch import Tensor
from typing import Tuple


def _round_to_d8(x: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Round to nearest D8 lattice point: integer coords with even coordinate sum.

    Strategy: round each coordinate to nearest integer, then if the sum is odd,
    flip the coordinate whose rounding introduced the most error.

    Fully vectorized -- no Python loops over batch elements.

    Args:
        x: input vectors, shape (..., 8)

    Returns:
        (lattice_point, squared_distance) where lattice_point has shape (..., 8)
        and squared_distance has shape (...).
    """
    rounded = x.round()
    residual = x - rounded

    # Check if coordinate sum is even
    coord_sum = rounded.sum(dim=-1)
    is_odd = (coord_sum % 2 != 0)

    if is_odd.any():
        # For odd-sum points: flip the coordinate with largest |residual|
        abs_residual = residual.abs()
        flip_idx = abs_residual.argmax(dim=-1)  # (...)

        # Vectorized correction: determine flip direction per element
        flat_rounded = rounded.reshape(-1, 8)
        flat_residual = residual.reshape(-1, 8)
        flat_is_odd = is_odd.reshape(-1)
        flat_flip_idx = flip_idx.reshape(-1)
        n = flat_rounded.shape[0]

        # Gather the residual at the flip index for each row
        row_idx = torch.arange(n, device=x.device)
        flip_residual = flat_residual[row_idx, flat_flip_idx]

        # Direction: -1 if residual >= 0 (round down), +1 if < 0 (round up)
        direction = torch.where(flip_residual >= 0, -1.0, 1.0)

        # Build correction matrix: only modify the flip_idx coordinate, only for odd rows
        correction = torch.zeros_like(flat_rounded)
        correction[row_idx, flat_flip_idx] = direction * flat_is_odd.float()

        rounded = (flat_rounded + correction).reshape(x.shape)

    dist_sq = ((x - rounded) ** 2).sum(dim=-1)
    return rounded, dist_sq


def _round_to_d8_half(x: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Round to nearest D8 + (1/2,...,1/2) coset point.

    These are half-integer vectors whose coordinate sum is even (integer).

    Strategy: shift by -1/2, round to D8, shift back by +1/2.

    Args:
        x: input vectors, shape (..., 8)

    Returns:
        (lattice_point, squared_distance)
    """
    shifted = x - 0.5
    d8_point, _ = _round_to_d8(shifted)
    coset_point = d8_point + 0.5
    dist_sq = ((x - coset_point) ** 2).sum(dim=-1)
    return coset_point, dist_sq


def e8_closest_point(x: Tensor) -> Tensor:
    """
    Find the closest E8 lattice point to each input vector.

    E8 = D8 union (D8 + h) where h = (1/2, ..., 1/2).
    We round to both components and pick the closer one.

    Args:
        x: input vectors, shape (..., 8)

    Returns:
        Closest E8 lattice points, shape (..., 8).
    """
    d8_point, d8_dist = _round_to_d8(x)
    half_point, half_dist = _round_to_d8_half(x)

    # Pick whichever is closer
    use_half = half_dist < d8_dist  # (...) bool
    use_half_expanded = use_half.unsqueeze(-1).expand_as(x)
    return torch.where(use_half_expanded, half_point, d8_point)


def e8_quantize(
    x: Tensor,
    scale: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """
    Quantize 8D blocks to E8 lattice points.

    Args:
        x: input vectors, shape (..., 8)
        scale: scaling factor. Input is divided by scale before rounding,
               and the lattice point is multiplied by scale after.

    Returns:
        (lattice_points, indices) where lattice_points has shape (..., 8)
        and indices encode the lattice point for storage. Since E8 points
        have constrained structure, they can be encoded more compactly than
        arbitrary 8D vectors, but for now we return the points directly.
    """
    x_scaled = x / (scale + 1e-8)
    lattice_points = e8_closest_point(x_scaled)
    return lattice_points * scale, lattice_points


def e8_dequantize(
    lattice_points: Tensor,
    scale: float = 1.0,
) -> Tensor:
    """
    Dequantize E8 lattice points back to vectors.

    Args:
        lattice_points: E8 lattice points, shape (..., 8)
        scale: same scaling factor used in e8_quantize

    Returns:
        Reconstructed vectors, shape (..., 8).
    """
    return lattice_points * scale


def e8_auto_scale(x: Tensor, method: str = "mse_optimal") -> float:
    """
    Compute the scaling factor for E8 quantization.

    Three methods available:
        "mse_optimal": Minimize MSE via the normalized second moment formula.
            For a lattice with NSM G in n dimensions, and input with per-component
            variance sigma^2, the MSE-optimal scale s satisfies:
                MSE = n * G * s^2
            where s is the lattice spacing (= 1 for the standard E8 lattice).
            The input should be scaled so that it "fills" the Voronoi cell.
            Optimal: scale = sigma * (V_n)^{1/n} where V_n = det(E8)^{1/2} = 1.

            For post-rotation unit vectors with per-coord variance 1/d,
            sigma = 1/sqrt(d). The E8 lattice has minimum distance sqrt(2),
            so the fundamental Voronoi cell has inradius = sqrt(2)/2.
            Optimal scale = sigma / inradius_eff, tuned empirically.

        "grid_search": Try multiple scales on a sample and pick the best MSE.
            Slower but adapts to any input distribution.

        "heuristic": Fast RMS-based estimate (original method).

    Args:
        x: input vectors, shape (..., 8)
        method: "mse_optimal", "grid_search", or "heuristic"

    Returns:
        Optimal scale factor.
    """
    if method == "heuristic":
        x_rms = x.pow(2).mean().sqrt().item()
        return x_rms * 2.0

    if method == "grid_search":
        return _e8_grid_search_scale(x)

    # method == "mse_optimal"
    # The E8 lattice has:
    #   - Normalized second moment G(E8) = 0.07168
    #   - Minimum distance d_min = sqrt(2)
    #   - Voronoi cell volume V = 1 (det of generator matrix)
    #   - Covering radius rho = sqrt(2)
    #
    # For input with per-component variance sigma^2:
    #   MSE(scale) = G * scale^2 + (terms from points outside the cell)
    # The optimal scale balances quantization granularity against clipping.
    # For Gaussian input: scale ~ sigma * sqrt(n) / (d_min / 2)
    # = sigma * sqrt(8) / (sqrt(2)/2) = sigma * 4.0
    #
    # Empirical refinement: scale down by a factor that accounts for
    # the Gaussian tail extending beyond the Voronoi cell.
    sigma = x.pow(2).mean().sqrt().item()
    # Gauss-to-lattice mapping: the optimal scale places ~95% of probability
    # mass within the nearest Voronoi cell. For 8D Gaussian, the 95th
    # percentile radius is chi(8, 0.95) ~ 3.49. The E8 packing radius is
    # sqrt(2)/2 ~ 0.707. So: scale = sigma * 3.49 / 0.707 ~ sigma * 4.94
    # Clamped to practical range.
    return sigma * 4.0


def _e8_grid_search_scale(x: Tensor, n_trials: int = 20) -> float:
    """
    Find MSE-optimal scale by grid search over candidate values.

    Tests scales from 0.5x to 8x the RMS-based estimate and picks
    the one with lowest quantization MSE on the input data.

    Args:
        x: input vectors, shape (..., 8)
        n_trials: number of scale candidates to test

    Returns:
        Best scale factor.
    """
    x_flat = x.reshape(-1, 8)
    rms = x_flat.pow(2).mean().sqrt().item()
    if rms < 1e-10:
        return 1.0

    best_scale = rms * 4.0
    best_mse = float("inf")

    for i in range(n_trials):
        # Log-uniform search from 0.5*rms to 8*rms
        scale = rms * (0.5 * (16.0 ** (i / (n_trials - 1))))
        scaled = x_flat / scale
        lattice_pts = e8_closest_point(scaled)
        reconstructed = lattice_pts * scale
        mse = ((x_flat - reconstructed) ** 2).mean().item()
        if mse < best_mse:
            best_mse = mse
            best_scale = scale

    return best_scale


def generate_e8_roots() -> Tensor:
    """
    Generate all 240 E8 roots (vectors of squared length 2).

    The roots are:
    1. 112 roots: (+/-1, +/-1, 0, 0, 0, 0, 0, 0) and permutations
    2. 128 roots: (+/-1/2, ..., +/-1/2) with even number of minus signs

    Returns:
        Tensor of shape (240, 8).
    """
    roots = []

    # Type 1: two +/-1 coordinates, rest 0
    for i in range(8):
        for j in range(i + 1, 8):
            for si in [-1.0, 1.0]:
                for sj in [-1.0, 1.0]:
                    r = [0.0] * 8
                    r[i] = si
                    r[j] = sj
                    roots.append(r)

    # Type 2: all +/-1/2 with even number of minus signs
    for pattern in range(256):
        minus_count = bin(pattern).count("1")
        if minus_count % 2 == 0:
            r = [0.5] * 8
            for bit in range(8):
                if (pattern >> bit) & 1:
                    r[bit] = -0.5
            roots.append(r)

    result = torch.tensor(roots, dtype=torch.float32)
    assert result.shape == (240, 8), f"Expected 240 roots, got {result.shape[0]}"
    return result


def e8_normalized_second_moment() -> float:
    """
    Return the normalized second moment G(E8) = 0.07168.

    This is the fundamental figure of merit for lattice quantizers.
    Compare: G(Z) = 1/12 = 0.08333, G(D4) = 0.07655.
    E8 achieves the best known value in 8 dimensions.

    The MSE ratio between E8 and scalar quantization at the same bit rate is:
        MSE(E8) / MSE(Z) = G(E8) / G(Z) = 0.07168 / 0.08333 = 0.860

    So E8 gives a 14% MSE reduction.
    """
    return 0.07168


def e8_bits_per_dim(shells: int = 1) -> float:
    """
    Bits per dimension for E8 quantization with a given number of shells.

    Shell 0 (origin): 1 point
    Shell 1 (roots): 240 points at distance sqrt(2)
    Shell 2: 2160 points at distance 2
    ...

    For 1 shell (240+1 = 241 points): log2(241)/8 = 0.99 bits/dim
    For 2 shells (241+2160 = 2401): log2(2401)/8 = 1.41 bits/dim

    Args:
        shells: number of shells (1 = just the roots + origin)

    Returns:
        Bits per dimension.
    """
    # Known shell sizes for E8
    shell_sizes = [1, 240, 2160, 6720]
    total = sum(shell_sizes[:shells + 1])
    return math.log2(total) / 8

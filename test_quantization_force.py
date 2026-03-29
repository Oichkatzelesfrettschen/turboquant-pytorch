"""Unit tests for quantization_force.py: Lagrange-optimal bit allocation."""
import math
import torch
import pytest
from turboquant.quantization_force import (
    RegionStats, quantization_force, lagrange_optimal_allocation,
    lloyd_max_distortion, compute_structure_factor, unified_bit_allocation,
)


class TestLloydMaxDistortion:
    """Verify distortion formula matches known values."""

    def test_distortion_decreases_with_bits(self):
        v = 1.0
        d_prev = float("inf")
        for b in range(1, 6):
            d = lloyd_max_distortion(b, v)
            assert d < d_prev
            d_prev = d

    def test_distortion_scales_with_variance(self):
        d1 = lloyd_max_distortion(3, 1.0)
        d2 = lloyd_max_distortion(3, 4.0)
        assert abs(d2 / d1 - 4.0) < 1e-10

    def test_known_value_3bit(self):
        # D(3, 1/128) = (1/128) * pi*sqrt(3)/2 * 4^{-3}
        v = 1.0 / 128
        d = lloyd_max_distortion(3, v)
        expected = v * math.pi * math.sqrt(3) / 2 * (4 ** -3)
        assert abs(d - expected) < 1e-15


class TestQuantizationForce:
    """Verify F_Q computation and scaling."""

    def test_grows_with_bits(self):
        v = 1.0
        fq_prev = 0
        for b in range(1, 6):
            fq = quantization_force(b, v)
            assert fq > fq_prev
            fq_prev = fq

    def test_zero_bits_returns_zero(self):
        assert quantization_force(0, 1.0) == 0.0

    def test_zero_variance_returns_zero(self):
        assert quantization_force(3, 0.0) == 0.0

    def test_exponential_growth(self):
        # F_Q ~ 4^b, so ratio F_Q(b+1)/F_Q(b) ~ 4 * (b+1)/b
        fq3 = quantization_force(3, 1.0)
        fq4 = quantization_force(4, 1.0)
        ratio = fq4 / fq3
        # 4^4/4^3 * 4/3 = 4 * 4/3 = 5.33
        assert 4.0 < ratio < 6.0


class TestLagrangeOptimalAllocation:
    """Core allocation tests."""

    def test_equal_variance_gives_uniform(self):
        """If all regions have equal variance, allocation should be uniform."""
        regions = [RegionStats(i, 10, 1.0) for i in range(5)]
        bits = lagrange_optimal_allocation(regions, total_budget=3.0)
        assert all(b == 3 for b in bits)

    def test_higher_variance_gets_more_bits(self):
        """Region with 100x variance should get more bits."""
        regions = [
            RegionStats(0, 10, 1.0),
            RegionStats(1, 10, 100.0),
            RegionStats(2, 10, 1.0),
        ]
        bits = lagrange_optimal_allocation(regions, total_budget=3.0)
        assert bits[1] >= bits[0]
        assert bits[1] >= bits[2]

    def test_respects_min_max(self):
        regions = [
            RegionStats(0, 10, 0.001),  # very low variance
            RegionStats(1, 10, 1000.0), # very high variance
        ]
        bits = lagrange_optimal_allocation(regions, total_budget=3.0, min_bits=2, max_bits=5)
        assert all(2 <= b <= 5 for b in bits)

    def test_budget_constraint(self):
        """Total allocated bits should not exceed budget."""
        regions = [RegionStats(i, 10, (i + 1) * 0.5) for i in range(6)]
        bits = lagrange_optimal_allocation(regions, total_budget=3.5, min_bits=2, max_bits=6)
        total_used = sum(r.n_elements * b for r, b in zip(regions, bits))
        total_budget = 3.5 * sum(r.n_elements for r in regions)
        # Allow small overshoot from integer rounding
        assert total_used <= total_budget + max(r.n_elements for r in regions)

    def test_empty_regions(self):
        assert lagrange_optimal_allocation([], 3.0) == []

    def test_single_region(self):
        regions = [RegionStats(0, 10, 1.0)]
        bits = lagrange_optimal_allocation(regions, total_budget=3.0)
        assert bits == [3]

    def test_structure_factor_increases_bits(self):
        """Region with structure factor > 1 should get more bits."""
        regions = [
            RegionStats(0, 10, 1.0, structure_factor=1.0),
            RegionStats(1, 10, 1.0, structure_factor=5.0),  # ZD-vulnerable
            RegionStats(2, 10, 1.0, structure_factor=1.0),
        ]
        bits = lagrange_optimal_allocation(regions, total_budget=3.0)
        assert bits[1] >= bits[0]

    def test_dominates_greedy_on_synthetic(self):
        """Lagrange should achieve <= total MSE vs uniform allocation."""
        variances = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        regions = [RegionStats(i, 10, v) for i, v in enumerate(variances)]

        # Lagrange allocation
        bits_lagrange = lagrange_optimal_allocation(regions, total_budget=3.0)

        # Uniform allocation
        bits_uniform = [3] * len(variances)

        # Total distortion
        mse_lagrange = sum(
            lloyd_max_distortion(b, v) * 10 for b, v in zip(bits_lagrange, variances)
        )
        mse_uniform = sum(
            lloyd_max_distortion(b, v) * 10 for b, v in zip(bits_uniform, variances)
        )

        assert mse_lagrange <= mse_uniform * 1.01  # allow tiny rounding tolerance


class TestComputeStructureFactor:
    """Tests for CD structure factor computation."""

    def test_quaternion_returns_one(self):
        data = torch.randn(20, 4)
        sf = compute_structure_factor(data, block_dim=4)
        assert sf == 1.0

    def test_octonion_returns_one(self):
        data = torch.randn(20, 8)
        sf = compute_structure_factor(data, block_dim=8)
        assert sf == 1.0

    def test_sedenion_exceeds_one(self):
        """Sedenions have nonzero associator, so factor should be > 1."""
        torch.manual_seed(42)
        data = torch.randn(100, 16)
        sf = compute_structure_factor(data, block_dim=16, n_triplets=200)
        assert sf > 1.0

    def test_small_data_returns_one(self):
        data = torch.randn(2, 16)  # too few vectors for triplets
        sf = compute_structure_factor(data, block_dim=16)
        assert sf == 1.0


class TestUnifiedBitAllocation:
    """Tests for the convenience wrapper."""

    def test_basic_call(self):
        bits = unified_bit_allocation([1.0, 2.0, 0.5], total_budget=3.0)
        assert len(bits) == 3
        assert all(isinstance(b, int) for b in bits)

    def test_with_structure_factors(self):
        bits = unified_bit_allocation(
            [1.0, 1.0, 1.0],
            total_budget=3.0,
            structure_factors=[1.0, 3.0, 1.0],
        )
        assert bits[1] >= bits[0]

    def test_matches_lagrange(self):
        variances = [0.5, 1.0, 2.0, 4.0]
        bits_unified = unified_bit_allocation(variances, total_budget=3.0)
        regions = [RegionStats(i, 1, v) for i, v in enumerate(variances)]
        bits_direct = lagrange_optimal_allocation(regions, total_budget=3.0)
        assert bits_unified == bits_direct

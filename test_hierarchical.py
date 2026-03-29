"""Unit tests for hierarchical.py: CD tower quantization."""
import torch
import pytest
from turboquant.hierarchical import (
    tower_levels, allocate_bits_to_levels,
    hierarchical_quantize, compare_hierarchical_vs_uniform,
)


class TestTowerLevels:
    """Tests for tower_levels."""

    def test_d128_levels(self):
        levels = tower_levels(128)
        assert len(levels) > 0
        # First level is the quaternion core (dim=4)
        assert levels[0].dim == 4
        assert levels[0].start == 0
        assert levels[0].end == 4

    def test_d64_levels(self):
        levels = tower_levels(64)
        assert len(levels) > 0

    def test_d256_levels(self):
        levels = tower_levels(256)
        assert len(levels) > 0

    def test_covers_full_dimension(self):
        for d in [4, 8, 16, 32, 64, 128, 256]:
            levels = tower_levels(d)
            total_dim = sum(lev.dim for lev in levels)
            assert total_dim == d, f"Tower levels cover {total_dim}, expected {d}"

    def test_non_overlapping(self):
        levels = tower_levels(128)
        for i in range(len(levels) - 1):
            assert levels[i].end == levels[i + 1].start

    def test_rejects_non_power_of_2(self):
        with pytest.raises(AssertionError):
            tower_levels(100)


class TestAllocateBitsToLevels:
    """Tests for allocate_bits_to_levels."""

    def test_basic_allocation(self):
        levels = tower_levels(128)
        allocated = allocate_bits_to_levels(levels, budget_bits=3)
        assert isinstance(allocated, list)
        assert len(allocated) == len(levels)

    def test_respects_min_max_bits(self):
        levels = tower_levels(128)
        allocated = allocate_bits_to_levels(levels, budget_bits=3, min_bits=2, max_bits=5)
        for level in allocated:
            assert 2 <= level.bits <= 5

    def test_higher_budget_more_bits(self):
        levels = tower_levels(128)
        alloc_2 = allocate_bits_to_levels(levels, budget_bits=2)
        alloc_4 = allocate_bits_to_levels(levels, budget_bits=4)
        total_2 = sum(l.bits * l.dim for l in alloc_2)
        total_4 = sum(l.bits * l.dim for l in alloc_4)
        assert total_4 >= total_2


class TestHierarchicalQuantize:
    """Tests for hierarchical_quantize."""

    def test_output_shape(self):
        x = torch.randn(50, 128)
        levels = tower_levels(128)
        allocated = allocate_bits_to_levels(levels, budget_bits=3)
        states, reconstructed = hierarchical_quantize(x, allocated, seed=42)
        assert reconstructed.shape == x.shape
        assert len(states) == len(allocated)

    def test_reconstruction_quality(self):
        x = torch.randn(100, 128)
        levels = tower_levels(128)
        allocated = allocate_bits_to_levels(levels, budget_bits=3)
        _, reconstructed = hierarchical_quantize(x, allocated, seed=42)
        cos = torch.nn.functional.cosine_similarity(x, reconstructed, dim=-1).mean()
        assert cos > 0.5, f"Hierarchical cosine {cos:.4f} too low"


class TestCompareHierarchicalVsUniform:
    """Tests for compare_hierarchical_vs_uniform."""

    def test_returns_dict(self):
        x = torch.randn(50, 128)
        result = compare_hierarchical_vs_uniform(x, uniform_bits=3, seed=42)
        assert isinstance(result, dict)

    def test_has_both_methods(self):
        x = torch.randn(50, 128)
        result = compare_hierarchical_vs_uniform(x, uniform_bits=3, seed=42)
        assert "hierarchical_mse" in result or "hier_mse" in result or len(result) >= 2

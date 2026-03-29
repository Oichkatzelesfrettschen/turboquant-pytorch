"""Unit tests for zd_bias.py: zero-divisor affinity scoring."""
import torch
import pytest
from turboquant.zd_bias import sedenion_zd_affinity, batch_zd_affinity, zd_quartile_analysis


class TestSedenionZDAffinity:
    """Tests for sedenion_zd_affinity."""

    def test_output_shape_single(self):
        v = torch.randn(16)
        result = sedenion_zd_affinity(v, n_samples=50)
        assert result.shape == ()

    def test_output_shape_batch(self):
        v = torch.randn(10, 16)
        result = sedenion_zd_affinity(v, n_samples=50)
        assert result.shape == (10,)

    def test_nonnegative(self):
        v = torch.randn(10, 16)
        result = sedenion_zd_affinity(v, n_samples=50)
        assert (result >= 0).all()

    def test_known_zd_pair_has_low_affinity(self):
        """e1 + e10 is a known zero-divisor partner (with e4 - e15)."""
        v = torch.zeros(16)
        v[1] = 1.0   # e1
        v[10] = 1.0  # e10
        result = sedenion_zd_affinity(v.unsqueeze(0), n_samples=500, seed=42)
        assert result.item() < 0.5

    def test_requires_dim_16(self):
        with pytest.raises(AssertionError):
            sedenion_zd_affinity(torch.randn(8))

    def test_deterministic(self):
        v = torch.randn(5, 16)
        r1 = sedenion_zd_affinity(v, n_samples=50, seed=42)
        r2 = sedenion_zd_affinity(v, n_samples=50, seed=42)
        assert torch.equal(r1, r2)


class TestBatchZDAffinity:
    """Tests for batch_zd_affinity."""

    def test_d128_projection(self):
        residuals = torch.randn(10, 128)
        result = batch_zd_affinity(residuals, n_samples=50)
        assert result.shape == (10,)

    def test_d8_padding(self):
        residuals = torch.randn(10, 8)
        result = batch_zd_affinity(residuals, n_samples=50)
        assert result.shape == (10,)

    def test_d16_exact(self):
        residuals = torch.randn(10, 16)
        result = batch_zd_affinity(residuals, n_samples=50)
        assert result.shape == (10,)


class TestZDQuartileAnalysis:
    """Tests for zd_quartile_analysis."""

    def test_returns_tensor(self):
        residuals = torch.randn(100, 128)
        errors = torch.rand(100)
        result = zd_quartile_analysis(residuals, errors, n_samples=20)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4,)

    def test_nonnegative(self):
        residuals = torch.randn(100, 128)
        errors = torch.rand(100).abs()
        result = zd_quartile_analysis(residuals, errors, n_samples=20)
        assert (result >= 0).all()

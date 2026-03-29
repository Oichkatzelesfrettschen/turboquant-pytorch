"""Unit tests for cd_fidelity.py: phase-geometry preservation metrics."""
import torch
import pytest
from turboquant.cd_fidelity import (
    cd_fidelity_ratio, sliding_cd_fidelity, fidelity_summary,
    residual_associator_per_token, distortion_decomposition,
)


class TestCDFidelityRatio:
    """Tests for cd_fidelity_ratio."""

    def test_perfect_preservation(self):
        """If quantized == original, ratio should be ~1.0."""
        a = torch.randn(8)
        b = torch.randn(8)
        c = torch.randn(8)
        ratio, a_pre, a_post = cd_fidelity_ratio(a, b, c, a, b, c)
        assert torch.allclose(ratio, torch.tensor(1.0), atol=1e-5)

    def test_returns_correct_shapes(self):
        a = torch.randn(10, 8)
        b = torch.randn(10, 8)
        c = torch.randn(10, 8)
        a_q = a + torch.randn_like(a) * 0.1
        ratio, a_pre, a_post = cd_fidelity_ratio(a, b, c, a_q, b, c)
        assert ratio.shape == a_pre.shape

    def test_quaternion_zero_associator(self):
        """Quaternions are associative, so associator should be ~0."""
        a = torch.randn(4)
        b = torch.randn(4)
        c = torch.randn(4)
        a_q = a + torch.randn_like(a) * 0.01
        ratio, a_pre, a_post = cd_fidelity_ratio(a, b, c, a_q, b, c)
        # Both should be near zero for quaternions (dim=4, associative)
        assert a_pre.item() < 0.1  # associator ~ 0 for quaternions

    def test_octonion_nonzero_associator(self):
        """Octonions are non-associative, so associator should be nonzero."""
        torch.manual_seed(42)
        a = torch.randn(8)
        b = torch.randn(8)
        c = torch.randn(8)
        a_q = a + torch.randn_like(a) * 0.1
        ratio, a_pre, a_post = cd_fidelity_ratio(a, b, c, a_q, b, c)
        # Octonion associator is typically nonzero
        assert a_pre.item() > 0.01


class TestSlidingCDFidelity:
    """Tests for sliding_cd_fidelity."""

    def test_output_shape(self):
        original = torch.randn(20, 8)
        quantized = original + torch.randn_like(original) * 0.1
        result = sliding_cd_fidelity(original, quantized)
        assert result.shape == (18,)  # n - 2

    def test_perfect_input(self):
        original = torch.randn(20, 8)
        result = sliding_cd_fidelity(original, original)
        assert torch.allclose(result, torch.ones_like(result), atol=1e-5)

    def test_short_sequence(self):
        original = torch.randn(2, 8)
        result = sliding_cd_fidelity(original, original)
        assert result.numel() == 0


class TestFidelitySummary:
    """Tests for fidelity_summary."""

    def test_returns_dataclass(self):
        from turboquant.cd_fidelity import FidelitySummary
        original = torch.randn(20, 8)
        quantized = original + torch.randn_like(original) * 0.1
        summary = fidelity_summary(original, quantized)
        assert isinstance(summary, FidelitySummary)
        assert summary.n_triplets == 18

    def test_perfect_input(self):
        original = torch.randn(20, 8)
        summary = fidelity_summary(original, original)
        assert abs(summary.mean_ratio - 1.0) < 1e-5


class TestResidualAssociator:
    """Tests for residual_associator_per_token."""

    def test_output_shape(self):
        residuals = torch.randn(20, 8)
        result = residual_associator_per_token(residuals)
        assert result.shape == (20,)

    def test_short_sequence(self):
        residuals = torch.randn(2, 8)
        result = residual_associator_per_token(residuals)
        assert result.shape == (2,)
        assert (result == 0).all()

    def test_nonnegative(self):
        residuals = torch.randn(20, 8)
        result = residual_associator_per_token(residuals)
        assert (result >= 0).all()


class TestDistortionDecomposition:
    """Tests for distortion_decomposition."""

    def test_returns_dict(self):
        original = torch.randn(20, 8)
        quantized = original + torch.randn_like(original) * 0.1
        result = distortion_decomposition(original, quantized)
        assert isinstance(result, dict)
        assert "total" in result
        assert "magnitude" in result
        assert "phase" in result

    def test_zero_distortion(self):
        original = torch.randn(20, 8)
        result = distortion_decomposition(original, original)
        assert result["total"].abs().max() < 1e-5
        assert result["magnitude"].abs().max() < 1e-5
        assert result["phase"].abs().max() < 1e-5

    def test_shapes(self):
        original = torch.randn(20, 8)
        quantized = original + torch.randn_like(original) * 0.1
        result = distortion_decomposition(original, quantized)
        assert result["total"].shape == (20,)
        assert result["magnitude"].shape == (20,)
        assert result["phase"].shape == (20,)

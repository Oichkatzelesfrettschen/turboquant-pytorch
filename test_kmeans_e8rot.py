"""Unit tests for kmeans_vq.py and e8_rotation.py."""
import math
import torch
import pytest
from turboquant.kmeans_vq import KMeans8DQuantizer, train_kmeans_codebook, get_kmeans_codebook
from turboquant.e8_rotation import E8BlockRotation, select_diverse_roots
from turboquant.e8_quantizer import generate_e8_roots


class TestKMeansCodebook:
    """Tests for K-Means codebook training."""

    def test_train_produces_correct_shape(self):
        cb = train_kmeans_codebook(dim=8, n_codewords=32, n_train=500, seed=42)
        assert cb.shape == (32, 8)

    def test_codebook_cached(self):
        cb1 = get_kmeans_codebook(8, 64)
        cb2 = get_kmeans_codebook(8, 64)
        assert torch.equal(cb1, cb2)

    def test_different_k_different_codebook(self):
        cb1 = get_kmeans_codebook(8, 32)
        cb2 = get_kmeans_codebook(8, 64)
        assert cb1.shape != cb2.shape


class TestKMeans8DQuantizer:
    """Tests for KMeans8DQuantizer."""

    def test_requires_d_divisible_by_8(self):
        with pytest.raises(ValueError):
            KMeans8DQuantizer(d=100)

    def test_quantize_dequantize_roundtrip(self):
        q = KMeans8DQuantizer(d=128, n_codewords=256)
        x = torch.randn(50, 128)
        state = q.quantize(x)
        recon = q.dequantize(state)
        assert recon.shape == x.shape

    def test_reconstruction_quality(self):
        q = KMeans8DQuantizer(d=128, n_codewords=256)
        x = torch.randn(100, 128)
        state = q.quantize(x)
        recon = q.dequantize(state)
        cos = torch.nn.functional.cosine_similarity(x, recon, dim=-1).mean()
        assert cos > 0.3, f"KMeans cosine {cos:.4f} too low"

    def test_bits_per_dimension(self):
        q256 = KMeans8DQuantizer(d=128, n_codewords=256)
        assert abs(q256.bits_per_dimension() - 1.0) < 1e-10  # log2(256)/8 = 1.0

        q1024 = KMeans8DQuantizer(d=128, n_codewords=1024)
        assert abs(q1024.bits_per_dimension() - 1.25) < 1e-10  # log2(1024)/8

    def test_indices_shape(self):
        q = KMeans8DQuantizer(d=128, n_codewords=256)
        x = torch.randn(20, 128)
        state = q.quantize(x)
        assert state["indices"].shape == (20, 16)  # 128/8 = 16 blocks

    def test_calibrate(self):
        q = KMeans8DQuantizer(d=128, n_codewords=256)
        data = torch.randn(100, 128) * 5.0
        q.calibrate(data)
        assert q._scale is not None
        assert q._scale > 1.0  # data has std ~5, so scale should be large

    def test_to_device(self):
        q = KMeans8DQuantizer(d=128, n_codewords=256)
        q2 = q.to("cpu")
        assert q2._codebook.device.type == "cpu"


class TestSelectDiverseRoots:
    """Tests for E8 root selection."""

    def test_output_shape(self):
        roots = generate_e8_roots()
        selected = select_diverse_roots(roots, n_roots=8)
        assert selected.shape == (8, 8)

    def test_deterministic(self):
        roots = generate_e8_roots()
        s1 = select_diverse_roots(roots, n_roots=4, seed=42)
        s2 = select_diverse_roots(roots, n_roots=4, seed=42)
        assert torch.equal(s1, s2)

    def test_different_seeds_different_selection(self):
        roots = generate_e8_roots()
        s1 = select_diverse_roots(roots, n_roots=4, seed=42)
        s2 = select_diverse_roots(roots, n_roots=4, seed=99)
        assert not torch.equal(s1, s2)

    def test_selected_are_valid_roots(self):
        roots = generate_e8_roots()
        selected = select_diverse_roots(roots, n_roots=8)
        # Each selected root should have norm sqrt(2) (E8 root property)
        norms = selected.norm(dim=-1)
        assert torch.allclose(norms, torch.full_like(norms, math.sqrt(2)), atol=1e-5)


class TestE8BlockRotation:
    """Tests for E8BlockRotation."""

    def test_requires_d_divisible_by_16(self):
        with pytest.raises(ValueError):
            E8BlockRotation(d=100)

    def test_rotate_shape(self):
        rot = E8BlockRotation(d=128, seed=42)
        x = torch.randn(10, 128)
        y = rot.rotate(x)
        assert y.shape == x.shape

    def test_invertibility(self):
        rot = E8BlockRotation(d=128, seed=42)
        x = torch.randn(10, 128)
        y = rot.rotate(x)
        x_rec = rot.unrotate(y)
        assert torch.allclose(x, x_rec, atol=1e-4), \
            f"E8BlockRotation not invertible: max err {(x - x_rec).abs().max():.6f}"

    def test_storage_elements(self):
        rot = E8BlockRotation(d=128, seed=42)
        # d=128, 8 blocks of 16D, 8 E8 root coords each = 64
        assert rot.storage_elements() == 64

    def test_batch_dimensions(self):
        rot = E8BlockRotation(d=128, seed=42)
        x = torch.randn(2, 5, 128)
        y = rot.rotate(x)
        assert y.shape == (2, 5, 128)
        x_rec = rot.unrotate(y)
        assert torch.allclose(x, x_rec, atol=1e-4)

    def test_to_device(self):
        rot = E8BlockRotation(d=128, seed=42)
        rot2 = rot.to("cpu")
        assert rot2.elements.device.type == "cpu"

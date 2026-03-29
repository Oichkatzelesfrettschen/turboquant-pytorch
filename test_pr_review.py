"""
Tests for PR review findings: compressors, nsn_preprocess, hybrid_pipeline.

These are the top-3 untested production files identified by the test coverage
review agent. Each test function covers a specific behavioral path.
"""

import importlib.util
import os
import sys

# Bootstrap package before any turboquant imports
if "turboquant" not in sys.modules:
    _d = os.path.dirname(os.path.abspath(__file__))
    _s = importlib.util.spec_from_file_location(
        "turboquant", os.path.join(_d, "__init__.py"),
        submodule_search_locations=[_d],
    )
    _m = importlib.util.module_from_spec(_s)
    sys.modules["turboquant"] = _m
    _s.loader.exec_module(_m)

import pytest
import torch
import math


# --- compressors.py tests ---

class TestTurboQuantCompressorV2:
    """Tests for the production key compressor."""

    def test_compress_decompress_roundtrip(self):
        from turboquant.compressors import TurboQuantCompressorV2
        comp = TurboQuantCompressorV2(128, bits=3, seed=42, device="cpu")
        states = torch.randn(1, 2, 64, 128)  # (B, H, S, D)
        compressed = comp.compress(states, use_nsn=False, use_sign_pack=False)
        assert "k_mse" in compressed
        assert compressed["k_mse"].shape == (1, 2, 64, 128)
        assert compressed["k_mse"].dtype == torch.float16

    def test_compress_with_nsn(self):
        from turboquant.compressors import TurboQuantCompressorV2
        comp = TurboQuantCompressorV2(128, bits=3, seed=42, device="cpu")
        states = torch.randn(1, 2, 64, 128)
        compressed = comp.compress(states, use_nsn=True, use_sign_pack=False)
        assert "nsn_state" in compressed
        assert compressed["nsn_state"] is not None

    def test_compress_with_sign_packing(self):
        from turboquant.compressors import TurboQuantCompressorV2
        comp = TurboQuantCompressorV2(128, bits=3, seed=42, device="cpu")
        states = torch.randn(1, 2, 64, 128)
        compressed = comp.compress(states, use_nsn=False, use_sign_pack=True)
        assert "sign_data" in compressed
        assert "packed" in compressed["sign_data"]

    def test_asymmetric_attention_scores_shape(self):
        from turboquant.compressors import TurboQuantCompressorV2
        comp = TurboQuantCompressorV2(128, bits=3, seed=42, device="cpu")
        keys = torch.randn(1, 2, 64, 128)
        queries = torch.randn(1, 2, 1, 128)
        compressed = comp.compress(keys, use_nsn=False, use_sign_pack=False)
        scores = comp.asymmetric_attention_scores(queries, compressed)
        assert scores.shape == (1, 2, 1, 64)

    def test_nsn_improves_cosine_similarity(self):
        from turboquant.compressors import TurboQuantCompressorV2
        import torch.nn.functional as F
        comp = TurboQuantCompressorV2(128, bits=3, seed=42, device="cpu")
        # Data with channel outliers
        states = torch.randn(1, 1, 100, 128)
        states[0, 0, :, 3] *= 10
        queries = states[:, :, -1:, :]

        comp_no_nsn = comp.compress(states, use_nsn=False, use_sign_pack=False)
        scores_no = comp.asymmetric_attention_scores(queries, comp_no_nsn)
        real_scores = torch.matmul(queries.float(), states.float().transpose(-2, -1))

        cos_no = F.cosine_similarity(
            real_scores.flatten().unsqueeze(0), scores_no.flatten().unsqueeze(0)
        ).item()

        comp_nsn = comp.compress(states, use_nsn=True, use_sign_pack=False)
        scores_nsn = comp.asymmetric_attention_scores(queries, comp_nsn)
        cos_nsn = F.cosine_similarity(
            real_scores.flatten().unsqueeze(0), scores_nsn.flatten().unsqueeze(0)
        ).item()

        assert cos_nsn >= cos_no - 0.01, f"NSN should help: {cos_nsn} vs {cos_no}"


class TestTurboQuantCompressorMSE:
    """Tests for the value compressor."""

    def test_compress_decompress_roundtrip(self):
        from turboquant.compressors import TurboQuantCompressorMSE
        comp = TurboQuantCompressorMSE(128, bits=3, seed=42, device="cpu")
        states = torch.randn(1, 2, 64, 128)
        compressed = comp.compress(states)
        decompressed = comp.decompress(compressed)
        assert decompressed.shape == (1, 2, 64, 128)
        mse = ((states - decompressed) ** 2).mean().item()
        assert mse < 1.0, f"MSE too high: {mse}"


# --- nsn_preprocess.py tests ---

class TestNSNPreprocess:
    """Tests for normalize-shift-normalize pre-processing."""

    def test_roundtrip_identity(self):
        from turboquant.nsn_preprocess import nsn_preprocess, nsn_restore
        x = torch.randn(50, 128)
        x_nsn, state = nsn_preprocess(x)
        x_restored = nsn_restore(x_nsn, state)
        err = (x - x_restored).norm() / x.norm()
        assert err < 1e-5, f"NSN roundtrip error: {err}"

    def test_output_unit_norm(self):
        from turboquant.nsn_preprocess import nsn_preprocess
        x = torch.randn(50, 128)
        x_nsn, _ = nsn_preprocess(x)
        norms = x_nsn.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_handles_zero_vector(self):
        from turboquant.nsn_preprocess import nsn_preprocess, nsn_restore
        x = torch.randn(10, 64)
        x[3] = 0.0  # zero vector
        x_nsn, state = nsn_preprocess(x)
        x_restored = nsn_restore(x_nsn, state)
        # Zero vector should restore to near-zero (not NaN)
        assert not torch.isnan(x_restored).any(), "NaN in restored output"

    def test_channel_centering(self):
        from turboquant.nsn_preprocess import nsn_preprocess
        x = torch.randn(100, 64) + 5.0  # large channel bias
        x_nsn, _ = nsn_preprocess(x)
        # After NSN, per-channel mean should be much smaller
        mean_before = x.mean(dim=0).abs().mean().item()
        mean_after = x_nsn.mean(dim=0).abs().mean().item()
        assert mean_after < mean_before * 0.1


class TestKIVI:
    """Tests for KIVI key/value asymmetric quantization."""

    def test_keys_roundtrip(self):
        from turboquant.nsn_preprocess import kivi_quantize_keys, kivi_dequantize_keys
        keys = torch.randn(100, 128)
        comp = kivi_quantize_keys(keys, bits=2)
        deq = kivi_dequantize_keys(comp)
        assert deq.shape == keys.shape
        mse = ((keys - deq) ** 2).mean().item()
        assert mse < 1.0

    def test_values_roundtrip(self):
        from turboquant.nsn_preprocess import kivi_quantize_values, kivi_dequantize_values
        vals = torch.randn(100, 128)
        comp = kivi_quantize_values(vals, bits=2)
        deq = kivi_dequantize_values(comp)
        assert deq.shape == vals.shape


class TestPrecisionWindows:
    """Tests for FP16 precision windows."""

    def test_sink_tokens_not_quantized(self):
        from turboquant.nsn_preprocess import PrecisionWindows
        pw = PrecisionWindows(recent_window=128, sink_window=4)
        mask = pw.quantize_mask(1024)
        assert not mask[0].item()
        assert not mask[3].item()
        assert mask[4].item()

    def test_recent_tokens_not_quantized(self):
        from turboquant.nsn_preprocess import PrecisionWindows
        pw = PrecisionWindows(recent_window=128, sink_window=4)
        mask = pw.quantize_mask(1024)
        assert not mask[-1].item()
        assert not mask[-128].item()
        assert mask[500].item()

    def test_short_sequence(self):
        from turboquant.nsn_preprocess import PrecisionWindows
        pw = PrecisionWindows(recent_window=128, sink_window=4)
        n_q, n_fp = pw.count_quantized(10)
        assert n_q + n_fp == 10


# --- hybrid_pipeline.py tests ---

class TestHybridWHTCDRotation:
    """Tests for the hybrid WHT + CD block rotation."""

    def test_invertibility(self):
        from turboquant.hybrid_pipeline import HybridWHTCDRotation
        rot = HybridWHTCDRotation(128, seed=42)
        x = torch.randn(100, 128)
        y = rot.rotate(x)
        x_rec = rot.unrotate(y)
        err = (x - x_rec).norm() / x.norm()
        assert err < 1e-4, f"Hybrid invertibility error: {err}"

    def test_norm_preservation(self):
        from turboquant.hybrid_pipeline import HybridWHTCDRotation
        rot = HybridWHTCDRotation(128, seed=42)
        x = torch.randn(50, 128)
        y = rot.rotate(x)
        iso_err = ((y.norm(dim=-1) / x.norm(dim=-1)) - 1).abs().mean()
        assert iso_err < 0.01, f"Isometry error: {iso_err}"

    def test_storage_elements(self):
        from turboquant.hybrid_pipeline import HybridWHTCDRotation
        rot = HybridWHTCDRotation(128, seed=42)
        # WHT (256) + CD (128) = 384
        assert rot.storage_elements() == 384


class TestHybridCompress:
    """Tests for the full hybrid compression pipeline."""

    def test_compress_returns_valid_structure(self):
        from turboquant.hybrid_pipeline import hybrid_compress
        keys = torch.randn(1, 2, 64, 128)
        compressed, k_mse = hybrid_compress(keys, bits=3)
        assert k_mse.shape == (1, 2, 64, 128)
        assert compressed.packed_signs is not None
        assert compressed.residual_norm.shape == (1, 2, 64)

    def test_attention_scores_shape(self):
        from turboquant.hybrid_pipeline import hybrid_compress, hybrid_attention_scores
        keys = torch.randn(1, 2, 64, 128)
        queries = torch.randn(1, 2, 1, 128)
        compressed, _ = hybrid_compress(keys, bits=3)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(10042)
        S = torch.randn(128, 128, generator=gen)
        scores = hybrid_attention_scores(queries, compressed, S)
        assert scores.shape == (1, 2, 1, 64)

    def test_compression_ratio(self):
        from turboquant.hybrid_pipeline import hybrid_compress
        keys = torch.randn(1, 2, 256, 128)
        compressed, _ = hybrid_compress(keys, bits=3)
        ratio = compressed.compression_ratio()
        assert ratio > 1.0, f"Should compress: ratio={ratio}"


# --- Error handling tests ---

class TestErrorHandling:
    """Tests for proper error messages on invalid inputs."""

    def test_invalid_rotation_name(self):
        from turboquant.turboquant import _make_rotation
        with pytest.raises(ValueError, match="Unknown rotation"):
            _make_rotation(128, "invalid_name", 42, "cpu")

    def test_invalid_cd_spec(self):
        from turboquant.turboquant import _make_rotation
        with pytest.raises(ValueError, match="Invalid CD rotation"):
            _make_rotation(128, "cdabc", 42, "cpu")

    def test_cd_empty_suffix(self):
        from turboquant.turboquant import _make_rotation
        with pytest.raises(ValueError, match="Invalid CD rotation"):
            _make_rotation(128, "cd", 42, "cpu")

    def test_invalid_quantizer_name(self):
        from turboquant.turboquant import _make_quantizer
        with pytest.raises(ValueError, match="Unknown quantizer"):
            _make_quantizer(128, 3, "invalid_name", "cpu")

    def test_e8_bits_warning(self):
        from turboquant.turboquant import _make_quantizer
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _make_quantizer(128, 3, "e8", "cpu")
            assert len(w) == 1
            assert "fixed ~1 bit/dim" in str(w[0].message)


if __name__ == "__main__":
    # Run all test classes
    import traceback
    n_pass = 0
    n_fail = 0
    for cls in [TestTurboQuantCompressorV2, TestTurboQuantCompressorMSE,
                TestNSNPreprocess, TestKIVI, TestPrecisionWindows,
                TestHybridWHTCDRotation, TestHybridCompress, TestErrorHandling]:
        for name in dir(cls):
            if name.startswith("test_"):
                inst = cls()
                try:
                    getattr(inst, name)()
                    print(f"  [PASS] {cls.__name__}.{name}")
                    n_pass += 1
                except Exception as e:
                    print(f"  [FAIL] {cls.__name__}.{name}: {e}")
                    traceback.print_exc()
                    n_fail += 1
    print(f"\n  {n_pass} passed, {n_fail} failed")

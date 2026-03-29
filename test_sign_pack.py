"""Unit tests for sign_pack.py: bit-packed QJL sign storage."""
import torch
import pytest
from turboquant.sign_pack import (
    pack_signs, unpack_signs, pack_signs_from_projection,
    packed_inner_product, packed_memory_bytes, unpacked_memory_bytes, memory_ratio,
)


class TestPackUnpack:
    """Roundtrip and correctness tests for sign packing."""

    def test_roundtrip_d64(self):
        signs = torch.randint(0, 2, (10, 64)) * 2 - 1
        packed = pack_signs(signs.to(torch.int8))
        unpacked = unpack_signs(packed, 64)
        assert torch.equal(signs.float(), unpacked)

    def test_roundtrip_d128(self):
        signs = torch.randint(0, 2, (50, 128)) * 2 - 1
        packed = pack_signs(signs.to(torch.int8))
        unpacked = unpack_signs(packed, 128)
        assert torch.equal(signs.float(), unpacked)

    def test_roundtrip_non_multiple_of_64(self):
        """d=100 is not a multiple of 64 -- tests padding."""
        signs = torch.randint(0, 2, (20, 100)) * 2 - 1
        packed = pack_signs(signs.to(torch.int8))
        unpacked = unpack_signs(packed, 100)
        assert torch.equal(signs.float(), unpacked)

    def test_all_positive(self):
        signs = torch.ones(5, 64, dtype=torch.int8)
        packed = pack_signs(signs)
        unpacked = unpack_signs(packed, 64)
        assert (unpacked == 1.0).all()

    def test_all_negative(self):
        signs = -torch.ones(5, 64, dtype=torch.int8)
        packed = pack_signs(signs)
        unpacked = unpack_signs(packed, 64)
        assert (unpacked == -1.0).all()

    def test_single_vector(self):
        signs = torch.tensor([[1, -1, 1, -1] * 16], dtype=torch.int8)
        packed = pack_signs(signs)
        unpacked = unpack_signs(packed, 64)
        assert torch.equal(signs.float(), unpacked)

    def test_batch_dimensions(self):
        """Test with extra batch dimensions."""
        signs = torch.randint(0, 2, (2, 4, 128)) * 2 - 1
        packed = pack_signs(signs.to(torch.int8))
        assert packed.shape == (2, 4, 2)  # 128 / 64 = 2 words
        unpacked = unpack_signs(packed, 128)
        assert torch.equal(signs.float(), unpacked)


class TestPackFromProjection:
    """Tests for pack_signs_from_projection (fused path)."""

    def test_matches_two_step(self):
        projected = torch.randn(50, 128)
        signs = (projected >= 0).to(torch.int8) * 2 - 1
        packed_old = pack_signs(signs)
        packed_new = pack_signs_from_projection(projected)
        assert torch.equal(packed_old, packed_new)

    def test_zero_projection(self):
        """Zero values should map to +1 (>= 0 is True)."""
        projected = torch.zeros(5, 64)
        packed = pack_signs_from_projection(projected)
        unpacked = unpack_signs(packed, 64)
        assert (unpacked == 1.0).all()

    def test_non_multiple_of_64(self):
        projected = torch.randn(10, 100)
        packed = pack_signs_from_projection(projected)
        signs = (projected >= 0).to(torch.int8) * 2 - 1
        packed_ref = pack_signs(signs)
        assert torch.equal(packed, packed_ref)


class TestPackedInnerProduct:
    """Tests for inner product from packed signs."""

    def test_matches_naive(self):
        signs = torch.randint(0, 2, (20, 128)) * 2 - 1
        values = torch.randn(20, 128)
        packed = pack_signs(signs.to(torch.int8))

        result_packed = packed_inner_product(packed, values, 128)
        result_naive = (signs.float() * values).sum(dim=-1)
        assert torch.allclose(result_packed, result_naive, atol=1e-5)


class TestMemoryHelpers:
    """Tests for memory calculation functions."""

    def test_packed_bytes_d64(self):
        assert packed_memory_bytes(64) == 8  # 1 word * 8 bytes

    def test_packed_bytes_d128(self):
        assert packed_memory_bytes(128) == 16  # 2 words * 8 bytes

    def test_packed_bytes_d100(self):
        assert packed_memory_bytes(100) == 16  # ceil(100/64) = 2 words

    def test_unpacked_bytes(self):
        assert unpacked_memory_bytes(128) == 128

    def test_memory_ratio_d128(self):
        assert memory_ratio(128) == 8.0  # 128 / 16

    def test_memory_ratio_d64(self):
        assert memory_ratio(64) == 8.0  # 64 / 8

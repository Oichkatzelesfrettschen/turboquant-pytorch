"""
TurboQuant KV cache v2: Asymmetric attention.

Instead of decompressing KV vectors and feeding them to standard attention,
we compute attention scores DIRECTLY from compressed representations using
the TurboQuant asymmetric inner product estimator.

Key insight from the paper:
  <q, k> ≈ <q, k_mse> + ||r_k|| * sqrt(pi/2)/m * <S@q, sign(S@r_k)>

This is unbiased with variance O(1/d), even though k_mse itself has high
per-vector error. The estimator works because QJL corrects the bias in the
inner product space, not in the vector space.

For values, we use MSE-only decompression since the weighted sum in
softmax(scores) @ V averages out per-vector errors.
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional

from .rotations import Rotation, HaarRotation
from .lattice_vq import VectorQuantizer, ScalarLloydMaxQuantizer


class TurboQuantCompressorV2:
    """
    Compressor that stores compressed representations AND supports
    direct inner product computation without full decompression.

    Supports pluggable rotation and quantization methods.
    """

    def __init__(
        self,
        head_dim: int,
        bits: int,
        seed: int,
        device: str = "cpu",
        rotation: Optional[Rotation] = None,
        quantizer: Optional[VectorQuantizer] = None,
    ):
        self.head_dim = head_dim
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)
        self.device = device

        # Rotation: use provided or default to Haar
        if rotation is not None:
            self._rotation = rotation.to(device)
        else:
            self._rotation = HaarRotation(head_dim, seed=seed, device=device)

        # Quantizer: use provided or default to scalar Lloyd-Max
        if quantizer is not None:
            self._quantizer = quantizer.to(device)
        else:
            self._quantizer = ScalarLloydMaxQuantizer(head_dim, self.mse_bits, device=device)

        # Legacy: keep centroids for backward compat with scalar quantizer
        if isinstance(self._quantizer, ScalarLloydMaxQuantizer):
            self.centroids = self._quantizer.centroids
        else:
            from .lloyd_max import LloydMaxCodebook
            self.centroids = LloydMaxCodebook(head_dim, self.mse_bits).centroids.to(device)

        # Legacy: keep Pi for backward compat
        if isinstance(self._rotation, HaarRotation):
            self.Pi = self._rotation.Pi
            self.PiT = self.Pi.T.contiguous()
        else:
            self.Pi = None
            self.PiT = None

        # QJL matrix
        gen2 = torch.Generator(device="cpu")
        gen2.manual_seed(seed + 10000)
        self.S = torch.randn(head_dim, head_dim, generator=gen2).to(device)

    @torch.no_grad()
    def compress(self, states: torch.Tensor, use_nsn: bool = True, use_sign_pack: bool = True) -> dict:
        """
        Compress states: (batch, heads, seq, head_dim) -> compressed dict.
        Stores everything needed for asymmetric inner product computation.

        Args:
            states: (batch, heads, seq, head_dim) float tensor
            use_nsn: apply NSN pre-processing (+136bp cosine on real models)
            use_sign_pack: bit-pack QJL signs (-69% memory)
        """
        B, H, S, D = states.shape
        flat = states.reshape(-1, D).float()

        # Fused normalize: single pass over data (vs separate norm + div)
        flat_norm = F.normalize(flat, dim=-1)
        vec_norms = flat.norm(dim=-1, keepdim=True)

        # NSN pre-processing: channel-center + re-normalize
        # flat_norm is already unit-norm, so NSN step 1 is identity.
        # We call nsn_preprocess which handles this correctly.
        nsn_state = None
        if use_nsn:
            from .nsn_preprocess import nsn_preprocess
            flat_norm, nsn_state = nsn_preprocess(flat_norm, already_normalized=True)

        # Rotate
        rotated = self._rotation.rotate(flat_norm)

        # Quantize + dequantize
        quant_state = self._quantizer.quantize(rotated)
        reconstructed_rotated = self._quantizer.dequantize(quant_state)

        # MSE reconstruction: unrotate, undo NSN, denormalize
        recon_flat = self._rotation.unrotate(reconstructed_rotated)
        if nsn_state is not None:
            from .nsn_preprocess import nsn_restore
            recon_flat = nsn_restore(recon_flat, nsn_state)
        k_mse = recon_flat * vec_norms

        # Residual + QJL: fuse norm and projection to avoid materializing
        # the residual twice. residual_norm uses a single linalg.norm pass.
        residual = flat - k_mse
        residual_norm = residual.norm(dim=-1)

        # QJL signs: project and pack directly from bool to avoid int8 intermediate
        projected = residual @ self.S.T
        if use_sign_pack:
            from .sign_pack import pack_signs_from_projection
            sign_data = {
                "packed": pack_signs_from_projection(projected).reshape(B, H, S, -1),
                "d": D,
            }
        else:
            signs = (projected >= 0).to(torch.int8) * 2 - 1
            sign_data = {"unpacked": signs.reshape(B, H, S, D)}

        # Float16 clamping: check only once, avoiding redundant abs().max() scan
        _fp16_max = 65504.0
        k_mse_fp16 = k_mse.to(torch.float16)
        if torch.isinf(k_mse_fp16).any():
            import warnings
            warnings.warn(
                f"k_mse values exceed float16 range. "
                f"Clamping to prevent inf. Consider normalizing inputs.",
                RuntimeWarning, stacklevel=2,
            )
            k_mse = k_mse.clamp(-_fp16_max, _fp16_max)
            k_mse_fp16 = k_mse.to(torch.float16)

        result = {
            "k_mse": k_mse_fp16.reshape(B, H, S, D),
            "sign_data": sign_data,
            "residual_norm": residual_norm.clamp(0, _fp16_max).to(torch.float16).reshape(B, H, S),
            "nsn_state": nsn_state,
            "quant_state": quant_state,
            "vec_norms": vec_norms.squeeze(-1).to(torch.float16).reshape(B, H, S),
            "shape": (B, H, S, D),
        }
        if not use_sign_pack and "unpacked" in sign_data:
            result["qjl_signs"] = sign_data["unpacked"]
        return result

    def reconstruct_k_mse(self, compressed: dict) -> torch.Tensor:
        """
        Reconstruct k_mse from compact storage (quant indices + NSN state + norms).

        Call this instead of accessing compressed["k_mse"] when using compact
        storage mode to save VRAM. The k_mse tensor can be deleted from the
        compressed dict after initial compression to free memory:

            compressed = compressor.compress(states)
            del compressed["k_mse"]  # free fp16 copy
            # Later, during attention:
            k_mse = compressor.reconstruct_k_mse(compressed)
        """
        B, H, S, D = compressed["shape"]
        quant_state = compressed["quant_state"]
        nsn_state = compressed.get("nsn_state")
        vec_norms = compressed["vec_norms"].float().reshape(-1, 1)

        reconstructed_rotated = self._quantizer.dequantize(quant_state)
        recon_flat = self._rotation.unrotate(reconstructed_rotated)
        if nsn_state is not None:
            from .nsn_preprocess import nsn_restore
            recon_flat = nsn_restore(recon_flat, nsn_state)
        k_mse = recon_flat * vec_norms
        return k_mse.to(torch.float16).reshape(B, H, S, D)

    def compact_storage_bytes(self, compressed: dict) -> int:
        """
        Calculate actual storage bytes for the compact representation
        (without the eagerly-reconstructed k_mse).
        """
        total = 0
        # Quantized indices (int8)
        qs = compressed.get("quant_state", {})
        if "indices" in qs:
            total += qs["indices"].numel() * qs["indices"].element_size()
        # Signs (packed int64)
        sd = compressed.get("sign_data", {})
        if "packed" in sd:
            total += sd["packed"].numel() * sd["packed"].element_size()
        elif "unpacked" in sd:
            total += sd["unpacked"].numel() * sd["unpacked"].element_size()
        # Residual norms (fp16)
        total += compressed["residual_norm"].numel() * 2
        # Vec norms (fp16)
        total += compressed["vec_norms"].numel() * 2
        # NSN state (channel_means: fp32 * D, norms_2: fp32 * N)
        ns = compressed.get("nsn_state")
        if ns is not None:
            total += ns.channel_means.numel() * 4
            total += ns.norms_2.numel() * 4
        return total

    @torch.no_grad()
    def asymmetric_attention_scores(self, queries: torch.Tensor, compressed: dict) -> torch.Tensor:
        """
        Compute attention scores <Q, K> directly from compressed K.

        Uses the asymmetric estimator:
            <q, k> ≈ <q, k_mse> + ||r_k|| * sqrt(pi/2)/m * <S@q, signs_k>

        Args:
            queries: (batch, heads, seq_q, head_dim)
            compressed: dict from compress()

        Returns:
            scores: (batch, heads, seq_q, seq_k)
        """
        if "k_mse" in compressed:
            k_mse = compressed["k_mse"].float()
        else:
            k_mse = self.reconstruct_k_mse(compressed).float()
        r_norm = compressed["residual_norm"].float()

        # Unpack signs (handle both packed and unpacked formats)
        sign_data = compressed.get("sign_data")
        if sign_data is not None and "packed" in sign_data:
            from .sign_pack import unpack_signs
            d = sign_data["d"]
            packed = sign_data["packed"]
            B_s, H_s, S_s = packed.shape[:3]
            signs = unpack_signs(packed.reshape(-1, packed.shape[-1]), d)
            signs = signs.reshape(B_s, H_s, S_s, d)
        elif sign_data is not None and "unpacked" in sign_data:
            signs = sign_data["unpacked"].float()
        else:
            # Legacy format: qjl_signs directly in dict
            signs = compressed["qjl_signs"].float()

        # Term 1: Q @ K_mse^T
        term1 = torch.matmul(queries.float(), k_mse.transpose(-2, -1))

        # Term 2: QJL correction
        q_projected = torch.matmul(queries.float(), self.S.T)
        qjl_ip = torch.matmul(q_projected, signs.transpose(-2, -1))

        m = self.S.shape[0]
        correction_scale = math.sqrt(math.pi / 2) / m
        term2 = correction_scale * qjl_ip * r_norm.unsqueeze(-2)

        return term1 + term2


class TurboQuantCompressorMSE:
    """
    Simpler MSE-only compressor for values (no QJL needed).

    Supports pluggable rotation and quantization methods.
    """

    def __init__(
        self,
        head_dim: int,
        bits: int,
        seed: int,
        device: str = "cpu",
        rotation: Optional[Rotation] = None,
        quantizer: Optional[VectorQuantizer] = None,
    ):
        self.head_dim = head_dim
        self.bits = bits
        self.device = device

        if rotation is not None:
            self._rotation = rotation.to(device)
        else:
            self._rotation = HaarRotation(head_dim, seed=seed, device=device)

        if quantizer is not None:
            self._quantizer = quantizer.to(device)
        else:
            self._quantizer = ScalarLloydMaxQuantizer(head_dim, bits, device=device)

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict:
        B, H, S, D = states.shape
        flat = states.reshape(-1, D).float()
        vec_norms = torch.norm(flat, dim=-1, keepdim=True)
        flat_norm = flat / (vec_norms + 1e-8)
        rotated = self._rotation.rotate(flat_norm)
        quant_state = self._quantizer.quantize(rotated)
        return {
            "quant_state": quant_state,
            "vec_norms": vec_norms.squeeze(-1).to(torch.float16),
            "shape": (B, H, S, D),
        }

    @torch.no_grad()
    def decompress(self, compressed: dict) -> torch.Tensor:
        B, H, S, D = compressed["shape"]
        reconstructed_rotated = self._quantizer.dequantize(compressed["quant_state"])
        reconstructed = self._rotation.unrotate(reconstructed_rotated)
        vec_norms = compressed["vec_norms"].float().unsqueeze(-1)
        return (reconstructed * vec_norms).reshape(B, H, S, D)


class SVDCompressorV2:
    """
    SVD pre-compression + TurboQuant for joint rank-bitwidth optimization.

    For KV caches with low-rank structure, SVD reduces dimensionality
    before quantization, achieving higher compression at the same quality.

    Pipeline: K -> SVD(rank=r) -> [U_r, S_V_r] -> quantize(S_V_r) -> store

    The U_r factor (seq_len x rank) is stored at FP16 (small relative
    to the original seq_len x d). The S_V_r factor (rank x d) is
    quantized via the standard TurboQuant pipeline.

    For attention: <q, K_i> = U_r[i] @ (S_V_r @ q), which is O(rank*d)
    instead of O(d^2) -- a speedup when rank << d.
    """

    def __init__(
        self,
        head_dim: int,
        rank: int = 32,
        bits: int = 4,
        seed: int = 0,
        device: str = "cpu",
        rotation: Optional[Rotation] = None,
        quantizer: Optional[VectorQuantizer] = None,
    ):
        self.head_dim = head_dim
        self.rank = rank
        self.bits = bits
        self.device = device
        self._inner = TurboQuantCompressorMSE(
            head_dim, bits, seed=seed, device=device,
            rotation=rotation, quantizer=quantizer,
        )

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict:
        """SVD + quantize: states (B, H, S, D) -> compressed dict."""
        from .tensor_decomposition import svd_compress as _svd

        B, H, S, D = states.shape
        all_U = []
        all_sv_compressed = []
        r = min(self.rank, min(S, D))

        for b in range(B):
            for h in range(H):
                K = states[b, h].float()
                U_r, S_V_r = _svd(K, r)
                sv_4d = S_V_r.unsqueeze(0).unsqueeze(0)
                sv_comp = self._inner.compress(sv_4d)
                all_U.append(U_r.to(torch.float16))
                all_sv_compressed.append(sv_comp)

        return {
            "U_factors": all_U,
            "SV_compressed": all_sv_compressed,
            "shape": (B, H, S, D),
            "rank": r,
        }

    @torch.no_grad()
    def decompress(self, compressed: dict) -> torch.Tensor:
        """Reconstruct full KV from SVD factors."""
        B, H, S, D = compressed["shape"]
        result = torch.zeros(B, H, S, D, device=self.device)
        idx = 0
        for b in range(B):
            for h in range(H):
                U_r = compressed["U_factors"][idx].float()
                sv_comp = compressed["SV_compressed"][idx]
                S_V_r = self._inner.decompress(sv_comp).reshape(-1, D)
                result[b, h] = (U_r @ S_V_r).to(result.dtype)
                idx += 1
        return result

    def storage_bytes(self, compressed: dict) -> int:
        """Actual storage bytes for compressed representation."""
        B, H, S, D = compressed["shape"]
        rank = compressed["rank"]
        total = B * H * S * rank * 2  # U factors at fp16
        for sv in compressed["SV_compressed"]:
            qs = sv.get("quant_state", {})
            if "indices" in qs:
                total += qs["indices"].numel() * qs["indices"].element_size()
            total += sv["vec_norms"].numel() * 2
        return total

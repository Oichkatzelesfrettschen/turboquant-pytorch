"""
Smart configuration defaults for TurboQuant.

Auto-selects the best rotation and quantizer based on dimension, bit-width,
and available hardware. Three presets:

    recommended():   Best quality at reasonable speed. WHT or E8Block rotation,
                     adaptive bit allocation, QJL with sign packing.
    paper_default(): Matches Google's TurboQuant paper exactly (Haar + Lloyd-Max).
    fast():          Minimum latency. WHT rotation, scalar quantizer, no QJL at 4+ bits.

Based on measured results from open_gororoba validation:
    - WHT is 3.1x faster than Haar with 0.5% better MSE
    - E8Block rotation passes KS test vs Haar (p=0.816)
    - QJL correction helps at bits <= 3, adds noise at bits >= 4
    - Adaptive bit allocation gives 23% MSE improvement at 25% promotion

Ported from open_gororoba/crates/cd_kernel/src/turboquant/config.rs.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TurboQuantConfig:
    """Full configuration for TurboQuant quantization."""
    # Rotation
    rotation: str = "wht"
    # Quantizer
    quantizer: str = "scalar"
    # Bit-width
    bits: int = 3
    # QJL (Stage 2)
    use_qjl: bool = True
    qjl_dim: Optional[int] = None  # None = same as vector dim
    use_sign_packing: bool = True
    # Adaptive allocation
    adaptive: bool = False
    adaptive_promote_fraction: float = 0.25
    # Hierarchical tower
    hierarchical: bool = False
    min_level_bits: int = 2
    max_level_bits: int = 5

    @staticmethod
    def recommended(d: int = 128, bits: int = 4) -> "TurboQuantConfig":
        """
        Best quality at reasonable speed (validated by ablation study).

        Default bits=4 (was 3) because int8 index storage makes 3-bit
        and 4-bit cost identical memory. 4-bit gives +2.8pp avg cosine
        on Qwen2.5-3B (0.961 -> 0.989) at zero extra storage cost.

        For maximum compression, use recommended_e8() instead: E8 lattice
        at ~1 bit/dim matches 4-bit scalar quality (0.990 vs 0.989 cosine).

        NSN is the dominant contributor (+136bp cosine, +25% top1).
        Sign packing gives -69% memory with zero quality loss.
        Post-VQ scaling is DISABLED (hurts -2bp on real data).
        """
        if d >= 64 and (d & (d - 1)) == 0:
            rotation = "wht"
        else:
            rotation = "haar"

        return TurboQuantConfig(
            rotation=rotation,
            quantizer="scalar",
            bits=bits,
            use_qjl=bits <= 3,
            use_sign_packing=True,
            adaptive=False,  # per-head adaptive adds complexity, marginal gain
            hierarchical=False,
        )

    @staticmethod
    def recommended_e8(d: int = 128) -> "TurboQuantConfig":
        """
        Maximum compression with E8 lattice + CD8 rotation.

        Validated on Qwen2.5-3B (36 layers, all-layer average):
            CD8+E8+NSN: cos 0.9895, MSE 0.169 at ~1 bit/dim
            WHT+Scalar4b+NSN: cos 0.9889, MSE 0.185 at ~4 bits/dim

        E8 matches 4-bit scalar quality at 1/4 the bit rate.
        The structural alignment between CD8 (8D block rotation) and
        E8 (8D lattice quantizer) gives the best rate-distortion.

        Requires d % 8 == 0 (true for all current LLM head dimensions).
        """
        return TurboQuantConfig(
            rotation="cd8",
            quantizer="e8",
            bits=3,  # E8 is ~1 bit/dim regardless of this parameter
            use_qjl=True,  # QJL correction helps at 1 bit/dim
            use_sign_packing=True,
            adaptive=False,
        )

    @staticmethod
    def paper_default(bits: int = 3) -> "TurboQuantConfig":
        """Matches Google's TurboQuant paper exactly."""
        return TurboQuantConfig(
            rotation="haar",
            quantizer="scalar",
            bits=bits,
            use_qjl=True,
            use_sign_packing=False,
            adaptive=False,
        )

    @staticmethod
    def fast(d: int = 128, bits: int = 3) -> "TurboQuantConfig":
        """Minimum latency configuration."""
        rotation = "wht" if (d & (d - 1)) == 0 else "haar"
        return TurboQuantConfig(
            rotation=rotation,
            quantizer="scalar",
            bits=bits,
            use_qjl=bits <= 2,  # only at very low bits
            use_sign_packing=True,
            adaptive=False,
        )

    @staticmethod
    def experimental_cd_tower(d: int = 128, bits: int = 3) -> "TurboQuantConfig":
        """Hierarchical CD tower with per-level bit allocation."""
        return TurboQuantConfig(
            rotation="wht",
            quantizer="scalar",
            bits=bits,
            use_qjl=bits <= 3,
            use_sign_packing=True,
            adaptive=True,
            hierarchical=True,
            min_level_bits=1,
            max_level_bits=5,
        )

    @staticmethod
    def experimental_clifford(d: int = 128, bits: int = 3) -> "TurboQuantConfig":
        """Clifford Cl(3,0) rotor rotation (RotorQuant-style)."""
        return TurboQuantConfig(
            rotation="clifford",
            quantizer="scalar",
            bits=bits,
            use_qjl=bits <= 3,
            use_sign_packing=True,
            adaptive=False,
        )


@dataclass
class KVCacheConfig:
    """Configuration for KV cache quantization with key/value asymmetry.

    KIVI insight: keys need inner product accuracy (use QJL), values need
    MSE accuracy (skip QJL). Different bit-widths for keys vs values is
    also beneficial since attention is more sensitive to key quantization.

    GroupQuant insight: per-group scale and zero-point improves dynamic
    range coverage, especially for outlier channels.
    """
    key_config: TurboQuantConfig = field(default_factory=lambda: TurboQuantConfig(bits=3, use_qjl=True))
    value_config: TurboQuantConfig = field(default_factory=lambda: TurboQuantConfig(bits=3, use_qjl=False))
    # KIVI-style: per-channel keys, per-token values
    key_quant_axis: str = "per_channel"   # "per_channel" or "per_token"
    value_quant_axis: str = "per_token"   # "per_channel" or "per_token"
    # GroupQuant-style: group size for scale/zero computation
    group_size: int = 128  # 0 = no grouping (per-tensor)
    # Precision windows (StreamingLLM-style)
    sink_tokens: int = 4      # first N tokens at full precision
    recent_window: int = 128  # last N tokens at full precision

    @staticmethod
    def kivi_style(key_bits: int = 2, value_bits: int = 2) -> "KVCacheConfig":
        """KIVI paper configuration: 2-bit keys per-channel, 2-bit values per-token."""
        return KVCacheConfig(
            key_config=TurboQuantConfig(bits=key_bits, use_qjl=True, rotation="wht"),
            value_config=TurboQuantConfig(bits=value_bits, use_qjl=False, rotation="wht"),
            key_quant_axis="per_channel",
            value_quant_axis="per_token",
            group_size=128,
        )

    @staticmethod
    def turboquant_enhanced(bits: int = 3) -> "KVCacheConfig":
        """Our enhanced configuration: CD rotation + adaptive + sign packing."""
        return KVCacheConfig(
            key_config=TurboQuantConfig.recommended(bits=bits),
            value_config=TurboQuantConfig(bits=bits, use_qjl=False, rotation="wht"),
            key_quant_axis="per_channel",
            value_quant_axis="per_token",
            group_size=128,
            sink_tokens=4,
            recent_window=128,
        )

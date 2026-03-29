# NSN-TurboQuant

**Distribution-Aware KV Cache Compression with 7.5x Bandwidth Reduction**

Extends Google's TurboQuant (ICLR 2026) with NSN pre-processing, structured Walsh-Hadamard rotation, and bit-packed QJL residual correction. Grounded in Cayley-Dickson algebraic framework with 30/36 properties formally verified via Z3.

## Key Results

| Metric | Vanilla TurboQuant | NSN-TurboQuant | Gain |
|--------|-------------------|----------------|------|
| Cosine similarity | 0.9849 | **0.9985** | +136bp |
| Top-1 attention match | 55.6% | **80.6%** | +25pp |
| KV cache memory | 9.0 MB | **2.8 MB** | -69% |
| Bandwidth per score | 256 B | **34 B** | 7.5x |

Validated on Qwen2.5-3B-Instruct and Mistral-7B-Instruct-v0.3.

## The Surprising Finding

**Distribution normalization (NSN) contributes +136bp cosine similarity. Rotation method choice contributes <1bp.** The per-channel outlier distribution of real KV caches (423x variance ratio) is the dominant quality bottleneck, not the decorrelation method.

## Installation

```bash
pip install -e "."          # core only
pip install -e ".[models]"  # + transformers, accelerate, bitsandbytes
pip install -e ".[all]"     # + pytest, z3-solver
```

## Quick Start

```python
from turboquant import TurboQuantMSE
from turboquant.config import TurboQuantConfig

# Recommended config (ablation-validated)
cfg = TurboQuantConfig.recommended(d=128, bits=3)

# Or manually
tq = TurboQuantMSE(d=128, bits=3, rotation="wht")
x_hat, state = tq(x)
```

### Full pipeline with NSN + sign packing

```python
from turboquant.compressors import TurboQuantCompressorV2

comp = TurboQuantCompressorV2(head_dim=128, bits=3, seed=42, device="cuda")
compressed = comp.compress(keys, use_nsn=True, use_sign_pack=True)
scores = comp.asymmetric_attention_scores(queries, compressed)
```

## Architecture

| Category | Modules |
|----------|---------|
| Core pipeline | turboquant.py, compressors.py, lloyd_max.py |
| CD algebra | cd_algebra.py, cd_rotation.py, cd_fidelity.py, zd_bias.py |
| Rotations | rotations.py, e8_rotation.py, clifford_rotor.py, hybrid_pipeline.py |
| Quantizers | lattice_vq.py, e8_quantizer.py, lattice_codebook.py, kmeans_vq.py |
| Pre/post-processing | nsn_preprocess.py, sign_pack.py, adaptive.py, hierarchical.py |
| GPU/CPU dispatch | gpu_dispatch.py, cpu_dispatch.py, triton_kernels.py, cuda_ops.py |
| Verification | formal_verify.py (30 Z3 proof groups, 55 statements) |

## GPU Architecture Dispatch

Auto-detects and optimizes for your GPU:

| Architecture | SM | Optimization |
|---|---|---|
| Hopper (H100) | 9.0+ | TF32/FP8, TMA |
| Ada (RTX 40xx) | 8.9 | TF32 (3.02x), 128KB L1 |
| Ampere (RTX 30xx) | 8.0+ | TF32, BF16 |
| Turing (RTX 20xx) | 7.5 | FP16 tensor cores |
| Generic | <7.5 | FP32 cuBLAS |

## Verification

```bash
python test_turboquant.py     # 7 original tests
python test_synthesis.py      # 9 synthesis tests
python test_pr_review.py      # 26 PR review tests
python formal_verify.py       # 55 Z3 SMT proofs (<2 seconds)
```

97 total verification points. 30/36 algebraic properties Z3-proven.

## Paper

MICRO 2026 submission: `paper/micro2026.pdf` (7 pages, ACM sigconf, double-blind)

## License

MIT

# Changelog

All notable changes to this project follow [Keep a Changelog](https://keepachangelog.com/).

## [0.2.0] - 2026-03-29

### Added
- NSN pre-processing (+136bp cosine similarity on Qwen2.5-3B)
- Walsh-Hadamard rotation via cuBLAS with TF32 dispatch (3.02x on Ampere+)
- Bit-packed QJL signs (8x memory reduction, zero quality loss)
- Cayley-Dickson algebra engine (cd_algebra.py) with recursive multiply/conjugate/norm
- CD block rotations (quaternion, octonion, sedenion tower)
- E8 lattice quantizer (Conway-Sloane closest-point decoder)
- Z8 prefix-cut codebooks (Lambda_2048 through Lambda_32)
- Adaptive per-layer bit allocation (variance-based, calibration-free)
- Hierarchical CD tower quantization
- Clifford Cl(3,0) rotor rotation
- Spectral analysis and distribution diagnostics
- Tensor decomposition (SVD pre-compression)
- 5-tier GPU architecture dispatch (Hopper through generic)
- CPU SIMD-aware dispatch with searchsorted optimization
- 55 Z3 SMT proofs across 30 proof groups (formal_verify.py)
- MICRO 2026 paper (7-page ACM sigconf, double-blind)
- Full 6-phase profiling pipeline (15 non-overlapping tools)
- Real compressed inference benchmark on SmolLM2-135M and Qwen2.5-3B
- Monkey-patched attention from compressed KV cache
- 128 pytest tests (unit + integration + Z3 proofs)
- ruff linter configuration (0 warnings)
- CLAUDE.md project instructions
- Compact storage format with reconstruct_k_mse()
- "Free bits" discovery: 4-bit MSE is free vs 3-bit with int8 storage

### Changed
- Default quantizer indices now int8 (was int64, 8x reduction)
- NSN already_normalized fast path (skip redundant normalization)
- pack_signs_from_projection (skip int8 intermediate allocation)
- Float16 overflow check uses torch.isinf (faster than abs().max())
- F.normalize fusion in compressor (single pass)

### Fixed
- pytest collection crash from __init__.py relative imports
- WHT butterfly aliasing (clone before write)
- Clifford Cl(3,0) geometric product (46% error -> 1.1e-7)
- E8 quantizer constant cosine (grid_search default)
- ScalarLloydMaxQuantizer.to() missing _boundaries
- Codebook cache TOCTOU race (double-checked locking)
- fp16 equivalent calculation for values
- Materialized WHT cache dangling references (data_ptr keys)
- cuda_ops bare except (specific ImportError/RuntimeError)
- NSN epsilon 1e-15 too small for float32 (now 1e-8)
- Compressor fp16 overflow (clamp + warning)

## [0.1.0] - 2026-03-28

### Added
- Initial TurboQuant implementation (Google ICLR 2026 paper)
- Haar rotation + Lloyd-Max scalar quantization + QJL residual correction
- TurboQuantMSE, TurboQuantProd, TurboQuantKVCache core classes
- 7 integration tests (test_turboquant.py)
- pyproject.toml with optional dependencies

# TurboQuant PyTorch -- Project Instructions

## Overview

KV cache compression pipeline extending Google's TurboQuant (ICLR 2026) with:
- NSN pre-processing (+136bp cosine -- the dominant contribution)
- Structured Walsh-Hadamard rotation via cuBLAS (64x fewer params than Haar)
- Bit-packed QJL signs (8x sign storage reduction)
- Cayley-Dickson algebraic framework (55 Z3-proven statements)
- Adaptive per-layer bit allocation
- 5-tier GPU architecture dispatch (Hopper through generic)

Research repo targeting MICRO 2026 (deadline: March 31, 2026).

## Build

```bash
pip install -e "."            # core: torch + scipy
pip install -e ".[models]"    # + transformers, accelerate, bitsandbytes
pip install -e ".[test]"      # + pytest, hypothesis
pip install -e ".[verify]"    # + z3-solver
pip install -e ".[all]"       # everything
```

CUDA extensions (optional, auto-fallback to PyTorch):
```bash
cd csrc && python setup.py install
```

## Test

```bash
pytest -x -q                        # all pytest-compatible tests
python formal_verify.py             # 55 Z3 SMT proofs (<2s)
python test_turboquant.py           # 7 original integration tests
python bench_compressed_inference.py # real model benchmark
```

Treat warnings as errors. All tests must pass before commit.

## Lint

```bash
ruff check .                        # lint (when configured)
mypy compressors.py rotations.py    # type check core modules
```

## Standards

- **Warnings as errors**: no bare excepts, no unguarded divisions, no silent fallbacks
- **Z3 verification**: algebraic properties must be SMT-proven, not just tested
- **Epsilon guards**: all divisions use + 1e-8 (float32) or + 1e-15 (float64)
- **Device agnostic**: every GPU path must have a CPU fallback
- **Reproducibility**: seed parameters everywhere, deterministic codebooks
- **No TODO/FIXME**: codebase is complete -- new features get full implementation

## Architecture (bottom-up)

1. **Algebra**: cd_algebra, algebra_foundations, zd_bias, clifford_rotor, cariow
2. **Quantizers**: lloyd_max, lattice_vq, e8_quantizer, lattice_codebook, kmeans_vq
3. **Rotations**: rotations (ABC), cd_rotation, e8_rotation, hybrid_pipeline, triton_kernels
4. **Core**: turboquant, compressors (pluggable rotation + quantizer)
5. **Enhancements**: nsn_preprocess, sign_pack, adaptive, hierarchical
6. **Dispatch**: gpu_dispatch, cpu_dispatch, cuda_ops, config
7. **Verification**: formal_verify (55 Z3 proofs across 30 groups)

## Key Conventions

- `compressors.py` is the production API -- benchmarks use this
- `turboquant.py` is the lower-level pipeline with rotation/quantizer dispatch
- All rotations implement the `Rotation` ABC from `rotations.py`
- All quantizers implement `VectorQuantizer` ABC from `lattice_vq.py`
- NSN is always-on in production (use_nsn=True) -- ablation validated
- Sign packing is always-on (use_sign_pack=True) -- zero quality loss
- int8 indices mean 3-bit and 4-bit have SAME storage cost ("free bits")

## Common Issues

1. **pytest import error on test_turboquant.py**: conftest.py handles __init__.py exclusion but test_turboquant.py uses a legacy import pattern. Run it directly with `python test_turboquant.py`.
2. **py-spy fails on Python 3.14**: known incompatibility. Use memray instead.
3. **BitsAndBytes missing CUDA .so**: symlink cuda130.so -> cuda131.so if needed.
4. **DynamicCache API (transformers 5.0+)**: use `cache.layers[i].keys`, not `cache[i][0]`.

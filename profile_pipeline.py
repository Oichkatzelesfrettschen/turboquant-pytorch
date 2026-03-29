#!/usr/bin/env python3
"""
Profile the full TurboQuant pipeline to identify bottlenecks.

Measures wall-clock time for each stage:
    1. NSN pre-processing (normalize-shift-normalize)
    2. Rotation (Haar / WHT / CD / E8Block)
    3. Quantization (Lloyd-Max / E8 / KMeans)
    4. Dequantization
    5. Unrotation
    6. NSN restore
    7. QJL sign projection + inner product

Reports: per-stage time, percentage of total, throughput (vectors/sec).
Also profiles memory allocation patterns.
"""

import os
import sys
import importlib.util
import time
from collections import defaultdict

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "turboquant", os.path.join(_pkg_dir, "__init__.py"),
    submodule_search_locations=[_pkg_dir],
)
_tq = importlib.util.module_from_spec(_spec)
sys.modules["turboquant"] = _tq
_spec.loader.exec_module(_tq)

import torch
import torch.cuda

from turboquant.rotations import HaarRotation, WHTRotation, CDRotation
from turboquant.e8_rotation import E8BlockRotation
from turboquant.lloyd_max import LloydMaxCodebook
from turboquant.nsn_preprocess import nsn_preprocess, nsn_restore
from turboquant.lattice_vq import ScalarLloydMaxQuantizer, E8LatticeQuantizer
from turboquant.kmeans_vq import KMeans8DQuantizer
from turboquant.sign_pack import pack_signs, unpack_signs, packed_inner_product


def _sync():
    """Synchronize CUDA if available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def profile_stage(name, fn, n_warmup=3, n_iter=20):
    """Profile a single stage, return mean time in ms."""
    for _ in range(n_warmup):
        fn()
    _sync()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    _sync()
    elapsed = (time.perf_counter() - t0) / n_iter * 1000
    return elapsed


def profile_full_pipeline(
    d=128, n=2000, bits=3, device="cpu",
    rotation_name="wht", quantizer_name="scalar",
    use_nsn=True, use_qjl=True,
):
    """Profile the complete quantization pipeline."""
    torch.manual_seed(42)

    # Generate test data with realistic channel outliers
    x = torch.randn(n, d, device=device)
    x[:, 3] *= 8
    x[:, 47] *= 6
    x[:, 99] *= 10

    stages = {}

    # --- Stage 1: NSN Pre-processing ---
    if use_nsn:
        nsn_state = [None]
        x_nsn = [None]
        def _nsn():
            x_nsn[0], nsn_state[0] = nsn_preprocess(x)
        stages["1_nsn_preprocess"] = profile_stage("nsn", _nsn)
        _nsn()  # execute once for subsequent stages
        x_input = x_nsn[0]
    else:
        x_input = x
        nsn_state = [None]

    # --- Stage 2: Rotation ---
    rot_map = {
        "haar": lambda: HaarRotation(d, seed=42, device=device),
        "wht": lambda: WHTRotation(d, seed=42, device=device),
        "cd8": lambda: CDRotation(d, block_dim=8, seed=42, device=device),
        "e8block": lambda: E8BlockRotation(d, seed=42, device=device),
    }
    rot = rot_map[rotation_name]()
    x_rotated = [None]
    def _rotate():
        x_rotated[0] = rot.rotate(x_input)
    stages["2_rotation"] = profile_stage("rot", _rotate)
    _rotate()

    # --- Stage 3: Quantize ---
    quant_map = {
        "scalar": lambda: ScalarLloydMaxQuantizer(d, bits, device=device),
        "e8": lambda: E8LatticeQuantizer(d, device=device),
        "kmeans": lambda: KMeans8DQuantizer(d, 256, device=device),
    }
    quant = quant_map[quantizer_name]()
    if hasattr(quant, "calibrate"):
        quant.calibrate(x_rotated[0])
    q_state = [None]
    def _quantize():
        q_state[0] = quant.quantize(x_rotated[0])
    stages["3_quantize"] = profile_stage("quant", _quantize)
    _quantize()

    # --- Stage 4: Dequantize ---
    x_dequant = [None]
    def _dequantize():
        x_dequant[0] = quant.dequantize(q_state[0])
    stages["4_dequantize"] = profile_stage("dequant", _dequantize)
    _dequantize()

    # --- Stage 5: Unrotation ---
    x_unrot = [None]
    def _unrotate():
        x_unrot[0] = rot.unrotate(x_dequant[0])
    stages["5_unrotation"] = profile_stage("unrot", _unrotate)
    _unrotate()

    # --- Stage 6: NSN Restore ---
    if use_nsn:
        def _nsn_restore():
            nsn_restore(x_unrot[0], nsn_state[0])
        stages["6_nsn_restore"] = profile_stage("nsn_r", _nsn_restore)

    # --- Stage 7: QJL Sign Projection ---
    if use_qjl:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(43)
        S = torch.randn(d, d, generator=gen).to(device)
        residual = x - (x_unrot[0] if not use_nsn else nsn_restore(x_unrot[0], nsn_state[0]))
        residual_norm = residual.norm(dim=-1)
        signs = torch.sign(residual @ S.T)
        signs[signs == 0] = 1.0

        def _qjl_project():
            projected = x[:10] @ S.T
            return (projected * signs[:10]).sum(dim=-1)
        stages["7_qjl_project"] = profile_stage("qjl", _qjl_project)

        # Sign packing
        signs_i8 = signs.to(torch.int8)
        def _sign_pack():
            return pack_signs(signs_i8)
        stages["7b_sign_pack"] = profile_stage("pack", _sign_pack)
        packed = pack_signs(signs_i8)

        query_batch = x[:10]
        def _packed_ip():
            return packed_inner_product(packed[:10], query_batch, d)
        stages["7c_packed_ip"] = profile_stage("pip", _packed_ip)

    # --- Summary ---
    total = sum(stages.values())
    throughput = n / (total / 1000)

    return stages, total, throughput


def run_profile_sweep(device="cpu"):
    """Run profiling across all configurations."""
    d = 128
    n = 2000
    bits = 3

    configs = [
        ("Haar+Scalar (vanilla)", "haar", "scalar", False, True),
        ("WHT+Scalar", "wht", "scalar", False, True),
        ("NSN+WHT+Scalar", "wht", "scalar", True, True),
        ("E8Block+Scalar", "e8block", "scalar", False, True),
        ("NSN+E8Block+Scalar", "e8block", "scalar", True, True),
        ("WHT+KMeans", "wht", "kmeans", False, False),
        ("CD8+Scalar", "cd8", "scalar", False, True),
    ]

    print(f"\n{'=' * 80}")
    print(f"  TurboQuant Pipeline Profiler  (d={d}, n={n}, bits={bits}, device={device})")
    print(f"{'=' * 80}")

    all_results = []

    for label, rot, quant, nsn, qjl in configs:
        try:
            stages, total, throughput = profile_full_pipeline(
                d, n, bits, device, rot, quant, nsn, qjl,
            )
        except Exception as e:
            print(f"\n  {label}: FAILED ({e})")
            continue

        all_results.append((label, stages, total, throughput))

        print(f"\n  --- {label} ---")
        print(f"  {'Stage':<25} {'Time (ms)':>10} {'%':>6}")
        print(f"  {'-'*25} {'-'*10} {'-'*6}")
        for stage, ms in sorted(stages.items()):
            pct = ms / total * 100
            print(f"  {stage:<25} {ms:>10.2f} {pct:>5.1f}%")
        print(f"  {'TOTAL':<25} {total:>10.2f}")
        print(f"  Throughput: {throughput:,.0f} vectors/sec")

    # Comparison table
    if all_results:
        print(f"\n{'=' * 80}")
        print(f"  COMPARISON TABLE")
        print(f"{'=' * 80}")
        print(f"  {'Config':<30} {'Total (ms)':>12} {'Throughput':>14} {'vs Vanilla':>12}")
        print(f"  {'-'*30} {'-'*12} {'-'*14} {'-'*12}")
        baseline = all_results[0][2] if all_results else 0
        for label, stages, total, throughput in all_results:
            speedup = total / all_results[0][2] if all_results[0][2] > 0 else 0
            vs = f"{1/speedup:.2f}x" if speedup > 0 else "---"
            print(f"  {label:<30} {total:>12.2f} {throughput:>11,.0f}/s {vs:>12}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_profile_sweep(device)
    if device == "cuda":
        print("\n--- Also profiling CPU for comparison ---")
        run_profile_sweep("cpu")

"""
Benchmark suite comparing quantization methods for TurboQuant.

Compares: scalar Lloyd-Max, E8 lattice, and Z^8 prefix-cut codebooks
at various bit rates.

Metrics:
    - Rate-distortion curves (bits/dim vs MSE)
    - Encoding/decoding speed
    - Inner product accuracy after full TurboQuant pipeline
"""

import time
import torch
from typing import Dict, List

from .rotations import HaarRotation
from .lattice_vq import ScalarLloydMaxQuantizer, E8LatticeQuantizer, Z8PrefixCutQuantizer
from .lloyd_max import LloydMaxCodebook


def _time_quantize(quantizer, x: torch.Tensor, n_iters: int = 50) -> float:
    """Time quantize + dequantize cycle in milliseconds."""
    for _ in range(3):
        state = quantizer.quantize(x)
        quantizer.dequantize(state)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        state = quantizer.quantize(x)
        quantizer.dequantize(state)
    return (time.perf_counter() - t0) / n_iters * 1000


def benchmark_quantizer(
    quantizer,
    x_rotated: torch.Tensor,
    name: str,
) -> Dict:
    """
    Run benchmark for a single quantizer on pre-rotated data.

    Args:
        quantizer: VectorQuantizer instance
        x_rotated: rotated vectors, shape (n, d)
        name: display name

    Returns:
        Dict with metrics.
    """
    state = quantizer.quantize(x_rotated)
    x_hat = quantizer.dequantize(state)

    mse = ((x_rotated - x_hat) ** 2).mean().item()
    time_ms = _time_quantize(quantizer, x_rotated)
    bits_per_dim = quantizer.bits_per_dimension()

    # Inner product accuracy
    n = x_rotated.shape[0]
    q = x_rotated[:min(100, n)]
    true_ip = (q.unsqueeze(1) * x_rotated.unsqueeze(0)).sum(dim=-1)
    est_ip = (q.unsqueeze(1) * x_hat.unsqueeze(0)).sum(dim=-1)
    ip_rmse = ((true_ip - est_ip) ** 2).mean().sqrt().item()
    ip_corr = torch.corrcoef(
        torch.stack([true_ip.flatten(), est_ip.flatten()])
    )[0, 1].item()

    return {
        "name": name,
        "bits_per_dim": bits_per_dim,
        "mse": mse,
        "time_ms": time_ms,
        "ip_rmse": ip_rmse,
        "ip_correlation": ip_corr,
    }


def run_full_benchmark(
    d: int = 128,
    n: int = 10000,
    seed: int = 42,
    device: str = "cpu",
) -> List[Dict]:
    """
    Run the full quantizer benchmark.

    Args:
        d: vector dimension (must be divisible by 8)
        n: number of test vectors
        seed: random seed
        device: torch device

    Returns:
        List of benchmark result dicts.
    """
    torch.manual_seed(seed)
    x = torch.randn(n, d, device=device)
    x = x / x.norm(dim=-1, keepdim=True)

    # Rotate with Haar (the rotation is separate from quantization)
    rotation = HaarRotation(d, seed=seed, device=device)
    x_rotated = rotation.rotate(x)

    quantizers = []

    # Scalar Lloyd-Max at various bit-widths
    for bits in [1, 2, 3, 4]:
        quantizers.append((
            f"Lloyd-Max {bits}b",
            ScalarLloydMaxQuantizer(d, bits, device=device),
        ))

    # E8 lattice
    e8 = E8LatticeQuantizer(d, device=device)
    e8.calibrate(x_rotated)
    quantizers.append(("E8 Lattice", e8))

    # Z^8 prefix-cut at various levels
    for level in ["2048", "512", "256", "32"]:
        z8 = Z8PrefixCutQuantizer(d, level=level, device=device)
        z8.calibrate(x_rotated)
        quantizers.append((f"Z8-{level}", z8))

    results = []
    for name, quant in quantizers:
        result = benchmark_quantizer(quant, x_rotated, name)
        results.append(result)

    return results


def print_benchmark_table(results: List[Dict]):
    """Print a formatted comparison table."""
    print(f"\n{'Method':<20} {'Bits/dim':<10} {'MSE':<14} {'Time(ms)':<10} "
          f"{'IP RMSE':<12} {'IP Corr':<10}")
    print("-" * 86)
    for r in results:
        print(f"{r['name']:<20} {r['bits_per_dim']:<10.3f} {r['mse']:<14.8f} "
              f"{r['time_ms']:<10.2f} {r['ip_rmse']:<12.6f} {r['ip_correlation']:<10.6f}")


if __name__ == "__main__":
    results = run_full_benchmark()
    print_benchmark_table(results)

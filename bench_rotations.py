"""
Benchmark suite comparing rotation methods for TurboQuant.

Compares: Haar, WHT, CD quaternion (4D), CD octonion (8D), CD sedenion (16D),
          CD multi-layer, PCA, and Kac rotations.

Metrics:
    - Wall-clock time (rotate + unrotate, averaged over 100 iterations)
    - Storage (number of float parameters)
    - Decorrelation quality (coordinate variance uniformity, cross-correlation)
    - Quantization MSE at 1/2/3/4-bit after rotation + Lloyd-Max
    - Isometry error (norm preservation)
    - Associator norm for CD rotations (measures non-associativity distortion)
"""

import time
import torch
from typing import Dict, List

from .rotations import (
    Rotation, HaarRotation, WHTRotation, CDRotation,
    CDMultiLayerRotation, PCARotation, KacRotation,
)
from .spectral import distribution_analysis, rotation_quality_score
from .lloyd_max import LloydMaxCodebook


def _time_rotation(rotation: Rotation, x: torch.Tensor, n_iters: int = 100) -> float:
    """Time rotate + unrotate cycle in milliseconds."""
    # Warmup
    for _ in range(5):
        y = rotation.rotate(x)
        rotation.unrotate(y)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        y = rotation.rotate(x)
        rotation.unrotate(y)
    elapsed = (time.perf_counter() - t0) / n_iters * 1000
    return elapsed


def _quantization_mse(rotation: Rotation, x: torch.Tensor, bits: int) -> float:
    """Compute quantization MSE after rotation + Lloyd-Max."""
    d = x.shape[-1]
    y = rotation.rotate(x)
    codebook = LloydMaxCodebook(d, bits)
    centroids = codebook.centroids.to(x.device)
    diffs = y.unsqueeze(-1) - centroids
    indices = diffs.abs().argmin(dim=-1)
    y_hat = centroids[indices]
    mse = (y - y_hat).pow(2).mean().item()
    return mse


def benchmark_rotation(
    rotation: Rotation,
    x: torch.Tensor,
    name: str,
    bits_range: List[int] = None,
) -> Dict:
    """
    Run full benchmark suite for a single rotation method.

    Args:
        rotation: the Rotation instance to benchmark
        x: test vectors, shape (n, d), should be unit normalized
        name: display name for this rotation
        bits_range: list of bit-widths to test (default: [1, 2, 3, 4])

    Returns:
        Dict with all metrics.
    """
    if bits_range is None:
        bits_range = [1, 2, 3, 4]

    d = x.shape[-1]
    y = rotation.rotate(x)
    x_recovered = rotation.unrotate(y)

    # Timing
    time_ms = _time_rotation(rotation, x)

    # Storage
    storage = rotation.storage_elements()

    # Quality
    quality = rotation_quality_score(x, y)

    # Invertibility
    invert_error = (x - x_recovered).norm().item() / x.norm().item()

    # Quantization MSE at each bit-width
    quant_mse = {}
    for bits in bits_range:
        quant_mse[bits] = _quantization_mse(rotation, x, bits)

    return {
        "name": name,
        "time_ms": time_ms,
        "storage_elements": storage,
        "storage_compression": d * d / max(storage, 1),
        "invertibility_error": invert_error,
        "variance_uniformity": quality["variance_uniformity"],
        "mean_cross_correlation": quality["mean_cross_correlation"],
        "mean_excess_kurtosis": quality["mean_excess_kurtosis"],
        "isometry_error": quality["isometry_error"],
        "combined_quality": quality["combined_score"],
        "quantization_mse": quant_mse,
    }


def run_full_benchmark(
    d: int = 128,
    n: int = 10000,
    seed: int = 42,
    bits_range: List[int] = None,
    device: str = "cpu",
) -> List[Dict]:
    """
    Run the full rotation benchmark comparing all methods.

    Args:
        d: vector dimension
        n: number of test vectors
        seed: random seed
        bits_range: bit-widths to test
        device: torch device

    Returns:
        List of benchmark result dicts, one per rotation method.
    """
    torch.manual_seed(seed)
    x = torch.randn(n, d, device=device)
    x = x / x.norm(dim=-1, keepdim=True)

    rotations = [
        ("Haar", HaarRotation(d, seed=seed, device=device)),
        ("WHT", WHTRotation(d, seed=seed, device=device)),
        ("CD-Quat (4D)", CDRotation(d, block_dim=4, seed=seed, device=device)),
        ("CD-Oct (8D)", CDRotation(d, block_dim=8, seed=seed, device=device)),
        ("CD-Sed (16D)", CDRotation(d, block_dim=16, seed=seed, device=device)),
        ("CD-Multi (4+8)", CDMultiLayerRotation(d, block_dims=[4, 8], seed=seed, device=device)),
        ("Kac (d*logd)", KacRotation(d, seed=seed, device=device)),
    ]

    # PCA needs calibration
    pca = PCARotation(d, device=device)
    pca.calibrate(x)
    rotations.append(("PCA", pca))

    results = []
    for name, rotation in rotations:
        result = benchmark_rotation(rotation, x, name, bits_range)
        results.append(result)

    return results


def print_benchmark_table(results: List[Dict]):
    """Print a formatted comparison table."""
    print(f"\n{'Method':<20} {'Time(ms)':<10} {'Storage':<10} {'InvErr':<12} "
          f"{'VarUnif':<10} {'CrossCorr':<10} {'IsoErr':<10}")
    print("-" * 92)
    for r in results:
        print(f"{r['name']:<20} {r['time_ms']:<10.2f} {r['storage_elements']:<10} "
              f"{r['invertibility_error']:<12.2e} {r['variance_uniformity']:<10.6f} "
              f"{r['mean_cross_correlation']:<10.6f} {r['isometry_error']:<10.6f}")

    print(f"\n{'Method':<20}", end="")
    bits_keys = sorted(results[0]["quantization_mse"].keys())
    for b in bits_keys:
        print(f" MSE@{b}bit    ", end="")
    print()
    print("-" * (20 + 14 * len(bits_keys)))
    for r in results:
        print(f"{r['name']:<20}", end="")
        for b in bits_keys:
            print(f" {r['quantization_mse'][b]:<12.8f}", end="")
        print()


if __name__ == "__main__":
    results = run_full_benchmark()
    print_benchmark_table(results)

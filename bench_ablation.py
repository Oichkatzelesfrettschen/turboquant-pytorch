#!/usr/bin/env python3
"""
Ablation study: measure cumulative impact of each feature on real model KV cache.

Layers features one at a time to measure individual and combined contributions:
    Layer 0: Haar + Scalar (vanilla baseline)
    Layer 1: WHT + Scalar (structured rotation)
    Layer 2: + NSN pre-processing (distribution normalization)
    Layer 3: + sign packing (8x QJL memory reduction)
    Layer 4: + post-VQ scaling (angular preservation)
    Layer 5: + CD fidelity metric (diagnostic)

Measures: cosine similarity, top-1/top-5 match, memory bytes, throughput.
"""

import gc
import os
import sys
import importlib.util
import math
import time

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "turboquant", os.path.join(_pkg_dir, "__init__.py"),
    submodule_search_locations=[_pkg_dir],
)
_tq = importlib.util.module_from_spec(_spec)
sys.modules["turboquant"] = _tq
_spec.loader.exec_module(_tq)

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from turboquant.rotations import HaarRotation, WHTRotation
from turboquant.lloyd_max import LloydMaxCodebook
from turboquant.nsn_preprocess import nsn_preprocess, nsn_restore, adaptive_vq_scale
from turboquant.sign_pack import pack_signs, unpack_signs, packed_inner_product
from turboquant.cd_fidelity import fidelity_summary, distortion_decomposition


def load_model(model_name="Qwen/Qwen2.5-3B-Instruct"):
    """Load model and extract KV cache."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            ),
            device_map="auto", torch_dtype=torch.float16,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto",
        )
    model.eval()

    prompt = "The quick brown fox " * 100 + "The secret code is AURORA. " + "The quick brown fox " * 100
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    cache = outputs.past_key_values
    return cache, model, tokenizer


def extract_layer_kv(cache, layer_idx):
    """Extract keys and values from a cache layer, handling API versions."""
    if hasattr(cache, 'layers') and hasattr(cache.layers[0], 'keys'):
        return cache.layers[layer_idx].keys, cache.layers[layer_idx].values
    elif hasattr(cache, 'key_cache'):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    else:
        return cache[layer_idx][0], cache[layer_idx][1]


def run_ablation(cache, bits=3):
    """Run the ablation study across all feature combinations."""
    # Get dimensions from first layer
    keys_0, _ = extract_layer_kv(cache, 0)
    B, H, S, D = keys_0.shape
    n_layers = len(cache.layers) if hasattr(cache, 'layers') else len(cache)

    print(f"\n  Model: {n_layers} layers, {H} KV heads, seq={S}, d={D}, bits={bits}")
    print(f"  Original KV size: {n_layers * H * S * D * 2 * 2 / 1024 / 1024:.1f} MB (FP16 K+V)")

    # Configs to test (cumulative layers)
    configs = [
        {
            "name": "L0: Haar+Scalar (vanilla)",
            "rotation": "haar",
            "use_nsn": False,
            "use_sign_pack": False,
            "use_post_vq_scale": False,
            "measure_fidelity": False,
        },
        {
            "name": "L1: WHT+Scalar (rotation)",
            "rotation": "wht",
            "use_nsn": False,
            "use_sign_pack": False,
            "use_post_vq_scale": False,
            "measure_fidelity": False,
        },
        {
            "name": "L2: +NSN pre-processing",
            "rotation": "wht",
            "use_nsn": True,
            "use_sign_pack": False,
            "use_post_vq_scale": False,
            "measure_fidelity": False,
        },
        {
            "name": "L3: +sign packing (8x mem)",
            "rotation": "wht",
            "use_nsn": True,
            "use_sign_pack": True,
            "use_post_vq_scale": False,
            "measure_fidelity": False,
        },
        {
            "name": "L4: +post-VQ scaling",
            "rotation": "wht",
            "use_nsn": True,
            "use_sign_pack": True,
            "use_post_vq_scale": True,
            "measure_fidelity": False,
        },
        {
            "name": "L5: +CD fidelity diagnostic",
            "rotation": "wht",
            "use_nsn": True,
            "use_sign_pack": True,
            "use_post_vq_scale": True,
            "measure_fidelity": True,
        },
    ]

    results = []

    for cfg in configs:
        cosine_sims = []
        top1_ok = 0
        top5_ok = 0
        n_checks = 0
        total_compressed_bytes = 0
        total_original_bytes = 0
        fidelity_ratios = []
        mag_distortions = []
        phase_distortions = []

        t_start = time.perf_counter()

        for layer_idx in range(n_layers):
            keys, values = extract_layer_kv(cache, layer_idx)
            B_l, H_l, S_l, D_l = keys.shape

            for h in range(H_l):
                seed = layer_idx * 1000 + h
                head_keys = keys[:, h, :, :].float()  # (B, S, D)
                head_query = head_keys[:, -1:, :]      # (B, 1, D)

                # Real attention scores
                real_scores = (head_query @ head_keys.transpose(-2, -1)).squeeze(1)  # (B, S)

                # --- NSN pre-processing ---
                if cfg["use_nsn"]:
                    head_flat = head_keys.reshape(-1, D_l)
                    head_nsn, nsn_state = nsn_preprocess(head_flat)
                    to_quantize = head_nsn
                else:
                    head_flat = head_keys.reshape(-1, D_l)
                    to_quantize = head_flat
                    nsn_state = None

                # --- Rotation ---
                if cfg["rotation"] == "wht":
                    rot = WHTRotation(D_l, seed=seed, device=keys.device)
                else:
                    rot = HaarRotation(D_l, seed=seed, device=keys.device)

                rotated = rot.rotate(to_quantize)

                # --- Quantize ---
                mse_bits = max(bits - 1, 1)
                codebook = LloydMaxCodebook(D_l, mse_bits)
                centroids = codebook.centroids.to(keys.device)
                boundaries = ((centroids[:-1] + centroids[1:]) / 2)
                indices = torch.searchsorted(boundaries, rotated)

                # --- Dequantize ---
                recon_rotated = centroids[indices]

                # --- Post-VQ scaling ---
                if cfg["use_post_vq_scale"]:
                    recon_rotated = adaptive_vq_scale(rotated, recon_rotated)

                # --- Unrotate ---
                recon_flat = rot.unrotate(recon_rotated)

                # --- NSN restore ---
                if cfg["use_nsn"] and nsn_state is not None:
                    recon_flat = nsn_restore(recon_flat, nsn_state)

                # --- QJL residual ---
                residual = head_flat - recon_flat
                gen = torch.Generator(device="cpu")
                gen.manual_seed(seed + 10000)
                S_mat = torch.randn(D_l, D_l, generator=gen).to(keys.device)
                projected = residual @ S_mat.T
                signs = torch.sign(projected)
                signs[signs == 0] = 1.0
                r_norm = residual.norm(dim=-1)

                # --- Sign packing ---
                if cfg["use_sign_pack"]:
                    packed = pack_signs(signs)
                    sign_bytes = packed.numel() * 8  # int64
                else:
                    sign_bytes = signs.numel() * 1  # int8 equivalent

                # --- Compute attention scores ---
                k_mse = recon_flat.reshape(B_l, S_l, D_l)
                term1 = (head_query @ k_mse.transpose(-2, -1)).squeeze(1)

                q_proj = (head_query.reshape(-1, D_l) @ S_mat.T)
                correction_scale = math.sqrt(math.pi / 2) / D_l
                if cfg["use_sign_pack"]:
                    # Unpack for IP (in real deployment, use packed_inner_product)
                    signs_unpacked = unpack_signs(packed, D_l)
                    qjl_ip = (q_proj @ signs_unpacked.T)
                else:
                    qjl_ip = (q_proj @ signs.T)

                term2 = correction_scale * qjl_ip * r_norm.unsqueeze(0)
                tq_scores = (term1 + term2).squeeze(0)  # (B, S) -> (S,)
                if tq_scores.dim() > 1:
                    tq_scores = tq_scores[0]

                rs = real_scores[0]
                ts = tq_scores

                cos = F.cosine_similarity(rs.unsqueeze(0), ts.unsqueeze(0)).item()
                cosine_sims.append(cos)
                if rs.argmax().item() == ts.argmax().item():
                    top1_ok += 1
                if rs.argmax().item() in ts.topk(min(5, S_l)).indices.tolist():
                    top5_ok += 1
                n_checks += 1

                # Memory
                n_vecs = B_l * S_l
                idx_bytes = n_vecs * D_l * mse_bits // 8
                norm_bytes = n_vecs * 2  # fp16
                total_compressed_bytes += idx_bytes + sign_bytes + norm_bytes
                total_original_bytes += n_vecs * D_l * 2  # fp16

                # --- CD fidelity diagnostic ---
                if cfg["measure_fidelity"] and S_l >= 3:
                    orig = head_flat[:min(50, S_l)]
                    recon = recon_flat[:min(50, S_l)]
                    if orig.shape[0] >= 3 and D_l >= 16:
                        summary = fidelity_summary(orig[:, :16], recon[:, :16])
                        fidelity_ratios.append(summary.mean_ratio)
                    decomp = distortion_decomposition(orig, recon)
                    mag_distortions.append(decomp["magnitude"].mean().item())
                    phase_distortions.append(decomp["phase"].mean().item())

        elapsed = time.perf_counter() - t_start
        avg_cos = sum(cosine_sims) / len(cosine_sims)
        top1_pct = 100 * top1_ok / n_checks
        top5_pct = 100 * top5_ok / n_checks
        compression = total_original_bytes / total_compressed_bytes

        result = {
            "name": cfg["name"],
            "cosine_sim": avg_cos,
            "top1_pct": top1_pct,
            "top5_pct": top5_pct,
            "compression": compression,
            "compressed_mb": total_compressed_bytes / 1024 / 1024,
            "original_mb": total_original_bytes / 1024 / 1024,
            "time_s": elapsed,
        }

        if fidelity_ratios:
            result["fidelity_ratio"] = sum(fidelity_ratios) / len(fidelity_ratios)
            result["mag_distortion"] = sum(mag_distortions) / len(mag_distortions)
            result["phase_distortion"] = sum(phase_distortions) / len(phase_distortions)

        results.append(result)

    return results


def print_results(results):
    """Print ablation results."""
    print(f"\n  {'Config':<35} {'CosSim':>8} {'Top1%':>6} {'Top5%':>6} {'Compr':>6} {'MB':>6} {'Time':>6}")
    print(f"  {'-'*35} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    baseline_cos = results[0]["cosine_sim"] if results else 0
    baseline_mb = results[0]["compressed_mb"] if results else 1

    for r in results:
        delta_cos = (r["cosine_sim"] - baseline_cos) * 100
        delta_mb = (r["compressed_mb"] - baseline_mb) / baseline_mb * 100
        print(f"  {r['name']:<35} {r['cosine_sim']:>8.4f} {r['top1_pct']:>5.1f}% {r['top5_pct']:>5.1f}% "
              f"{r['compression']:>5.1f}x {r['compressed_mb']:>5.1f} {r['time_s']:>5.1f}s")

        if "fidelity_ratio" in r:
            print(f"    CD fidelity: {r['fidelity_ratio']:.4f}  "
                  f"mag_dist: {r['mag_distortion']:.6f}  "
                  f"phase_dist: {r['phase_distortion']:.6f}")

    # Delta table
    print(f"\n  --- Cumulative Impact vs Vanilla ---")
    print(f"  {'Config':<35} {'dCosSim':>10} {'dTop1':>8} {'dMem%':>8}")
    print(f"  {'-'*35} {'-'*10} {'-'*8} {'-'*8}")
    for r in results:
        dc = (r["cosine_sim"] - baseline_cos) * 10000  # basis points
        dt1 = r["top1_pct"] - results[0]["top1_pct"]
        dm = (r["compressed_mb"] - baseline_mb) / baseline_mb * 100
        print(f"  {r['name']:<35} {dc:>+9.1f}bp {dt1:>+7.1f}% {dm:>+7.1f}%")


if __name__ == "__main__":
    print(f"\n{'=' * 80}")
    print(f"  Ablation Study: Cumulative Feature Impact on Qwen2.5-3B KV Cache")
    print(f"{'=' * 80}")

    cache, model, tokenizer = load_model()
    del model  # free model VRAM, keep cache
    gc.collect()
    torch.cuda.empty_cache()

    for bits in [3]:
        results = run_ablation(cache, bits=bits)
        print_results(results)

    print(f"\n{'=' * 80}")
    print(f"  ABLATION COMPLETE")
    print(f"{'=' * 80}\n")

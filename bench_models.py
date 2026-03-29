#!/usr/bin/env python3
"""
Multi-model KV cache quantization benchmark: vanilla vs modified TurboQuant.

Tests on multiple models to prove generality:
    1. Qwen2.5-3B-Instruct (already validated)
    2. Mistral-7B-Instruct-v0.3
    3. Llama-3.2-3B-Instruct (or Llama-3.1-8B if VRAM allows)

For each model, compares:
    - Haar + Scalar (vanilla TurboQuant, baseline)
    - WHT + Scalar (our structured rotation via cuBLAS materialized)
    - CD-Oct + Scalar (our algebraic rotation)

Metrics:
    - Attention score cosine similarity (compressed vs original)
    - Top-1 and Top-5 attention head match rates
    - Compression ratio
    - KV cache memory savings
"""

import gc
import os
import sys
import importlib.util
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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from turboquant.compressors import TurboQuantCompressorV2
from turboquant.rotations import HaarRotation, WHTRotation, CDRotation


# Ungated models that don't require HF token
MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/phi-2",  # 2.7B, ungated, different architecture
]

PROMPT_TEMPLATE = (
    "The quick brown fox jumps over the lazy dog. " * 50
    + "The secret code is AURORA-7749. "
    + "The quick brown fox jumps over the lazy dog. " * 50
)


def load_model(model_name, device="cuda"):
    """Load model in 4-bit with BitsAndBytes, or FP16 fallback."""
    print(f"  Loading {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            ),
            device_map="auto",
            torch_dtype=torch.float16,
        )
        load_mode = "4-bit"
    except Exception as e:
        print(f"    4-bit failed ({type(e).__name__}), trying FP16...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        load_mode = "FP16"

    model.eval()
    vram = torch.cuda.memory_allocated() // (1024 * 1024)
    print(f"    Loaded ({load_mode}). VRAM: {vram} MB")
    return model, tokenizer, load_mode


def extract_kv_cache(model, tokenizer, prompt, device="cuda"):
    """Run a forward pass and extract the KV cache."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    seq_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    cache = outputs.past_key_values

    # Handle Transformers 5.0 DynamicCache (cache.layers[i].keys/values)
    # and older tuple-of-tuples format
    if hasattr(cache, 'layers') and hasattr(cache.layers[0], 'keys'):
        # Transformers 5.0+ DynamicCache
        keys_0 = cache.layers[0].keys
        n_layers = len(cache.layers)
    elif hasattr(cache, 'key_cache'):
        keys_0 = cache.key_cache[0]
        n_layers = len(cache.key_cache)
    else:
        keys_0 = cache[0][0]
        n_layers = len(cache)
    B, n_heads, S, head_dim = keys_0.shape

    print(f"    KV cache: {n_layers} layers, {n_heads} heads, seq={S}, head_dim={head_dim}")
    return cache, n_layers, n_heads, S, head_dim


def benchmark_config(cache, n_layers, n_heads, head_dim, config_name, rot_factory, bits, device):
    """Benchmark a single rotation + quantizer config on the KV cache."""
    cosine_sims = []
    top1_matches = 0
    top5_matches = 0
    n_checks = 0
    total_compressed_bits = 0
    total_original_bits = 0

    for layer_idx in range(n_layers):
        if hasattr(cache, 'layers') and hasattr(cache.layers[0], 'keys'):
            keys = cache.layers[layer_idx].keys
        elif hasattr(cache, 'key_cache'):
            keys = cache.key_cache[layer_idx]
        else:
            keys = cache[layer_idx][0]

        B, H, S, D = keys.shape
        query = keys[:, :, -1:, :]

        real_scores = torch.matmul(
            query.float(), keys.float().transpose(-2, -1)
        ).squeeze(-2)

        for h in range(H):
            seed = layer_idx * 1000 + h
            rot = rot_factory(D, seed) if rot_factory else None

            comp = TurboQuantCompressorV2(
                D, bits, seed=seed, device=device, rotation=rot,
            )

            head_keys = keys[:, h:h+1, :, :]
            compressed = comp.compress(head_keys)
            head_query = query[:, h:h+1, :, :]
            tq_scores = comp.asymmetric_attention_scores(
                head_query, compressed
            ).squeeze(-2)

            rs = real_scores[0, h]
            ts = tq_scores[0, 0]

            cos = F.cosine_similarity(rs.unsqueeze(0), ts.unsqueeze(0)).item()
            cosine_sims.append(cos)

            if rs.argmax().item() == ts.argmax().item():
                top1_matches += 1
            if rs.argmax().item() in ts.topk(min(5, S)).indices.tolist():
                top5_matches += 1
            n_checks += 1

        # Memory accounting
        n_vecs = B * H * S
        mse_bits = max(bits - 1, 1)
        total_compressed_bits += n_vecs * D * mse_bits + n_vecs * D + n_vecs * 32
        total_original_bits += n_vecs * D * 16

    avg_cos = sum(cosine_sims) / len(cosine_sims)
    top1_pct = 100 * top1_matches / n_checks
    top5_pct = 100 * top5_matches / n_checks
    compression = total_original_bits / total_compressed_bits

    return {
        "config": config_name,
        "bits": bits,
        "cosine_sim": avg_cos,
        "top1_pct": top1_pct,
        "top5_pct": top5_pct,
        "compression": compression,
    }


def run_model_benchmark(model_name, device="cuda"):
    """Run full benchmark on a single model."""
    model, tokenizer, load_mode = load_model(model_name, device)
    cache, n_layers, n_heads, S, head_dim = extract_kv_cache(
        model, tokenizer, PROMPT_TEMPLATE, device
    )

    configs = [
        ("Haar (vanilla)", None),
        ("WHT (ours)", lambda d, s: WHTRotation(d, seed=s, device=device)),
        ("CD-Oct (ours)", lambda d, s: CDRotation(d, block_dim=8, seed=s, device=device)),
    ]

    results = []
    for config_name, rot_factory in configs:
        for bits in [2, 3, 4]:
            r = benchmark_config(
                cache, n_layers, n_heads, head_dim,
                config_name, rot_factory, bits, device,
            )
            results.append(r)
            print(f"    {config_name:<20} {bits}b: cos={r['cosine_sim']:.4f} "
                  f"top1={r['top1_pct']:.1f}% top5={r['top5_pct']:.1f}% "
                  f"{r['compression']:.1f}x", flush=True)

    # Free model VRAM
    del model, cache
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'=' * 80}")
    print(f"  Multi-Model KV Cache Quantization Benchmark")
    print(f"  Device: {torch.cuda.get_device_name() if device == 'cuda' else 'CPU'}")
    print(f"  VRAM: {torch.cuda.mem_get_info()[0]/1024**3:.1f} GB free")
    print(f"{'=' * 80}")

    all_results = {}

    for model_name in MODELS:
        print(f"\n  --- {model_name} ---")
        try:
            results = run_model_benchmark(model_name, device)
            all_results[model_name] = results
        except Exception as e:
            print(f"    FAILED: {e}")
            all_results[model_name] = None
            gc.collect()
            torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY: WHT vs Haar (vanilla) at 3-bit")
    print(f"{'=' * 80}")
    print(f"  {'Model':<40} {'Haar cos':>10} {'WHT cos':>10} {'Delta':>8}")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*8}")

    for model_name, results in all_results.items():
        if results is None:
            print(f"  {model_name:<40} {'FAILED':>10}")
            continue
        haar_3b = [r for r in results if r["config"] == "Haar (vanilla)" and r["bits"] == 3]
        wht_3b = [r for r in results if r["config"] == "WHT (ours)" and r["bits"] == 3]
        if haar_3b and wht_3b:
            h = haar_3b[0]["cosine_sim"]
            w = wht_3b[0]["cosine_sim"]
            delta = (w - h) * 100
            short = model_name.split("/")[-1]
            print(f"  {short:<40} {h:>10.4f} {w:>10.4f} {delta:>+7.2f}%")

    print(f"\n{'=' * 80}")
    print(f"  DONE")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()

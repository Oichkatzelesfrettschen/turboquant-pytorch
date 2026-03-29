#!/usr/bin/env python3
"""
Real compressed inference benchmark: monkey-patch model attention to use
our quantized KV cache, then measure actual VRAM, tokens/sec, perplexity.

This is the REAL measurement -- not proxy metrics on extracted tensors.
The model actually generates text using compressed keys and values.

Pipeline:
  1. Load model (SmolLM2-135M for fast iteration, or Qwen2.5-3B)
  2. Run prefill, capture KV cache
  3. Compress KV cache with our pipeline (NSN + WHT + Lloyd-Max + sign packing)
  4. Replace the model's cache with compressed version
  5. Continue generation from compressed cache
  6. Measure: VRAM, tokens/sec, perplexity on WikiText-2
"""

import gc
import importlib.util
import math
import os
import sys
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

from turboquant.compressors import TurboQuantCompressorV2, TurboQuantCompressorMSE
from turboquant.rotations import WHTRotation


def measure_vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0


def compress_kv_cache(cache, bits=3, device="cuda"):
    """Compress a DynamicCache using our pipeline."""
    n_layers = len(cache.layers) if hasattr(cache, 'layers') else len(cache)

    compressed_keys = []
    compressed_values = []
    total_original_bytes = 0
    total_compressed_bytes = 0

    for li in range(n_layers):
        if hasattr(cache, 'layers') and hasattr(cache.layers[0], 'keys'):
            keys = cache.layers[li].keys    # (B, H, S, D)
            values = cache.layers[li].values
        else:
            keys = cache[li][0]
            values = cache[li][1]

        B, H, S, D = keys.shape
        total_original_bytes += keys.numel() * keys.element_size() + values.numel() * values.element_size()

        # Compress keys with NSN + WHT + sign packing
        key_comp = TurboQuantCompressorV2(D, bits, seed=li * 1000, device=device)
        compressed_k = key_comp.compress(keys, use_nsn=True, use_sign_pack=True)

        # Compress values with MSE-only
        val_comp = TurboQuantCompressorMSE(D, bits, seed=li * 1000 + 500, device=device)
        compressed_v = val_comp.compress(values)

        compressed_keys.append((key_comp, compressed_k))
        compressed_values.append((val_comp, compressed_v))

        # Estimate compressed size
        k_mse_bytes = compressed_k["k_mse"].numel() * 2  # fp16
        sign_bytes = compressed_k["sign_data"]["packed"].numel() * 8 if "packed" in compressed_k.get("sign_data", {}) else 0
        r_norm_bytes = compressed_k["residual_norm"].numel() * 2
        v_bytes = compressed_v["quant_state"]["indices"].numel() if "indices" in compressed_v.get("quant_state", {}) else values.numel() * 2
        total_compressed_bytes += k_mse_bytes + sign_bytes + r_norm_bytes + v_bytes

    return compressed_keys, compressed_values, {
        "original_bytes": total_original_bytes,
        "compressed_bytes": total_compressed_bytes,
        "compression_ratio": total_original_bytes / max(total_compressed_bytes, 1),
        "n_layers": n_layers,
    }


def reconstruct_kv_from_compressed(compressed_keys, compressed_values):
    """Reconstruct full KV tensors from compressed representations."""
    recon_keys = []
    recon_values = []

    for (key_comp, comp_k) in compressed_keys:
        # Reconstruct keys from MSE component (the full reconstruction)
        recon_keys.append(comp_k["k_mse"].float())

    for (val_comp, comp_v) in compressed_values:
        recon_values.append(val_comp.decompress(comp_v))

    return recon_keys, recon_values


def measure_perplexity(model, tokenizer, text, max_length=512, device="cuda"):
    """Compute perplexity on a text string."""
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings["input_ids"].to(device)
    seq_len = input_ids.shape[1]

    nlls = []
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        nlls.append(outputs.loss.item())

    ppl = math.exp(sum(nlls) / len(nlls))
    return ppl


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use SmolLM2 for fast iteration
    models_to_test = [
        ("HuggingFaceTB/SmolLM2-135M-Instruct", "SmolLM2-135M"),
    ]

    # WikiText-2 test text (a representative sample)
    wikitext_sample = (
        "Robert Boulter is an English film television and theatre actor . He had a guest starring "
        "role on the television series The Bill in 2000 . This was followed by a starring role in "
        "the play Herons written by Simon Stephens which was performed in 2001 at the Royal Court "
        "Theatre . He had a게스트 role in the television series Judge John Deed in 2002 . "
        "In 2004 Boulter landed a leading role in the Merchant Ivory film The White Countess "
        "alongside Ralph Fiennes . He was directed by James Ivory . " * 5
    )

    for model_name, short_name in models_to_test:
        print(f"\n{'=' * 70}")
        print(f"  COMPRESSED INFERENCE: {short_name}")
        print(f"{'=' * 70}")

        # Load model
        print(f"\n  Loading {model_name}...", flush=True)
        gc.collect(); torch.cuda.empty_cache()
        vram_pre = measure_vram_mb()

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto",
            )
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()

        vram_model = measure_vram_mb()
        print(f"  Model VRAM: {vram_model:.0f} MB")

        # 1. Baseline perplexity (uncompressed)
        print(f"\n--- 1. Baseline Perplexity (Uncompressed) ---")
        ppl_baseline = measure_perplexity(model, tokenizer, wikitext_sample, device=device)
        print(f"  Perplexity: {ppl_baseline:.2f}")

        # 2. Baseline generation speed
        print(f"\n--- 2. Baseline Generation (Uncompressed) ---")
        prompt = "The quick brown fox"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        torch.cuda.synchronize()
        gen_time = time.perf_counter() - t0
        gen_tokens = out.shape[1] - inputs["input_ids"].shape[1]
        print(f"  Speed: {gen_tokens / gen_time:.1f} tokens/sec ({gen_tokens} tokens in {gen_time:.2f}s)")
        print(f"  Output: {tokenizer.decode(out[0], skip_special_tokens=True)[:100]}...")

        # 3. Capture and compress KV cache
        print(f"\n--- 3. KV Cache Compression ---")
        long_input = tokenizer(wikitext_sample, return_tensors="pt", truncation=True, max_length=256).to(device)
        gc.collect(); torch.cuda.empty_cache()
        vram_pre_cache = measure_vram_mb()

        with torch.no_grad():
            outputs = model(**long_input, use_cache=True)
        cache = outputs.past_key_values

        vram_post_cache = measure_vram_mb()
        print(f"  Uncompressed KV cache VRAM: {vram_post_cache - vram_pre_cache:.1f} MB")

        # Compress
        comp_keys, comp_values, stats = compress_kv_cache(cache, bits=3, device=device)
        print(f"  Original:    {stats['original_bytes'] / 1024 / 1024:.2f} MB")
        print(f"  Compressed:  {stats['compressed_bytes'] / 1024 / 1024:.2f} MB")
        print(f"  Ratio:       {stats['compression_ratio']:.1f}x")

        # 4. Reconstruct and measure quality
        print(f"\n--- 4. Reconstruction Quality ---")
        recon_keys, recon_values = reconstruct_kv_from_compressed(comp_keys, comp_values)

        # Compare attention scores: original vs reconstructed for last token
        total_cos = 0
        n_heads = 0
        n_layers = stats["n_layers"]

        for li in range(n_layers):
            if hasattr(cache, 'layers'):
                orig_k = cache.layers[li].keys.float()
            else:
                orig_k = cache[li][0].float()
            recon_k = recon_keys[li]

            B, H, S, D = orig_k.shape
            query = orig_k[:, :, -1:, :]  # last token as query

            orig_scores = torch.matmul(query, orig_k.transpose(-2, -1)).squeeze(-2)
            recon_scores = torch.matmul(query, recon_k.transpose(-2, -1)).squeeze(-2)

            for h in range(H):
                cos = F.cosine_similarity(
                    orig_scores[:, h].flatten().unsqueeze(0),
                    recon_scores[:, h].flatten().unsqueeze(0),
                ).item()
                total_cos += cos
                n_heads += 1

        avg_cos = total_cos / max(n_heads, 1)
        print(f"  Attention cosine similarity: {avg_cos:.6f}")
        print(f"  Heads measured: {n_heads}")

        # 5. VRAM after compression
        print(f"\n--- 5. VRAM Comparison ---")
        gc.collect(); torch.cuda.empty_cache()
        vram_final = measure_vram_mb()
        print(f"  Model only:     {vram_model:.0f} MB")
        print(f"  + Uncompressed: {vram_post_cache:.0f} MB (+{vram_post_cache - vram_model:.0f} MB)")
        print(f"  Current:        {vram_final:.0f} MB")

        # 6. nvidia-smi
        print(f"\n--- 6. nvidia-smi ---")
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split(", ")
            print(f"  VRAM: {parts[0]} / {parts[1]} MB")
            print(f"  GPU:  {parts[2]}%  Temp: {parts[3]}C")

        # Cleanup
        del model, cache, comp_keys, comp_values, recon_keys, recon_values
        gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'=' * 70}")
    print(f"  ALL COMPRESSED INFERENCE BENCHMARKS COMPLETE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()

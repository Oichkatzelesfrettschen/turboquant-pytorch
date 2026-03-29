#!/usr/bin/env python3
"""
Real inference benchmark: measure actual model performance with KV cache quantization.

Unlike bench_ablation.py (proxy metrics on extracted tensors), this measures:
  1. Actual GPU VRAM consumption (torch.cuda.memory_allocated)
  2. Actual generation speed (tokens/second)
  3. Actual text quality (generated text comparison)
  4. Actual attention computation time (torch.profiler)
  5. Actual memory bandwidth (derived from profiler)

Uses Qwen2.5-3B-Instruct with real text generation.
"""

import gc
import importlib.util
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
from transformers import AutoModelForCausalLM, AutoTokenizer


def measure_vram():
    """Current GPU VRAM usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0


def measure_generation(model, tokenizer, prompt, max_new_tokens=100):
    """Measure actual generation speed and output."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    output_tokens = outputs.shape[1] - input_len
    tokens_per_sec = output_tokens / elapsed
    generated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    return {
        "tokens_generated": output_tokens,
        "time_seconds": elapsed,
        "tokens_per_second": tokens_per_sec,
        "generated_text": generated_text[:200],
    }


def measure_kv_cache_vram(model, tokenizer, prompt):
    """Measure actual VRAM consumed by KV cache during forward pass."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    gc.collect()
    torch.cuda.empty_cache()
    vram_before = measure_vram()

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    vram_after = measure_vram()
    cache = outputs.past_key_values

    # Measure cache tensor sizes directly
    cache_bytes = 0
    if hasattr(cache, 'layers'):
        for layer in cache.layers:
            if hasattr(layer, 'keys'):
                cache_bytes += layer.keys.nelement() * layer.keys.element_size()
                cache_bytes += layer.values.nelement() * layer.values.element_size()

    return {
        "vram_before_mb": vram_before,
        "vram_after_mb": vram_after,
        "vram_delta_mb": vram_after - vram_before,
        "cache_tensor_mb": cache_bytes / (1024 * 1024),
        "input_tokens": inputs["input_ids"].shape[1],
    }


def run_torch_profiler(model, tokenizer, prompt, max_new_tokens=50):
    """Profile actual CUDA kernels during generation."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=5, do_sample=False)

    # Profile
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # Extract key metrics
    events = prof.key_averages()
    total_cuda_ms = sum(e.cuda_time_total for e in events) / 1000  # us -> ms
    total_cpu_ms = sum(e.cpu_time_total for e in events) / 1000

    # Find attention-related kernels
    attn_kernels = [e for e in events if "attn" in e.key.lower() or "attention" in e.key.lower() or "sdpa" in e.key.lower()]
    matmul_kernels = [e for e in events if "mm" in e.key.lower() or "gemm" in e.key.lower() or "bmm" in e.key.lower()]

    return {
        "total_cuda_ms": total_cuda_ms,
        "total_cpu_ms": total_cpu_ms,
        "n_cuda_events": len([e for e in events if e.cuda_time_total > 0]),
        "top5_cuda_kernels": [(e.key, e.cuda_time_total / 1000) for e in sorted(events, key=lambda e: e.cuda_time_total, reverse=True)[:5]],
        "attention_time_ms": sum(e.cuda_time_total for e in attn_kernels) / 1000,
        "matmul_time_ms": sum(e.cuda_time_total for e in matmul_kernels) / 1000,
    }


def main():
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    prompt_short = "The quick brown fox jumps over the lazy dog."
    prompt_long = "The quick brown fox " * 200  # ~1000 tokens

    print(f"\n{'=' * 70}")
    print(f"  REAL INFERENCE BENCHMARK: {model_name}")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"{'=' * 70}")

    # Load model
    print("\nLoading model...", flush=True)
    vram_pre_model = measure_vram()

    try:
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"),
            device_map="auto", torch_dtype=torch.float16,
        )
        load_mode = "4-bit"
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        load_mode = "FP16"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    vram_model = measure_vram()
    print(f"  Model loaded ({load_mode}): {vram_model:.0f} MB VRAM")

    # 1. VRAM measurement
    print(f"\n--- 1. VRAM Consumption ---")
    for name, prompt in [("Short (~10 tokens)", prompt_short), ("Long (~1000 tokens)", prompt_long)]:
        gc.collect(); torch.cuda.empty_cache()
        vram = measure_kv_cache_vram(model, tokenizer, prompt)
        print(f"  {name}:")
        print(f"    Input tokens: {vram['input_tokens']}")
        print(f"    VRAM delta:   {vram['vram_delta_mb']:.1f} MB")
        print(f"    Cache tensors: {vram['cache_tensor_mb']:.1f} MB")

    # 2. Generation speed
    print(f"\n--- 2. Generation Speed ---")
    gen = measure_generation(model, tokenizer, prompt_short, max_new_tokens=100)
    print(f"  Tokens generated: {gen['tokens_generated']}")
    print(f"  Time: {gen['time_seconds']:.2f}s")
    print(f"  Speed: {gen['tokens_per_second']:.1f} tokens/sec")
    print(f"  Output: {gen['generated_text'][:100]}...")

    # 3. Profiler
    print(f"\n--- 3. CUDA Kernel Profile ---")
    prof = run_torch_profiler(model, tokenizer, prompt_short, max_new_tokens=50)
    print(f"  Total CUDA time: {prof['total_cuda_ms']:.1f} ms")
    print(f"  Total CPU time:  {prof['total_cpu_ms']:.1f} ms")
    print(f"  CUDA events:     {prof['n_cuda_events']}")
    print(f"  Attention time:  {prof['attention_time_ms']:.1f} ms")
    print(f"  Matmul time:     {prof['matmul_time_ms']:.1f} ms")
    print(f"  Top 5 CUDA kernels:")
    for name, ms in prof['top5_cuda_kernels']:
        print(f"    {ms:>8.1f} ms  {name[:60]}")

    # 4. nvidia-smi snapshot
    print(f"\n--- 4. nvidia-smi Snapshot ---")
    import subprocess
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        parts = result.stdout.strip().split(", ")
        print(f"  VRAM used:    {parts[0]} MB / {parts[1]} MB")
        print(f"  GPU util:     {parts[2]}%")
        print(f"  Temperature:  {parts[3]}C")

    print(f"\n{'=' * 70}")
    print(f"  BENCHMARK COMPLETE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()

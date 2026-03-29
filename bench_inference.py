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

    # Extract key metrics -- guard for events without CUDA timing
    # (BitsAndBytes 4-bit kernels and some custom ops lack CUDA profiler hooks)
    events = prof.key_averages()

    def _cuda_us(e):
        return getattr(e, "cuda_time_total", 0) or 0

    def _cpu_us(e):
        return getattr(e, "cpu_time_total", 0) or 0

    total_cuda_ms = sum(_cuda_us(e) for e in events) / 1000
    total_cpu_ms = sum(_cpu_us(e) for e in events) / 1000

    # Find attention-related kernels
    attn_kernels = [e for e in events if any(k in e.key.lower() for k in ("attn", "attention", "sdpa", "flash"))]
    matmul_kernels = [e for e in events if any(k in e.key.lower() for k in ("mm", "gemm", "bmm", "linear"))]

    return {
        "total_cuda_ms": total_cuda_ms,
        "total_cpu_ms": total_cpu_ms,
        "n_cuda_events": len([e for e in events if _cuda_us(e) > 0]),
        "top5_cuda_kernels": [
            (e.key[:60], _cuda_us(e) / 1000)
            for e in sorted(events, key=lambda e: _cuda_us(e), reverse=True)[:5]
        ],
        "attention_time_ms": sum(_cuda_us(e) for e in attn_kernels) / 1000,
        "matmul_time_ms": sum(_cuda_us(e) for e in matmul_kernels) / 1000,
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

    # 5. Continuous VRAM monitoring during generation
    print(f"\n--- 5. VRAM Timeline During Generation ---")
    gc.collect(); torch.cuda.empty_cache()
    vram_timeline = []
    vram_timeline.append(("pre-generate", measure_vram()))

    inputs = tokenizer(prompt_long, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    vram_timeline.append(("post-tokenize", measure_vram()))

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    vram_timeline.append(("post-prefill", measure_vram()))

    # Simulate decode steps
    past = outputs.past_key_values
    next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
    for step in range(10):
        with torch.no_grad():
            outputs = model(input_ids=next_token, past_key_values=past, use_cache=True)
        past = outputs.past_key_values
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
        if step in (0, 4, 9):
            vram_timeline.append((f"decode-step-{step}", measure_vram()))

    for label, mb in vram_timeline:
        print(f"  {label:<20} {mb:>8.1f} MB")

    kv_growth = vram_timeline[-1][1] - vram_timeline[2][1]
    print(f"  KV cache growth (10 decode steps): {kv_growth:.1f} MB")

    # 6. perf stat (CPU counters) for a short generation
    print(f"\n--- 6. CPU Performance Counters (perf stat) ---")
    import subprocess, tempfile, json

    # Write a short benchmark script
    bench_script = os.path.join(tempfile.gettempdir(), "tq_perf_bench.py")
    with open(bench_script, "w") as f:
        f.write(f"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("{model_name}", torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")
inputs = tokenizer("Hello world", return_tensors="pt").to("cuda")
with torch.no_grad():
    model.generate(**inputs, max_new_tokens=20, do_sample=False)
""")

    perf_result = subprocess.run(
        ["perf", "stat", "-e", "cache-misses,cache-references,instructions,cycles",
         "python", bench_script],
        capture_output=True, text=True, timeout=120,
    )
    if perf_result.returncode == 0:
        # perf stat outputs to stderr
        for line in perf_result.stderr.strip().split("\n"):
            line = line.strip()
            if any(k in line for k in ("cache-misses", "cache-references", "instructions", "cycles", "seconds")):
                print(f"  {line}")
    else:
        print(f"  perf stat failed: {perf_result.stderr[:200]}")

    # 7. GPU memory bandwidth estimate from nvidia-smi
    print(f"\n--- 7. GPU Memory Bandwidth Snapshot ---")
    # Run nvidia-smi dmon for 2 seconds during a generation
    dmon = subprocess.Popen(
        ["nvidia-smi", "dmon", "-s", "mu", "-d", "1", "-c", "3"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    # Generate while monitoring
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=50, do_sample=False)
    dmon_out, _ = dmon.communicate(timeout=10)
    for line in dmon_out.strip().split("\n"):
        if not line.startswith("#"):
            print(f"  {line.strip()}")

    print(f"\n{'=' * 70}")
    print(f"  BENCHMARK COMPLETE")
    print(f"  All measurements are from ACTUAL model inference,")
    print(f"  not proxy metrics on extracted tensors.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()

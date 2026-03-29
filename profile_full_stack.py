#!/usr/bin/env python3
"""
Full-stack profiling pipeline for TurboQuant inference.

Runs all 6 profiling phases with non-overlapping tools:
  Phase 1: Quick health check (nvidia-smi + perf stat)
  Phase 2: Python profiling (py-spy + memray)
  Phase 3: CPU microarch (AMD uProf CLI + perf record)
  Phase 4: GPU kernel analysis (nsys + ncu)
  Phase 5: System tracing (bpftrace + strace)
  Phase 6: Memory correctness (valgrind, optional)

Each phase produces a named output file in /tmp/tq_profile_*.
"""

import os
import subprocess
import sys
import time

ASKPASS = "/usr/bin/unified-askpass"
BENCH_SCRIPT = os.path.join(os.path.dirname(__file__), "bench_compressed_inference.py")
OUTPUT_DIR = "/tmp/tq_profiling"


def run(cmd, desc, timeout=120, needs_sudo=False):
    """Run a profiling command and report."""
    print(f"\n  [{desc}]", flush=True)
    env = os.environ.copy()
    if needs_sudo:
        env["SUDO_ASKPASS"] = ASKPASS
        cmd = ["sudo", "-A"] + cmd

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        if result.returncode == 0:
            print(f"    OK ({len(result.stdout)} bytes output)")
            return result.stdout
        else:
            print(f"    FAILED (exit {result.returncode}): {result.stderr[:200]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT ({timeout}s)")
        return None
    except FileNotFoundError:
        print(f"    NOT FOUND: {cmd[0]}")
        return None


def phase1_health_check():
    """Quick GPU + CPU health check."""
    print("\n=== PHASE 1: Quick Health Check ===")

    # nvidia-smi
    out = run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
         "--format=csv,noheader"],
        "nvidia-smi",
    )
    if out:
        print(f"    GPU: {out.strip()}")

    # perf stat on a short Python script
    out = run(
        ["perf", "stat", "-e", "cache-misses,cache-references,instructions,cycles,branch-misses",
         "python", "-c", "import torch; x=torch.randn(1000,128); y=x@x.T"],
        "perf stat (quick matmul)",
        timeout=30,
    )
    if out:
        for line in (out or "").split("\n"):
            if any(k in line for k in ("cache", "instructions", "cycles", "branch", "seconds")):
                print(f"    {line.strip()}")


def phase2_python_profiling():
    """Python-level profiling with py-spy and memray."""
    print("\n=== PHASE 2: Python Profiling ===")

    script = f"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-135M-Instruct', torch_dtype=torch.float16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M-Instruct')
inputs = tokenizer('Hello world', return_tensors='pt').to('cuda')
with torch.no_grad():
    model.generate(**inputs, max_new_tokens=20, do_sample=False)
"""
    script_file = os.path.join(OUTPUT_DIR, "bench_short.py")
    with open(script_file, "w") as f:
        f.write(script)

    # py-spy flame graph
    svg_path = os.path.join(OUTPUT_DIR, "pyspy_flamegraph.svg")
    run(
        ["py-spy", "record", "-o", svg_path, "--format", "flamegraph", "--", "python", script_file],
        f"py-spy -> {svg_path}",
        timeout=120, needs_sudo=True,
    )

    # memray
    memray_path = os.path.join(OUTPUT_DIR, "memray_output.bin")
    run(
        ["python", "-m", "memray", "run", "-o", memray_path, script_file],
        f"memray -> {memray_path}",
        timeout=120,
    )
    if os.path.exists(memray_path):
        stats_path = os.path.join(OUTPUT_DIR, "memray_stats.txt")
        run(
            ["python", "-m", "memray", "stats", memray_path],
            f"memray stats -> {stats_path}",
        )


def phase3_cpu_microarch():
    """CPU microarchitecture profiling."""
    print("\n=== PHASE 3: CPU Microarchitecture ===")

    script_file = os.path.join(OUTPUT_DIR, "bench_short.py")

    # AMD uProf CLI profiling
    uprof_dir = os.path.join(OUTPUT_DIR, "uprof_output")
    os.makedirs(uprof_dir, exist_ok=True)
    run(
        ["/opt/amduprof/bin/AMDuProfCLI", "collect", "--config", "assess_performance",
         "-o", uprof_dir, "python", script_file],
        f"AMD uProf CLI -> {uprof_dir}",
        timeout=180, needs_sudo=True,
    )

    # perf record for hotspot
    perf_data = os.path.join(OUTPUT_DIR, "perf.data")
    run(
        ["perf", "record", "-g", "-o", perf_data, "python", script_file],
        f"perf record -> {perf_data}",
        timeout=120,
    )


def phase4_gpu_kernels():
    """GPU kernel analysis."""
    print("\n=== PHASE 4: GPU Kernel Analysis ===")

    script_file = os.path.join(OUTPUT_DIR, "bench_short.py")

    # nsys
    nsys_out = os.path.join(OUTPUT_DIR, "nsys_profile")
    run(
        ["nsys", "profile", "--output", nsys_out, "--force-overwrite", "true", "python", script_file],
        f"nsys -> {nsys_out}.nsys-rep",
        timeout=180,
    )

    # ncu on a focused matmul
    ncu_script = os.path.join(OUTPUT_DIR, "ncu_bench.py")
    with open(ncu_script, "w") as f:
        f.write("import torch\nx=torch.randn(1000,128,device='cuda')\nPi=torch.randn(128,128,device='cuda')\nfor _ in range(20): y=x@Pi.T\ntorch.cuda.synchronize()\n")

    ncu_out = os.path.join(OUTPUT_DIR, "ncu_report.txt")
    out = run(
        ["ncu", "--set", "full", "--launch-skip", "5", "--launch-count", "3",
         "python", ncu_script],
        f"ncu -> {ncu_out}",
        timeout=120, needs_sudo=True,
    )
    if out:
        with open(ncu_out, "w") as f:
            f.write(out)


def phase5_system_tracing():
    """System-wide tracing."""
    print("\n=== PHASE 5: System Tracing ===")

    # strace summary
    script_file = os.path.join(OUTPUT_DIR, "bench_short.py")
    out = run(
        ["strace", "-c", "python", script_file],
        "strace -c (syscall summary)",
        timeout=120,
    )

    # bpftrace one-liner: count syscalls during 5 seconds
    run(
        ["bpftrace", "-e", "tracepoint:raw_syscalls:sys_enter { @[comm] = count(); } interval:s:3 { exit(); }"],
        "bpftrace (syscall count, 3s)",
        timeout=15, needs_sudo=True,
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"  FULL-STACK PROFILING: TurboQuant Inference")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'=' * 60}")

    phase1_health_check()
    phase2_python_profiling()
    phase3_cpu_microarch()
    phase4_gpu_kernels()
    phase5_system_tracing()

    # Summary of outputs
    print(f"\n{'=' * 60}")
    print(f"  PROFILING COMPLETE. Outputs:")
    print(f"{'=' * 60}")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        path = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(path) if os.path.isfile(path) else 0
        print(f"  {size:>10,} B  {f}")


if __name__ == "__main__":
    main()

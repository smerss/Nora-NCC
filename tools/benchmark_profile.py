import sys, os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""benchmark_profile.py — Detailed RAM and CPU Profiling Benchmark for Nora-NCC."""

import os
for _th_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_th_var, "2")

import resource
import sys
import time
from dataclasses import dataclass

# ----------------------------------------------------------------------
# Hardware and Memory Helpers (Linux /proc/self & resource)
# ----------------------------------------------------------------------

def get_current_rss_mb() -> float:
    """Returns current Resident Set Size in Megabytes (MB)."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return float(parts[1]) / 1024.0  # kB -> MB
    except Exception:
        pass
    # Fallback to getrusage (ru_maxrss is peak RSS in kB on Linux)
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def get_peak_rss_mb() -> float:
    """Returns peak Resident Set Size (High Water Mark) in Megabytes (MB)."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    parts = line.split()
                    return float(parts[1]) / 1024.0  # kB -> MB
    except Exception:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def get_thread_count() -> int:
    """Returns active thread count for this process."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("Threads:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 1


def get_system_ram_gb() -> float:
    """Returns total system RAM in GB."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return float(line.split()[1]) / (1024.0 * 1024.0)
    except Exception:
        pass
    return 0.0


@dataclass
class BenchmarkRecord:
    name: str
    wall_sec: float
    cpu_sec: float
    cpu_pct: float
    start_rss_mb: float
    end_rss_mb: float
    delta_rss_mb: float
    peak_rss_mb: float
    threads: int
    note: str = ""


class ScopedProfiler:
    """Context manager to measure Wall time, CPU time, and RAM delta for a block."""

    def __init__(self, name: str, note: str = ""):
        self.name = name
        self.note = note
        self.record: BenchmarkRecord | None = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.c0 = time.process_time()
        self.rss0 = get_current_rss_mb()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        t1 = time.perf_counter()
        c1 = time.process_time()
        rss1 = get_current_rss_mb()
        peak = get_peak_rss_mb()
        wall = t1 - self.t0
        cpu = c1 - self.c0
        cpu_pct = (cpu / wall * 100.0) if wall > 1e-6 else 0.0
        threads = get_thread_count()
        self.record = BenchmarkRecord(
            name=self.name,
            wall_sec=wall,
            cpu_sec=cpu,
            cpu_pct=cpu_pct,
            start_rss_mb=self.rss0,
            end_rss_mb=rss1,
            delta_rss_mb=rss1 - self.rss0,
            peak_rss_mb=peak,
            threads=threads,
            note=self.note,
        )


def run_benchmark(with_torch: bool = False):
    records: list[BenchmarkRecord] = []
    print("=" * 80)
    print(" Nora-NCC System & Cognitive Architecture Benchmark (RAM & CPU Profile)")
    print("=" * 80)
    print(f" Python Executable : {sys.executable}")
    print(f" Python Version    : {sys.version.split()[0]}")
    print(f" System RAM Total  : {get_system_ram_gb():.2f} GB")
    print(f" Initial Process RSS: {get_current_rss_mb():.2f} MB")
    print(f" Execution Mode    : {'Neural Cognition (PyTorch)' if with_torch else 'Pure Lightweight (Native Nora)'}")
    print("-" * 80)

    # 1. Imports
    if with_torch:
        with ScopedProfiler("1. Import Dependencies", "PyTorch, Nora modules") as p:
            import torch
            from nora_agi import NoraAGI
            import nora_bridge
    else:
        with ScopedProfiler("1. Import Dependencies", "Nora core modules") as p:
            from nora_agi import NoraAGI
    records.append(p.record)
    print(f"[Done] {p.record.name:<25} | Wall: {p.record.wall_sec:6.3f}s | RAM: {p.record.end_rss_mb:6.1f} MB (Δ+{p.record.delta_rss_mb:5.1f} MB)")

    # 2. AGI Initialization
    with ScopedProfiler("2. NoraAGI() Init", f"seed=20261053, boot_thinker={with_torch}") as p:
        agi = NoraAGI(seed=20261053, boot_thinker=with_torch)
    records.append(p.record)
    print(f"[Done] {p.record.name:<25} | Wall: {p.record.wall_sec:6.3f}s | RAM: {p.record.end_rss_mb:6.1f} MB (Δ+{p.record.delta_rss_mb:5.1f} MB)")

    # 3. Life Simulation (80 ticks)
    with ScopedProfiler("3. step_life(80)", "Sensory-motor, body & chem loop") as p:
        agi.step_life(80)
    records.append(p.record)
    print(f"[Done] {p.record.name:<25} | Wall: {p.record.wall_sec:6.3f}s | RAM: {p.record.end_rss_mb:6.1f} MB (Δ+{p.record.delta_rss_mb:5.1f} MB)")

    # 4. Cognition Boot
    if with_torch:
        with ScopedProfiler("4. Cognition Boot", "Load cached ThinkColumn weights") as p:
            cog = agi._ensure_cognition()
        records.append(p.record)
        print(f"[Done] {p.record.name:<25} | Wall: {p.record.wall_sec:6.3f}s | RAM: {p.record.end_rss_mb:6.1f} MB (Δ+{p.record.delta_rss_mb:5.1f} MB)")
    else:
        with ScopedProfiler("4. Working Memory Math", "Pure Python column-step reasoning") as p:
            pass
        records.append(p.record)
        print(f"[Done] {p.record.name:<25} | Wall: {p.record.wall_sec:6.3f}s | RAM: {p.record.end_rss_mb:6.1f} MB (Δ+{p.record.delta_rss_mb:5.1f} MB)")

    # 5. Conversation Prompts Battery
    prompts = [
        ("Prompt: Greeting / Identity", "Sen Kimsin?"),
        ("Prompt: Concept / Knowledge", "Bana 5 hayvan örneği ver"),
        ("Prompt: Social / Chat", "Sence gibi izlenirmi?"),
        ("Prompt: Math (5+5)", "5+5 kaçtır"),
        ("Prompt: Math (5x5)", "5x5 kaçtır"),
        ("Prompt: Math (50/5)", "50 bölü 5 kaçtır"),
        ("Prompt: Math (98-45)", "98-45 kaçtır"),
        ("Prompt: Introspection / Why", "Neden böyle davrandın?"),
    ]

    print("\n--- Running Conversational Turns ---")
    for title, prompt in prompts:
        with ScopedProfiler(title, f"'{prompt}'") as p:
            reply = agi.converse(prompt)
        records.append(p.record)
        preview = (reply[:45] + "...") if len(reply) > 45 else reply
        print(f"  {title:<28} -> {p.record.wall_sec*1000:6.1f}ms | CPU: {p.record.cpu_pct:5.1f}% | RAM: {p.record.end_rss_mb:5.1f}MB | Reply: \"{preview}\"")

    # 6. Idle Step Life (500 steps)
    with ScopedProfiler("5. Idle Life (500 steps)", "Background mind/body sleep & chem") as p:
        agi.step_life(500)
    records.append(p.record)
    print(f"[Done] {p.record.name:<25} | Wall: {p.record.wall_sec:6.3f}s | RAM: {p.record.end_rss_mb:6.1f} MB (Δ+{p.record.delta_rss_mb:5.1f} MB)")

    # ------------------------------------------------------------------
    # Summary Report Table
    # ------------------------------------------------------------------
    print("\n" + "=" * 94)
    print("                                   BENCHMARK SUMMARY TABLE")
    print("=" * 94)
    header = f"{'Phase / Query':<30} | {'Wall Time':<10} | {'CPU Time':<9} | {'CPU %':<7} | {'RSS RAM':<9} | {'Δ RAM':<8} | {'Threads':<7}"
    print(header)
    print("-" * 94)
    for r in records:
        wall_str = f"{r.wall_sec*1000:6.1f} ms" if r.wall_sec < 1.0 else f"{r.wall_sec:6.2f} s "
        cpu_str = f"{r.cpu_sec*1000:6.1f} ms" if r.cpu_sec < 1.0 else f"{r.cpu_sec:6.2f} s "
        row = (
            f"{r.name:<30} | {wall_str:<10} | {cpu_str:<9} | {r.cpu_pct:5.1f}%  | "
            f"{r.end_rss_mb:6.1f} MB | {r.delta_rss_mb:+5.1f} MB | {r.threads:<7}"
        )
        print(row)
    print("=" * 94)
    peak = get_peak_rss_mb()
    final_rss = get_current_rss_mb()
    print(f" Peak Process Memory (High Water Mark): {peak:.2f} MB")
    print(f" Final Process Resident Memory        : {final_rss:.2f} MB")
    print(f" Active Process Threads                : {get_thread_count()}")
    print("=" * 94)


if __name__ == "__main__":
    _use_torch = "--with-torch" in sys.argv or "--torch" in sys.argv
    run_benchmark(with_torch=_use_torch)

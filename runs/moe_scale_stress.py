#!/usr/bin/env python3
"""64 gerçek diskteki Broca uzmanıyla 0,45B ölçek/RSS stresi; kalite iddiası yok."""
import gc
import json
import os
import resource
import time

import torch

from nora_broca_net import BrocaOrgan
from nora_besle import DIALOGUE, SENTENCES
from nora_agi import NoraAGI
from nora_voice import state_vector

ROOT = ".cache/moe_scale"
COUNT = 64


def rss_bytes():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


def main():
    os.makedirs(ROOT, exist_ok=True)
    torch.set_num_threads(1)
    base = BrocaOrgan.load("data/broca_tr.pt")
    base.expand_hidden(768, seed=140)
    per_expert = sum(p.numel() for p in base.net.parameters())
    del base
    gc.collect()
    started = time.time()
    for index in range(COUNT):
        path = f"{ROOT}/expert_{index:03d}.pt"
        if os.path.exists(path):
            continue
        organ = BrocaOrgan.load("data/broca_tr.pt")
        organ.expand_hidden(768, seed=1000 + index)
        torch.save(organ.net.state_dict(), path + ".tmp")
        os.replace(path + ".tmp", path)
        del organ
        gc.collect()
        if index % 8 == 7:
            print("materialized", index + 1, "seconds", time.time() - started, flush=True)
    active = BrocaOrgan.load("data/broca_tr.pt")
    active.expand_hidden(768, seed=140)
    active.net.eval()
    state = state_vector(NoraAGI(seed=140, boot_thinker=False))
    probe = (SENTENCES + DIALOGUE)[::5][:8]
    rows = []
    for index in range(COUNT):
        before = rss_bytes()
        t0 = time.perf_counter()
        weights = torch.load(f"{ROOT}/expert_{index:03d}.pt", weights_only=True)
        active.net.load_state_dict(weights)
        del weights
        load_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        metrics = active.evaluate(probe, state, use_anchor=False, max_sent=8)
        eval_seconds = time.perf_counter() - t0
        gc.collect()
        after = rss_bytes()
        row = {"expert": index, "load_seconds": load_seconds,
               "eval_seconds": eval_seconds, "rss_before": before, "rss_after": after,
               "top1": metrics["top1"], "n": metrics["n"]}
        rows.append(row)
        if index % 8 == 7:
            print("routed", index + 1, "rss_mb", after / 1e6,
                  "load_ms", 1000 * load_seconds, flush=True)
    disk_bytes = sum(os.path.getsize(f"{ROOT}/expert_{i:03d}.pt") for i in range(COUNT))
    result = {
        "status": "scale_stress_not_quality_claim",
        "experts": COUNT,
        "parameters_per_expert": per_expert,
        "total_logical_parameters": COUNT * per_expert,
        "active_parameters": per_expert,
        "active_fraction": per_expert / (COUNT * per_expert),
        "disk_bytes": disk_bytes,
        "seconds_total": time.time() - started,
        "rss_min": min(x["rss_after"] for x in rows),
        "rss_max": max(x["rss_after"] for x in rows),
        "rss_last8_mean": sum(x["rss_after"] for x in rows[-8:]) / 8,
        "load_seconds_mean": sum(x["load_seconds"] for x in rows) / COUNT,
        "eval_seconds_mean": sum(x["eval_seconds"] for x in rows) / COUNT,
        "max_rss_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "rows": rows,
    }
    with open("runs/moe_scale_stress.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()

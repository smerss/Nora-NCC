#!/usr/bin/env python3
"""Haricî wiki+film akışında tek-geçiş Broca uzmanı; her 250 adımda kalıcı checkpoint."""
import json
import math
import os
import time
from collections import deque

import numpy as np
import torch

from nora_agi import NoraAGI
from nora_besle import DIALOGUE, SENTENCES
from nora_broca_net import BrocaOrgan
from nora_voice import state_vector


def main():
    torch.set_num_threads(1)
    steps = int(os.environ.get("NORA_EXTERNAL_STEPS", "3500"))
    dropout = float(os.environ.get("NORA_EXTERNAL_DROPOUT", "0"))
    bpe_dropout = float(os.environ.get("NORA_EXTERNAL_BPE_DROPOUT", "0"))
    tag = os.environ.get("NORA_EXTERNAL_TAG", f"d{dropout:g}_b{bpe_dropout:g}")
    rows = [json.loads(x)["text"] for x in open(".cache/corpora/processed/external_train.jsonl")]
    held = (SENTENCES + DIALOGUE)[::5]
    forbidden = set(held)
    state = state_vector(NoraAGI(seed=140, boot_thinker=False))
    organ = BrocaOrgan.load("data/broca_tr.pt")
    organ.net.drop.p = dropout
    organ.opt = torch.optim.Adam(organ.net.parameters(), lr=organ.lr)
    torch.manual_seed(140)
    organ.rng = np.random.default_rng(90140)
    start_seen = organ.seen_tokens
    losses = deque(maxlen=250)
    history = []
    skipped = 0
    cursor = 0
    started = time.monotonic()
    best = None
    os.makedirs("runs/models", exist_ok=True)
    for step in range(1, steps + 1):
        batch = []
        while len(batch) < 32:
            if cursor >= len(rows):
                raise RuntimeError("streaming corpus ilk geçişte tükendi")
            text = rows[cursor]
            cursor += 1
            if text in forbidden:
                skipped += 1
                continue
            batch.append(text)
        lr = 0.001 / math.sqrt(1.0 + step / 2400.0)
        losses.append(organ.learn_batch(batch, state, lr=lr, anchor_dropout=1.0,
                                       loss_mode="full", bpe_dropout=bpe_dropout))
        if step % 250 == 0 or step == steps:
            metrics = organ.evaluate(held, state, use_anchor=False, max_sent=10000)
            row = {"step": step, "seen_delta": organ.seen_tokens - start_seen,
                   "loss250": float(np.mean(losses)), "lr": lr,
                   "seconds": time.monotonic() - started, "held_exact_skipped": skipped,
                   "dropout": dropout, "bpe_dropout": bpe_dropout, "batch": 32,
                   **metrics}
            history.append(row)
            print(row, flush=True)
            path = f"runs/models/external_{tag}_step{step}.pt"
            organ.save(path)
            key = (metrics["top1"], -metrics["ppl"])
            if best is None or key > best:
                best = key
                organ.save(f"runs/models/external_{tag}_best.pt")
            with open(f"runs/turkish_external_rebuild_{tag}.json", "w") as handle:
                json.dump(history, handle, indent=2)


if __name__ == "__main__":
    main()

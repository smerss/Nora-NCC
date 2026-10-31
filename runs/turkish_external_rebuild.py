#!/usr/bin/env python3
"""Haricî wiki+film akışında tek-geçiş Broca uzmanı; her 250 adımda kalıcı checkpoint."""
import json
import math
import os
import re
import time
from collections import deque

import numpy as np
import torch
import tokenizers

from nora_agi import NoraAGI
from nora_besle import DIALOGUE, SENTENCES
from nora_broca_net import BrocaOrgan
from nora_voice import state_vector


def main():
    torch.set_num_threads(1)
    steps = int(os.environ.get("NORA_EXTERNAL_STEPS", "3500"))
    dropout = float(os.environ.get("NORA_EXTERNAL_DROPOUT", "0"))
    bpe_dropout = float(os.environ.get("NORA_EXTERNAL_BPE_DROPOUT", "0"))
    deterministic_bpe = bool(int(os.environ.get("NORA_EXTERNAL_DETERMINISTIC_BPE", "0")))
    tag = os.environ.get("NORA_EXTERNAL_TAG", f"d{dropout:g}_b{bpe_dropout:g}")
    sources = {x for x in os.environ.get("NORA_EXTERNAL_SOURCES", "").split(",") if x}
    min_chars = int(os.environ.get("NORA_EXTERNAL_MIN_CHARS", "0"))
    max_chars = int(os.environ.get("NORA_EXTERNAL_MAX_CHARS", "0"))
    if min_chars < 0 or max_chars < 0 or (max_chars and max_chars < min_chars):
        raise ValueError("haricî karakter aralığı geçersiz")
    rows = [row["text"] for row in map(json.loads, open(".cache/corpora/processed/external_train.jsonl"))
            if (not sources or row["source"] in sources)
            and len(row["text"]) >= min_chars
            and (not max_chars or len(row["text"]) <= max_chars)]
    if not rows:
        raise ValueError("haricî filtre sonrası eğitim cümlesi kalmadı")
    held = (SENTENCES + DIALOGUE)[::5]
    normalize = lambda text: re.sub(r"\s+", " ", text).strip().casefold()
    forbidden = {normalize(text) for text in SENTENCES + DIALOGUE}
    state = state_vector(NoraAGI(seed=140, boot_thinker=False))
    organ = BrocaOrgan.load("data/broca_tr.pt")
    expanded_hidden = int(os.environ.get("NORA_EXTERNAL_HIDDEN", "0"))
    if expanded_hidden:
        organ.expand_hidden(expanded_hidden, seed=140)
    positional = bool(int(os.environ.get("NORA_EXTERNAL_POSITIONAL", "0")))
    if positional and organ.net.pos_emb is None:
        organ.net.pos_emb = torch.nn.Embedding(65, organ.net.emb.embedding_dim)
        torch.nn.init.zeros_(organ.net.pos_emb.weight)
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
            if normalize(text) in forbidden:
                skipped += 1
                continue
            batch.append(text)
        lr = 0.001 / math.sqrt(1.0 + step / 2400.0)
        losses.append(organ.learn_batch(batch, state, lr=lr, anchor_dropout=1.0,
                                       loss_mode="full", bpe_dropout=bpe_dropout,
                                       bpe_dropout_seed=140_000_000 + step if deterministic_bpe else None))
        if step % 250 == 0 or step == steps:
            metrics = organ.evaluate(held, state, use_anchor=False, max_sent=10000)
            row = {"step": step, "seen_delta": organ.seen_tokens - start_seen,
                   "loss250": float(np.mean(losses)), "lr": lr,
                   "seconds": time.monotonic() - started, "held_exact_skipped": skipped,
                   "dropout": dropout, "bpe_dropout": bpe_dropout,
                   "bpe_dropout_deterministic": (not bool(bpe_dropout)) or deterministic_bpe,
                   "seeded_bpe": deterministic_bpe,
                   "runtime": {"torch": torch.__version__, "tokenizers": tokenizers.__version__},
                   "batch": 32, "sources": sorted(sources), "stream_sentences": len(rows),
                   "min_chars": min_chars, "max_chars": max_chars,
                   "hidden": organ.net.h, "positional": organ.net.pos_emb is not None, **metrics}
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

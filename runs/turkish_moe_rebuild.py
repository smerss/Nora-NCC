#!/usr/bin/env python3
"""Oturumdan bağımsız, deterministik Broca uzmanı yeniden kurulumu.

Yalnız SENTENCES+DIALOGUE üzerindeki aynı every-5 development split kullanılır.
Haricî corpus bu betiğin hedef/geniş kollarına girmez. Checkpoint seçiminin development
üzerinden yapıldığı genelleme kanıtı değildir; bağımsız onam ayrıca ölçülür.
"""
import json
import os
import random

import numpy as np
import torch
from torch import nn

from nora_agi import NoraAGI
from nora_besle import DIALOGUE, SENTENCES
from nora_broca_net import BrocaOrgan
from nora_voice import state_vector


def main():
    torch.set_num_threads(1)
    kind = os.environ.get("NORA_REBUILD_KIND", "position")
    if kind not in {"position", "wide"}:
        raise ValueError("NORA_REBUILD_KIND position veya wide olmalı")
    seeds = tuple(int(x) for x in os.environ.get("NORA_REBUILD_SEEDS", "141,142,143").split(","))
    rows = SENTENCES + DIALOGUE
    held = rows[::5]
    base_train = [x for i, x in enumerate(rows) if i % 5]
    state = state_vector(NoraAGI(seed=140, boot_thinker=False))
    os.makedirs("runs/models", exist_ok=True)
    results = []
    for seed in seeds:
        train = list(base_train)
        random.Random(seed).shuffle(train)
        organ = BrocaOrgan.load("data/broca_tr.pt")
        if kind == "wide":
            organ.expand_hidden(1024, seed=seed)
            epochs = 11
        else:
            organ.net.pos_emb = nn.Embedding(65, organ.net.emb.embedding_dim)
            nn.init.zeros_(organ.net.pos_emb.weight)
            epochs = 21
        organ.net.drop.p = 0.1
        organ.opt = torch.optim.Adam(organ.net.parameters(), lr=organ.lr)
        torch.manual_seed(seed)
        organ.rng = np.random.default_rng(90000 + seed)
        best = None
        for epoch in range(epochs):
            metrics = organ.evaluate(held, state, use_anchor=False, max_sent=10000)
            row = {"kind": kind, "seed": seed, "epoch": epoch, "hidden": organ.net.h,
                   "positional": organ.net.pos_emb is not None, "dropout": 0.1,
                   "batch": 32, "lr": 5e-4, "bpe_dropout": 0.1,
                   "loss": "full_softmax", **metrics}
            results.append(row)
            print(row, flush=True)
            key = (metrics["top1"], -metrics["ppl"])
            if best is None or key > best[0]:
                best = (key, epoch)
                organ.save(f"runs/models/rebuilt_{kind}{seed}_best.pt")
            if epoch + 1 < epochs:
                for i in range(0, len(train), 32):
                    organ.learn_batch(train[i:i + 32], state, lr=5e-4,
                                      anchor_dropout=1.0, loss_mode="full", bpe_dropout=0.1)
        print("BEST", kind, seed, best, flush=True)
    output = f"runs/turkish_rebuild_{kind}_seeds.json"
    with open(output, "w") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()

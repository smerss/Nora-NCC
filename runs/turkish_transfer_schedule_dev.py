#!/usr/bin/env python3
"""Temiz haricî checkpoint -> hedef position ince ayarı; yalnız development fold 1/2.

Her epoch seçimi yalnız bu development çıktısı içindir. Confirmation fold'larında
yeniden epoch seçmek yerine burada dondurulan yapılandırma kullanılmalıdır.
"""
import json
import os
import random
import time

import numpy as np
import torch
from torch import nn

from nora_agi import NoraAGI
from nora_besle import DIALOGUE, SENTENCES
from nora_broca_net import BrocaOrgan
from nora_voice import state_vector


def main():
    torch.set_num_threads(1)
    rows = SENTENCES + DIALOGUE
    state = state_vector(NoraAGI(seed=140, boot_thinker=False))
    seed = int(os.environ.get("NORA_POSITION_SEED", "149"))
    folds = tuple(int(x) for x in os.environ.get("NORA_CROSSFOLDS", "1,2").split(","))
    steps = tuple(int(x) for x in os.environ.get("NORA_EXTERNAL_BASE_STEPS", "250,500,750,1000").split(","))
    learning_rates = tuple(float(x) for x in os.environ.get("NORA_TRANSFER_LRS", "0.0003,0.001").split(","))
    checkpoints = tuple(int(x) for x in os.environ.get("NORA_TRANSFER_EPOCHS", "6,12,24").split(","))
    output = os.environ.get("NORA_TRANSFER_OUTPUT", "runs/turkish_transfer_schedule_dev.json")
    results = []
    os.makedirs("runs/models", exist_ok=True)
    for fold in folds:
        target = [x for i, x in enumerate(rows) if i % 5 != fold]
        held = [x for i, x in enumerate(rows) if i % 5 == fold]
        for step in steps:
            base_path = f"runs/models/external_clean1000_step{step}.pt"
            for lr in learning_rates:
                organ = BrocaOrgan.load(base_path)
                organ.net.pos_emb = nn.Embedding(65, organ.net.emb.embedding_dim)
                nn.init.zeros_(organ.net.pos_emb.weight)
                organ.opt = torch.optim.Adam(organ.net.parameters(), lr=organ.lr)
                torch.manual_seed(seed)
                started = time.monotonic()
                best = None
                history = []
                for epoch in range(1, max(checkpoints) + 1):
                    shuffled = list(target)
                    random.Random(seed + epoch).shuffle(shuffled)
                    for i in range(0, len(shuffled), 32):
                        organ.learn_batch(shuffled[i:i + 32], state, lr=lr,
                                          anchor_dropout=1.0, loss_mode="full", bpe_dropout=0.1)
                    if epoch not in checkpoints:
                        continue
                    metrics = organ.evaluate(held, state, use_anchor=False, max_sent=10000)
                    row = {"fold": fold, "external_step": step, "lr": lr, "epoch": epoch,
                           "seed": seed, "seconds": time.monotonic() - started, **metrics}
                    history.append(row)
                    results.append(row)
                    print(row, flush=True)
                    key = (metrics["top1"], -metrics["ppl"])
                    if best is None or key > best[0]:
                        best = (key, epoch)
                        tag = f"step{step}_lr{lr:g}_fold{fold}"
                        organ.save(f"runs/models/transfer_{tag}_devbest.pt")
                print("BEST", fold, step, lr, best, flush=True)
    with open(output, "w") as handle:
        json.dump({"results": results, "development_folds": folds,
                   "warning": "epoch ve hiperparametre seçimi development içindir; confirmation'da yeniden seçilmez"},
                  handle, indent=2)


if __name__ == "__main__":
    main()

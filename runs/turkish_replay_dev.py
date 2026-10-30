#!/usr/bin/env python3
"""Haricî uzmandan hedef + genel-Türkçe replay; ağırlık soup'u veya L2-SP değil."""
import json
import os
import random
import time

import torch
from torch import nn

from nora_agi import NoraAGI
from nora_besle import DIALOGUE, SENTENCES
from nora_broca_net import BrocaOrgan
from nora_voice import state_vector


def main():
    torch.set_num_threads(1)
    rows = SENTENCES + DIALOGUE
    held = rows[::5]
    held_set = set(held)
    target = [x for i, x in enumerate(rows) if i % 5]
    external = []
    with open(".cache/corpora/processed/turkish_mix_train_shuffled.jsonl") as handle:
        for line in handle:
            text = json.loads(line)["text"]
            if text not in held_set:
                external.append(text)
            if len(external) >= 5000:
                break
    state = state_vector(NoraAGI(seed=140, boot_thinker=False))
    results = []
    ratios = tuple(int(x) for x in os.environ.get("NORA_REPLAY_RATIOS", "1,3").split(","))
    positional = bool(int(os.environ.get("NORA_REPLAY_POSITION", "0")))
    epochs = int(os.environ.get("NORA_REPLAY_EPOCHS", "13"))
    output = os.environ.get("NORA_REPLAY_OUTPUT", "runs/turkish_replay_dev.json")
    tag = "position_" if positional else ""
    for target_repeats in ratios:
        organ = BrocaOrgan.load("runs/models/external_rebuilt_best.pt")
        if positional and organ.net.pos_emb is None:
            organ.net.pos_emb = nn.Embedding(65, organ.net.emb.embedding_dim)
            nn.init.zeros_(organ.net.pos_emb.weight)
            organ.opt = torch.optim.Adam(organ.net.parameters(), lr=organ.lr)
        best = None
        started = time.time()
        for epoch in range(epochs):
            metrics = organ.evaluate(held, state, use_anchor=False, max_sent=10000)
            row = {"target_repeats": target_repeats, "positional": positional, "epoch": epoch,
                   "seconds": time.time() - started, **metrics}
            results.append(row)
            print(row, flush=True)
            key = (metrics["top1"], -metrics["ppl"])
            if best is None or key > best[0]:
                best = (key, epoch)
                organ.save(f"/tmp/turkish_replay_{tag}{target_repeats}_best.pt")
            if epoch == epochs - 1:
                break
            rng = random.Random(140 + 100 * target_repeats + epoch)
            replay = rng.sample(external, len(target))
            batch_rows = target * target_repeats + replay
            rng.shuffle(batch_rows)
            for i in range(0, len(batch_rows), 32):
                organ.learn_batch(batch_rows[i:i + 32], state, lr=1e-4,
                                  anchor_dropout=1.0, loss_mode="full", bpe_dropout=0.1)
        print("BEST", target_repeats, best, flush=True)
    with open(output, "w") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()

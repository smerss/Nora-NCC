#!/usr/bin/env python3
"""178'lik tarifi yeni fold'larda epoch/model seçmeden doğrula.

Orijinal development (fold 0) yalnız mimari, tohum, ağırlık ve epoch sayısını sabitlemek
için kullanılır. Fold 1/2 üzerinde en iyi checkpoint aranmaz.
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
from runs.turkish_confirmation_eval import probability_evaluate


def train(kind, seed, epochs, train, state):
    organ = BrocaOrgan.load("data/broca_tr.pt")
    if kind == "wide":
        organ.expand_hidden(1024, seed=seed)
    else:
        organ.net.pos_emb = nn.Embedding(65, organ.net.emb.embedding_dim)
        nn.init.zeros_(organ.net.pos_emb.weight)
    organ.net.drop.p = 0.1
    organ.opt = torch.optim.Adam(organ.net.parameters(), lr=organ.lr)
    torch.manual_seed(seed)
    organ.rng = np.random.default_rng(90000 + seed)
    for _ in range(epochs):
        for i in range(0, len(train), 32):
            organ.learn_batch(train[i:i + 32], state, lr=5e-4, anchor_dropout=1.0,
                              loss_mode="full", bpe_dropout=0.1)
    organ.net.eval()
    return organ


def main():
    torch.set_num_threads(1)
    rows = SENTENCES + DIALOGUE
    state = state_vector(NoraAGI(seed=140, boot_thinker=False))
    position_seed = int(os.environ.get("NORA_POSITION_SEED", "144"))
    wide_seed = int(os.environ.get("NORA_WIDE_SEED", "149"))
    position_epochs = int(os.environ.get("NORA_POSITION_EPOCHS", "8"))
    wide_epochs = int(os.environ.get("NORA_WIDE_EPOCHS", "7"))
    weight = float(os.environ.get("NORA_POSITION_WEIGHT", "0.65"))
    output = os.environ.get("NORA_CROSSFOLD_OUTPUT", "runs/turkish_ensemble178_crossfold.json")
    results = []
    for fold in [int(x) for x in os.environ.get("NORA_CROSSFOLDS", "1,2").split(",")]:
        if fold == 0 or not 0 < fold < 5:
            raise ValueError("confirmation fold 1..4 olmalı; fold 0 model seçimidir")
        held = rows[fold::5]
        train_rows = [x for i, x in enumerate(rows) if i % 5 != fold]
        started = time.monotonic()
        fold_result = {"fold": fold, "position_seed": position_seed, "wide_seed": wide_seed,
                       "position_epochs_fixed": position_epochs, "wide_epochs_fixed": wide_epochs,
                       "position_weight": weight}
        organs = []
        for kind, seed, epochs in [("position", position_seed, position_epochs),
                                   ("wide", wide_seed, wide_epochs)]:
            shuffled = list(train_rows)
            random.Random(seed).shuffle(shuffled)
            organ = train(kind, seed, epochs, shuffled, state)
            organ.save(f"runs/models/crossfold_fold{fold}_{kind}.pt")
            fold_result[kind] = probability_evaluate([organ], [1.0], held, state)
            organs.append(organ)
        fold_result["ensemble"] = probability_evaluate(organs, [weight, 1.0 - weight], held, state)
        fold_result["seconds"] = time.monotonic() - started
        results.append(fold_result)
        print(json.dumps(fold_result, indent=2), flush=True)
        with open(output, "w") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()

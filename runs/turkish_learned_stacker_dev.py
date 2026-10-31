#!/usr/bin/env python3
"""Fold-0'da olasılık-stack ağırlığı öğren; yalnız development fold 1/2'de ölç.

Stacker cümle/cevap seçmez. Beş eğitilmiş next-subword dağılımının aritmetik
karışım ağırlığını öğrenir. Fold 3/4 bu betikte açılmaz.
"""
import hashlib
import json
import os
import time

import numpy as np
import torch
from torch import nn

from nora_agi import NoraAGI
from nora_besle import DIALOGUE, SENTENCES
from nora_broca_net import BrocaOrgan
from nora_voice import state_vector

NAMES = ("original_position", "original_wide", "clean_position", "clean_wide", "clean_external")
BASE_WEIGHTS = torch.tensor([0.225, 0.225, 0.225, 0.225, 0.1], dtype=torch.float32)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def model_paths(fold):
    return [
        f"runs/models/pair2_fold{fold}_position.pt",
        f"runs/models/pair2_fold{fold}_wide.pt",
        f"runs/models/clean_init_fold{fold}_position.pt",
        f"runs/models/clean_init_fold{fold}_wide.pt",
        "runs/models/external_clean1000_step1000.pt",
    ]


@torch.no_grad()
def distributions(fold, rows, state):
    organs = [BrocaOrgan.load(path) for path in model_paths(fold)]
    for organ in organs:
        organ.net.eval()
    probabilities, targets, positions = [], [], []
    for text in rows[fold::5]:
        ids = organs[0].tok.tk.encode(text).ids[:organs[0].MAX_LEN]
        if len(ids) < 4:
            continue
        inp = torch.tensor([[organs[0].net.bos] + ids])
        target = torch.tensor(ids + [organs[0].net.eos])
        bag = torch.zeros(1, organs[0].net.V)
        st = torch.from_numpy(state.astype(np.float32)[None])
        current = []
        for organ in organs:
            logits, _ = organ.net(inp, organ.net.encode_cond(st, bag))
            current.append(torch.softmax(logits[0].float(), dim=-1))
        probabilities.append(torch.stack(current, dim=1))  # L,E,V
        targets.append(target)
        positions.append(torch.arange(len(target)))
    return torch.cat(probabilities), torch.cat(targets), torch.cat(positions)


def bucket_indices(positions, scheme):
    if scheme == "global":
        return torch.zeros_like(positions), 1
    if scheme == "three":
        return torch.where(positions < 3, 0, torch.where(positions < 6, 1, 2)), 3
    if scheme == "seven":
        return positions.clamp(max=6), 7
    raise ValueError(scheme)


def train_weights(probabilities, targets, positions, scheme, regularization, steps=300):
    buckets, count = bucket_indices(positions, scheme)
    target_probability = probabilities.gather(
        2, targets[:, None, None].expand(-1, len(NAMES), -1)).squeeze(-1)
    initial = BASE_WEIGHTS.repeat(count, 1)
    raw = nn.Parameter(initial.log())
    optimizer = torch.optim.Adam([raw], lr=0.05)
    for _ in range(steps):
        weights = raw.softmax(dim=-1)
        mixed = (target_probability * weights[buckets]).sum(dim=-1).clamp_min(1e-30)
        nll = -mixed.log().mean()
        penalty = regularization * (weights - initial).square().sum()
        loss = nll + penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return raw.detach().softmax(dim=-1)


@torch.no_grad()
def evaluate(probabilities, targets, positions, weights, scheme):
    buckets, count = bucket_indices(positions, scheme)
    if weights.shape[0] == 1:
        weights = weights.expand(count, -1)
    hit1 = hit5 = 0
    nll = 0.0
    for start in range(0, len(targets), 32):
        end = min(start + 32, len(targets))
        current = torch.empty(end - start, probabilities.shape[-1])
        for bucket in range(weights.shape[0]):
            mask = buckets[start:end] == bucket
            if mask.any():
                current[mask] = (probabilities[start:end][mask] * weights[bucket][None, :, None]).sum(dim=1)
        current.clamp_min_(1e-30)
        target = targets[start:end]
        top = current.topk(5, dim=-1).indices
        hit1 += int((top[:, 0] == target).sum())
        hit5 += int((top == target[:, None]).any(-1).sum())
        nll -= float(current.log().gather(1, target[:, None]).sum())
    n = len(targets)
    return {"n": n, "hit1": hit1, "hit5": hit5, "top1": hit1 / n,
            "top5": hit5 / n, "ppl": float(np.exp(nll / n))}


def main():
    torch.set_num_threads(1)
    rows = SENTENCES + DIALOGUE
    state = state_vector(NoraAGI(seed=140, boot_thinker=False))
    started = time.monotonic()
    train_data = distributions(0, rows, state)
    configurations = [(scheme, reg) for scheme in ("global", "three", "seven")
                      for reg in (0.001, 0.01, 0.1)]
    learned = {}
    results = []
    baseline = BASE_WEIGHTS[None, :]
    for scheme, regularization in configurations:
        key = f"{scheme}_reg{regularization:g}"
        learned[key] = train_weights(*train_data, scheme, regularization)
        print("LEARNED", key, learned[key].tolist(), flush=True)
    for fold in (0, 1, 2):
        data = train_data if fold == 0 else distributions(fold, rows, state)
        for scheme in ("global", "three", "seven"):
            metrics = evaluate(*data, baseline, scheme)
            row = {"fold": fold, "variant": f"baseline_{scheme}", "metrics": metrics}
            results.append(row)
            print(row, flush=True)
        for (scheme, regularization), weights in zip(configurations, learned.values()):
            key = f"{scheme}_reg{regularization:g}"
            metrics = evaluate(*data, weights, scheme)
            row = {"fold": fold, "variant": key, "metrics": metrics}
            results.append(row)
            print(row, flush=True)
        if fold != 0:
            del data
    payload = {
        "protocol": "fit fold0 only; evaluate development fold1/2; fold3/4 untouched",
        "experts": list(NAMES),
        "base_weights": BASE_WEIGHTS.tolist(),
        "model_paths": {str(fold): model_paths(fold) for fold in (0, 1, 2)},
        "model_sha256": {path: sha256(path) for fold in (0, 1, 2) for path in model_paths(fold)},
        "weights": {key: value.tolist() for key, value in learned.items()},
        "results": results,
        "seconds": time.monotonic() - started,
    }
    output = os.environ.get("NORA_STACKER_OUTPUT", "runs/turkish_learned_stacker_dev.json")
    with open(output, "w") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()

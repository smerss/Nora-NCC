#!/usr/bin/env python3
"""Fold0 correctness eğitimi; fold1/2 development seçimi. Causal features only."""
import json
import os
import time

import numpy as np
import torch
from torch import nn

from nora_agi import NoraAGI
from nora_besle import DIALOGUE, SENTENCES
from nora_broca_net import BrocaOrgan  # noqa: F401 - checkpoint sınıfı için
from nora_voice import state_vector
from runs.turkish_dynamic_router_dev import (
    ROLES, buckets_of, distributions, model_paths, sha256,
)


def train_router(data, scheme, regularization, steps=600):
    probabilities, targets, positions, features = data
    buckets, count = buckets_of(positions, scheme)
    experts = probabilities.shape[1]
    labels = (probabilities.argmax(-1) == targets[:, None]).float()
    feature_weight = nn.Parameter(torch.zeros(features.shape[-1]))
    expert_bias = nn.Parameter(torch.zeros(experts))
    position_bias = nn.Parameter(torch.zeros(count, experts))
    optimizer = torch.optim.Adam([feature_weight, expert_bias, position_bias], lr=0.03)
    for _ in range(steps):
        score = (features * feature_weight).sum(-1) + expert_bias + position_bias[buckets]
        loss = nn.functional.binary_cross_entropy_with_logits(score, labels)
        penalty = regularization * (feature_weight.square().sum() + expert_bias.square().sum()
                                    + position_bias.square().sum())
        (loss + penalty).backward()
        optimizer.step()
        optimizer.zero_grad()
    return feature_weight.detach(), expert_bias.detach(), position_bias.detach()


@torch.no_grad()
def evaluate(data, learned, scheme, mode):
    probabilities, targets, positions, features = data
    buckets, _ = buckets_of(positions, scheme)
    feature_weight, expert_bias, position_bias = learned
    score = (features * feature_weight).sum(-1) + expert_bias + position_bias[buckets]
    if mode == "hard":
        index = score.argmax(-1)
        weights = nn.functional.one_hot(index, probabilities.shape[1]).float()
    elif mode.startswith("top"):
        count = int(mode[3:])
        index = score.topk(count, dim=-1).indices
        weights = torch.zeros_like(score).scatter_(-1, index, 1.0 / count)
    elif mode.startswith("softmax"):
        temperature = float(mode.split(":")[1])
        weights = (score / temperature).softmax(-1)
    elif mode == "sigmoid":
        weights = score.sigmoid()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    else:
        raise ValueError(mode)
    hit1 = hit5 = 0
    nll = 0.0
    for start in range(0, len(targets), 16):
        end = min(start + 16, len(targets))
        current = (probabilities[start:end] * weights[start:end, :, None]).sum(1).clamp_min(1e-30)
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
    train_folds = tuple(int(x) for x in os.environ.get("NORA_ROUTER_TRAIN_FOLDS", "0").split(","))
    eval_folds = tuple(int(x) for x in os.environ.get("NORA_ROUTER_EVAL_FOLDS", "1,2").split(","))
    if set(train_folds) & set(eval_folds):
        raise ValueError("router train/eval fold çakışması")
    started = time.monotonic()
    raw = {fold: distributions(fold, rows, state) for fold in train_folds}
    flat = torch.cat([data[3].reshape(-1, data[3].shape[-1]) for data in raw.values()])
    mean, std = flat.mean(0), flat.std(0).clamp_min(1e-6)
    normalized = {fold: (*data[:3], (data[3] - mean) / std) for fold, data in raw.items()}
    train_data = tuple(torch.cat([normalized[fold][i] for fold in train_folds]) for i in range(4))
    del raw, normalized
    configurations = [(scheme, reg) for scheme in ("global", "three")
                      for reg in (0.001, 0.01, 0.1, 1.0)]
    learned = {f"{scheme}_reg{reg:g}": train_router(train_data, scheme, reg)
               for scheme, reg in configurations}
    modes = ("hard", "top2", "top4", "top8", "softmax:0.5", "softmax:1", "softmax:2", "sigmoid")
    results = []
    for fold in eval_folds:
        data = distributions(fold, rows, state)
        data = (*data[:3], (data[3] - mean) / std)
        for (scheme, reg), model in zip(configurations, learned.values()):
            key = f"{scheme}_reg{reg:g}"
            for mode in modes:
                metrics = evaluate(data, model, scheme, mode)
                row = {"fold": fold, "variant": key, "mode": mode, "metrics": metrics}
                results.append(row)
                print(row, flush=True)
        del data
    output = os.environ.get("NORA_CORRECTNESS_ROUTER_OUTPUT", "runs/turkish_correctness_router_dev.json")
    payload = {"protocol": "causal per-token features; no train/eval fold overlap; fold3/4 untouched",
               "train_folds": train_folds, "eval_folds": eval_folds, "roles": list(ROLES),
               "model_paths": {str(fold): model_paths(fold) for fold in train_folds + eval_folds},
               "model_sha256": {path: sha256(path) for fold in train_folds + eval_folds
                                for path in model_paths(fold)},
               "feature_mean": mean.tolist(), "feature_std": std.tolist(),
               "weights": {key: tuple(value.tolist() for value in model) for key, model in learned.items()},
               "results": results, "seconds": time.monotonic() - started}
    with open(output, "w") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()

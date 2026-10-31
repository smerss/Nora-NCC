#!/usr/bin/env python3
"""Causal confidence router: fit fold0, development evaluation fold1/2 only.

Her uzmanın dağılımı teacher-forced prefix'ten gelir. Router hedef tokenı görmez;
güven, entropi, marj, uzman oyu ve konum kovasıyla uzman ağırlığı üretir.
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

ROLES = (
    "original_position", "original_wide", "clean_position", "clean_wide",
    "early4_position", "early4_wide", "early8_position", "early8_wide",
    "focal1_position", "focal1_wide", "focal2_position", "focal2_wide",
    "chat_position", "chat_wide", "clean_external", "chat_external",
)
PREFIXES = {
    "original": "pair2", "clean": "clean_init", "early4": "early4", "early8": "early8",
    "focal1": "focal1", "focal2": "focal2", "chat": "chat_init",
}


def model_paths(fold):
    paths = []
    for role in ROLES[:14]:
        family, kind = role.rsplit("_", 1)
        paths.append(f"runs/models/{PREFIXES[family]}_fold{fold}_{kind}.pt")
    paths += ["runs/models/external_clean1000_step1000.pt",
              "runs/models/external_clean_chat1000_step1000.pt"]
    return paths


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def distributions(fold, rows, state):
    organs = [BrocaOrgan.load(path) for path in model_paths(fold)]
    for organ in organs:
        organ.net.eval()
    probabilities, targets, positions, features = [], [], [], []
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
        probability = torch.stack(current, dim=1)  # L,E,V
        top = probability.topk(2, dim=-1)
        maximum = top.values[..., 0]
        margin = maximum - top.values[..., 1]
        entropy = -(probability.clamp_min(1e-30).log() * probability).sum(-1) / np.log(probability.shape[-1])
        prediction = probability.argmax(-1)
        agreement = (prediction[:, :, None] == prediction[:, None, :]).float().mean(-1)
        rank = maximum.argsort(dim=-1, descending=True).argsort(dim=-1).float()
        confidence_rank = 1.0 - rank / max(1, len(organs) - 1)
        feature = torch.stack([maximum.clamp_min(1e-30).log(), entropy, margin,
                               agreement, confidence_rank], dim=-1)
        probabilities.append(probability)
        targets.append(target)
        positions.append(torch.arange(len(target)))
        features.append(feature)
    return (torch.cat(probabilities), torch.cat(targets), torch.cat(positions),
            torch.cat(features))


def buckets_of(positions, scheme):
    if scheme == "global":
        return torch.zeros_like(positions), 1
    if scheme == "three":
        return torch.where(positions < 3, 0, torch.where(positions < 6, 1, 2)), 3
    raise ValueError(scheme)


def initial_weights():
    return torch.tensor([0.9 / 14] * 14 + [0.05, 0.05], dtype=torch.float32)


def train_router(data, scheme, regularization, steps=400):
    probabilities, targets, positions, features = data
    buckets, count = buckets_of(positions, scheme)
    experts = probabilities.shape[1]
    target_probability = probabilities.gather(
        2, targets[:, None, None].expand(-1, experts, -1)).squeeze(-1)
    initial = initial_weights()
    feature_weight = nn.Parameter(torch.zeros(features.shape[-1]))
    expert_bias = nn.Parameter(initial.log())
    position_bias = nn.Parameter(torch.zeros(count, experts))
    optimizer = torch.optim.Adam([feature_weight, expert_bias, position_bias], lr=0.03)
    for _ in range(steps):
        score = (features * feature_weight).sum(-1) + expert_bias + position_bias[buckets]
        weights = score.softmax(dim=-1)
        mixed = (target_probability * weights).sum(-1).clamp_min(1e-30)
        nll = -mixed.log().mean()
        penalty = regularization * ((feature_weight.square().sum())
                                    + (expert_bias - initial.log()).square().sum()
                                    + position_bias.square().sum())
        loss = nll + penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (feature_weight.detach(), expert_bias.detach(), position_bias.detach())


@torch.no_grad()
def evaluate(data, learned, scheme, baseline=False):
    probabilities, targets, positions, features = data
    buckets, count = buckets_of(positions, scheme)
    if baseline:
        weights = initial_weights()[None, :].expand(len(targets), -1)
    else:
        feature_weight, expert_bias, position_bias = learned
        score = (features * feature_weight).sum(-1) + expert_bias + position_bias[buckets]
        weights = score.softmax(dim=-1)
    hit1 = hit5 = 0
    nll = 0.0
    for start in range(0, len(targets), 16):
        end = min(start + 16, len(targets))
        current = (probabilities[start:end] * weights[start:end, :, None]).sum(dim=1).clamp_min(1e-30)
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
    flat = train_data[3].reshape(-1, train_data[3].shape[-1])
    mean, std = flat.mean(0), flat.std(0).clamp_min(1e-6)
    train_data = (*train_data[:3], (train_data[3] - mean) / std)
    configurations = [(scheme, reg) for scheme in ("global", "three")
                      for reg in (0.001, 0.01, 0.1, 1.0)]
    learned = {}
    for scheme, reg in configurations:
        key = f"{scheme}_reg{reg:g}"
        learned[key] = train_router(train_data, scheme, reg)
        print("LEARNED", key, "feature", learned[key][0].tolist(), flush=True)
    results = []
    for fold in (0, 1, 2):
        data = train_data if fold == 0 else distributions(fold, rows, state)
        if fold != 0:
            data = (*data[:3], (data[3] - mean) / std)
        for scheme in ("global", "three"):
            metrics = evaluate(data, None, scheme, baseline=True)
            row = {"fold": fold, "variant": f"baseline_{scheme}", "metrics": metrics}
            results.append(row)
            print(row, flush=True)
        for (scheme, reg), model in zip(configurations, learned.values()):
            key = f"{scheme}_reg{reg:g}"
            metrics = evaluate(data, model, scheme)
            row = {"fold": fold, "variant": key, "metrics": metrics}
            results.append(row)
            print(row, flush=True)
        if fold != 0:
            del data
    payload = {
        "protocol": "fit fold0 only; development fold1/2; fold3/4 untouched; causal per-token features",
        "roles": list(ROLES), "model_paths": {str(fold): model_paths(fold) for fold in (0, 1, 2)},
        "model_sha256": {path: sha256(path) for fold in (0, 1, 2) for path in model_paths(fold)},
        "feature_mean": mean.tolist(), "feature_std": std.tolist(),
        "weights": {key: tuple(value.tolist() for value in model) for key, model in learned.items()},
        "results": results, "seconds": time.monotonic() - started,
    }
    output = os.environ.get("NORA_DYNAMIC_ROUTER_OUTPUT", "runs/turkish_dynamic_router_dev.json")
    with open(output, "w") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Temiz n-gram + 16 neural expert; fold1/2 development, no held-out training."""
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np
import torch

from nora_agi import NoraAGI
from nora_besle import DIALOGUE, SENTENCES
from nora_broca_net import BrocaOrgan
from nora_voice import state_vector
from runs.turkish_dynamic_router_dev import distributions, initial_weights


def add_counts(counts, ids, bos, eos, max_order=4):
    history = [bos] + list(ids)
    targets = list(ids) + [eos]
    for index in range(len(targets)):
        for order in range(1, max_order + 1):
            context = tuple(history[max(0, index + 2 - order):index + 1]) if order > 1 else ()
            counts[order][context][targets[index]] += 1


@torch.no_grad()
def neural_probability(fold, rows, state):
    probabilities, targets, positions, _ = distributions(fold, rows, state)
    base = initial_weights()
    neural = (probabilities * base[None, :, None]).sum(1)
    contexts = []
    organ = BrocaOrgan.load("data/broca_tr.pt")
    for text in rows[fold::5]:
        ids = organ.tok.tk.encode(text).ids[:organ.MAX_LEN]
        if len(ids) < 4:
            continue
        history = [organ.net.bos] + ids
        for index in range(len(ids) + 1):
            contexts.append(tuple(history[:index + 1]))
    assert len(contexts) == len(targets)
    return neural, targets, contexts, organ


def build_counts(fold, rows, organ):
    counts = {order: defaultdict(Counter) for order in range(1, 5)}
    for index, text in enumerate(rows):
        if index % 5 == fold:
            continue
        add_counts(counts, organ.tok.tk.encode(text).ids[:organ.MAX_LEN], organ.net.bos, organ.net.eos)
    source = os.environ.get("NORA_NGRAM_SOURCE", "chatbot_gpl3")
    for line in open(".cache/corpora/processed/external_train.jsonl", encoding="utf8"):
        row = json.loads(line)
        if source != "all" and row["source"] != source:
            continue
        add_counts(counts, organ.tok.tk.encode(row["text"]).ids[:organ.MAX_LEN], organ.net.bos, organ.net.eos)
    return counts


def ngram_probability(contexts, counts, order, vocab, alpha=0.1):
    unigram = counts[1][()]
    result = []
    for history in contexts:
        probability = torch.full((vocab,), 0.01 / (sum(unigram.values()) + 0.01 * vocab))
        for token, count in unigram.items():
            probability[token] += count / (sum(unigram.values()) + 0.01 * vocab)
        for current_order in range(2, order + 1):
            context = tuple(history[-(current_order - 1):])
            found = counts[current_order].get(context)
            if not found:
                continue
            total = sum(found.values())
            probability = probability * (alpha / (total + alpha))
            for token, count in found.items():
                probability[token] += count / (total + alpha)
        result.append(probability)
    return torch.stack(result)


@torch.no_grad()
def evaluate(probability, targets):
    probability = probability.clamp_min(1e-30)
    top = probability.topk(5, dim=-1).indices
    hit1 = int((top[:, 0] == targets).sum())
    hit5 = int((top == targets[:, None]).any(-1).sum())
    nll = -float(probability.log().gather(1, targets[:, None]).sum())
    n = len(targets)
    return {"n": n, "hit1": hit1, "hit5": hit5, "top1": hit1 / n,
            "top5": hit5 / n, "ppl": float(np.exp(nll / n))}


def main():
    torch.set_num_threads(1)
    rows = SENTENCES + DIALOGUE
    state = state_vector(NoraAGI(seed=140, boot_thinker=False))
    results = []
    started = time.monotonic()
    for fold in (1, 2):
        neural, targets, contexts, organ = neural_probability(fold, rows, state)
        counts = build_counts(fold, rows, organ)
        for order in (2, 3, 4):
            ngram = ngram_probability(contexts, counts, order, organ.net.V)
            for neural_weight in (1.0, 0.95, 0.9, 0.75, 0.5, 0.0):
                probability = neural_weight * neural + (1.0 - neural_weight) * ngram
                metrics = evaluate(probability, targets)
                row = {"fold": fold, "order": order, "neural_weight": neural_weight,
                       "source": os.environ.get("NORA_NGRAM_SOURCE", "chatbot_gpl3"), **metrics}
                results.append(row)
                print(row, flush=True)
            del ngram
        del neural, counts
    output = os.environ.get("NORA_NGRAM_CROSSFOLD_OUTPUT", "runs/turkish_ngram_crossfold_dev.json")
    with open(output, "w") as handle:
        json.dump({"results": results, "seconds": time.monotonic() - started,
                   "protocol": "target fold train + decontaminated external; no fold3/4"}, handle, indent=2)


if __name__ == "__main__":
    main()

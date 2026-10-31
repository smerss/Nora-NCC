#!/usr/bin/env python3
"""Salt-okuma Türkçe onamı: taban, iki-uzman ensemble veya eğitilmiş token router."""
import hashlib
import json
import os
import resource
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nora_agi import NoraAGI
from nora_broca_net import BrocaOrgan
from nora_voice import state_vector
from runs.moe_domain_router import token_confidence_evaluate


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def texts_of(path):
    if path.endswith(".jsonl"):
        return [json.loads(line)["text"] for line in open(path) if line.strip()]
    return [x.strip() for x in open(path) if x.strip()]


@torch.no_grad()
def probability_evaluate(organs, weights, texts, state, batch_size=64):
    for organ in organs:
        organ.net.eval()
    hit1 = hit5 = n = 0
    nll = 0.0
    batch_size = max(1, int(batch_size))
    for start in range(0, len(texts), batch_size):
        enc = organs[0].tok.tk.encode_batch(texts[start:start + batch_size])
        rows = [e.ids[:organs[0].MAX_LEN] for e in enc if len(e.ids) >= 4]
        if not rows:
            continue
        length = max(map(len, rows)) + 1
        batch = len(rows)
        inp = torch.full((batch, length), organs[0].net.pad, dtype=torch.long)
        tgt = torch.full((batch, length), -100, dtype=torch.long)
        for b, row in enumerate(rows):
            inp[b, 0] = organs[0].net.bos
            inp[b, 1:len(row) + 1] = torch.tensor(row)
            tgt[b, :len(row)] = torch.tensor(row)
            tgt[b, len(row)] = organs[0].net.eos
        bag = torch.zeros(batch, organs[0].net.V)
        st = torch.from_numpy(np.tile(state.astype(np.float32), (batch, 1)))
        mask = tgt != -100
        probability = None
        for organ, weight in zip(organs, weights):
            logits, _ = organ.net(inp, organ.net.encode_cond(st, bag))
            current = torch.softmax(logits[mask].float(), dim=-1) * float(weight)
            probability = current if probability is None else probability + current
        probability = probability.clamp_min(1e-30)
        yt = tgt[mask]
        top = probability.topk(5, dim=-1).indices
        hit1 += int((top[:, 0] == yt).sum())
        hit5 += int((top == yt[:, None]).any(-1).sum())
        n += len(yt)
        nll -= float(probability.log().gather(1, yt[:, None]).sum())
    return {"n": n, "hit1": hit1, "hit5": hit5, "top1": hit1 / max(1, n),
            "top5": hit5 / max(1, n), "ppl": float(np.exp(nll / max(1, n)))}


def main():
    torch.set_num_threads(1)
    mode = os.environ.get("NORA_CONFIRM_MODE", "base")
    corpus = os.environ["NORA_CONFIRM_CORPUS"]
    output = os.environ["NORA_CONFIRM_OUTPUT"]
    position_path = os.environ.get("NORA_POSITION_MODEL", "runs/models/position_gated_knn.pt")
    wide_path = os.environ.get("NORA_WIDE_MODEL", "runs/models/wide143_best.pt")
    external_path = os.environ.get("NORA_EXTERNAL_MODEL", "runs/models/external_rebuilt_best.pt")
    base_path = os.environ.get("NORA_BASE_MODEL", "data/broca_tr.pt")
    weight = float(os.environ.get("NORA_POSITION_WEIGHT", "0.65"))
    batch_size = int(os.environ.get("NORA_EVAL_BATCH", "64"))
    texts = texts_of(corpus)
    max_sent = int(os.environ.get("NORA_CONFIRM_MAX_SENT", "0"))
    if max_sent:
        texts = texts[:max_sent]
    state = state_vector(NoraAGI(seed=140, boot_thinker=False))
    started = time.monotonic()
    if mode == "base":
        models = [base_path]
        metrics = probability_evaluate([BrocaOrgan.load(base_path)], [1.0], texts, state,
                                       batch_size=batch_size)
    elif mode == "ensemble":
        models = [position_path, wide_path]
        metrics = probability_evaluate([BrocaOrgan.load(position_path), BrocaOrgan.load(wide_path)],
                                       [weight, 1.0 - weight], texts, state, batch_size=batch_size)
    elif mode == "router":
        router_path = os.environ["NORA_ROUTER_MODEL"]
        models = [external_path, position_path, wide_path, router_path]
        external, position, wide = [BrocaOrgan.load(x) for x in models[:3]]
        saved = torch.load(router_path, weights_only=False)
        router = nn.Linear(saved["router"]["weight"].shape[1], 1)
        router.load_state_dict(saved["router"])
        metrics = token_confidence_evaluate(
            router, saved["mean"], saved["std"], external, position, wide, texts, state,
            saved["ood_threshold"], token_ratio=saved.get("token_ratio", 0.75),
            prefix=saved.get("prefix", 3), position_weight=saved.get("position_weight", weight),
            gate=saved.get("gate", "embedding"), nll_threshold=saved.get("nll_threshold", 0.0),
            batch_size=batch_size)
    else:
        raise ValueError("NORA_CONFIRM_MODE base/ensemble/router olmalı")
    result = {"mode": mode, "corpus": corpus, "corpus_sha256": sha256(corpus),
              "input_sentences": len(texts), "models": models,
              "model_sha256": {path: sha256(path) for path in models},
              "position_weight": weight, "batch_size": batch_size,
              "seconds": time.monotonic() - started,
              "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "metrics": metrics}
    with open(output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

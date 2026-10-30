#!/usr/bin/env python3
"""Aynı Türkçe next-subword split/bütçede scratch GRU ve Transformer A/B."""
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nora_broca_net import BrocaOrgan, N_SPECIAL
from nora_besle import DIALOGUE, SENTENCES
from nora_agi import NoraAGI
from nora_voice import state_vector


class TokenTransformer(nn.Module):
    def __init__(self, vocab: int, state_size: int, d: int = 256):
        super().__init__()
        self.V = vocab + N_SPECIAL
        self.pad, self.bos, self.eos = vocab, vocab + 1, vocab + 2
        self.emb = nn.Embedding(self.V, d)
        self.pos = nn.Embedding(65, d)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos.weight, mean=0.0, std=0.02)
        self.state = nn.Linear(state_size, d)
        layer = nn.TransformerEncoderLayer(
            d, nhead=4, dim_feedforward=512, dropout=0.1,
            batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(d)

    def forward(self, ids, state):
        length = ids.shape[1]
        positions = torch.arange(length, device=ids.device)
        x = self.emb(ids) + self.pos(positions)[None] + self.state(state)[:, None]
        causal = torch.triu(torch.full((length, length), float("-inf"), device=ids.device), 1)
        padding = ids == self.pad
        z = self.norm(self.blocks(x, mask=causal, src_key_padding_mask=padding))
        return z @ self.emb.weight.t()


def batches(tokenizer, texts, net, max_len=40):
    enc = tokenizer.tk.encode_batch(texts)
    rows = [e.ids[:max_len] for e in enc if len(e.ids) >= 4]
    for start in range(0, len(rows), 64):
        chunk = rows[start:start + 64]
        length = max(map(len, chunk)) + 1
        inp = torch.full((len(chunk), length), net.pad, dtype=torch.long)
        tgt = torch.full((len(chunk), length), -100, dtype=torch.long)
        for b, row in enumerate(chunk):
            inp[b, 0] = net.bos
            inp[b, 1:len(row) + 1] = torch.tensor(row)
            tgt[b, :len(row)] = torch.tensor(row)
            tgt[b, len(row)] = net.eos
        yield inp, tgt


@torch.no_grad()
def evaluate_transformer(net, tokenizer, texts, state):
    net.eval()
    hit1 = hit5 = n = 0
    nll = 0.0
    for inp, tgt in batches(tokenizer, texts, net):
        st = torch.from_numpy(np.tile(state.astype(np.float32), (len(inp), 1)))
        logits = net(inp, st)
        mask = tgt != -100
        scores = logits[mask]
        yt = tgt[mask]
        top = scores.topk(5, dim=-1).indices
        hit1 += int((top[:, 0] == yt).sum())
        hit5 += int((top == yt[:, None]).any(-1).sum())
        n += len(yt)
        nll += float(F.cross_entropy(scores, yt, reduction="sum"))
    net.train()
    return {"n": n, "top1": hit1 / max(1, n), "top5": hit5 / max(1, n),
            "ppl": float(math.exp(nll / max(1, n)))}


def train_transformer(net, tokenizer, texts, state, lr):
    net.train()
    previous_dropout = tokenizer.tk.model.dropout
    try:
        tokenizer.tk.model.dropout = 0.1
        enc = tokenizer.tk.encode_batch(texts)
    finally:
        tokenizer.tk.model.dropout = previous_dropout
    rows = [e.ids[:40] for e in enc if len(e.ids) >= 3]
    length = max(map(len, rows)) + 1
    inp = torch.full((len(rows), length), net.pad, dtype=torch.long)
    tgt = torch.full((len(rows), length), -100, dtype=torch.long)
    for b, row in enumerate(rows):
        inp[b, 0] = net.bos
        inp[b, 1:len(row) + 1] = torch.tensor(row)
        tgt[b, :len(row)] = torch.tensor(row)
        tgt[b, len(row)] = net.eos
    st = torch.from_numpy(np.tile(state.astype(np.float32), (len(rows), 1)))
    logits = net(inp, st)
    loss = F.cross_entropy(logits[tgt != -100], tgt[tgt != -100])
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    return float(loss.detach())


def main():
    torch.set_num_threads(1)
    rows = SENTENCES + DIALOGUE
    held = rows[::5]
    base_train = [x for i, x in enumerate(rows) if i % 5]
    state = state_vector(NoraAGI(seed=140, boot_thinker=False))
    base = BrocaOrgan.load("data/broca_tr.pt")
    results = []
    global optimizer
    architectures = ("transformer",) if os.environ.get("NORA_TRANSFORMER_ONLY") else ("gru", "transformer")
    transformer_lr = float(os.environ.get("NORA_TRANSFORMER_LR", "0.0005"))
    output = os.environ.get("NORA_TRANSFORMER_OUTPUT", "runs/turkish_transformer_comparison.json")
    for seed in (140, 141, 142):
        train = list(base_train)
        random.Random(seed).shuffle(train)
        for architecture in architectures:
            torch.manual_seed(seed)
            if architecture == "gru":
                model = BrocaOrgan(base.tok, seed=seed, h=512, dropout=0.1)
                nn.init.normal_(model.net.emb.weight, mean=0.0, std=0.02)
                model.rng = np.random.default_rng(90000 + seed)
                params = sum(p.numel() for p in model.net.parameters())
            else:
                model = TokenTransformer(base.tok.tk.get_vocab_size(), len(state))
                optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
                params = sum(p.numel() for p in model.parameters())
            start = time.time()
            global_step = 0
            for epoch in range(20):
                for i in range(0, len(train), 32):
                    global_step += 1
                    if architecture == "transformer":
                        lr = (transformer_lr * min(1.0, global_step / 100)
                              / (1 + epoch / 20) ** 0.5)
                    else:
                        lr = 5e-4 / (1 + epoch / 20) ** 0.5
                    chunk = train[i:i + 32]
                    if architecture == "gru":
                        model.learn_batch(chunk, state, lr=lr, anchor_dropout=1.0,
                                          loss_mode="full", bpe_dropout=0.1)
                    else:
                        train_transformer(model, base.tok, chunk, state, lr)
                if architecture == "gru":
                    metrics = model.evaluate(held, state, use_anchor=False, max_sent=10000)
                else:
                    metrics = evaluate_transformer(model, base.tok, held, state)
                row = {"seed": seed, "architecture": architecture, "parameters": params,
                       "epoch": epoch + 1, "seconds": time.time() - start,
                       "initialization": "scratch", "loss": "full_softmax",
                       "bpe_dropout": 0.1, "anchor_dropout": 1.0,
                       "lr_final": lr, "warmup_steps": 100 if architecture == "transformer" else 0,
                       **metrics}
                results.append(row)
                print(row, flush=True)
                with open(output, "w") as f:
                    json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fair, experimental GRU vs small causal-attention decoder comparison.

This is a measurement tool, not another Nora cognitive organ.  Both models are
trained from scratch on the *same* one-pass stream, tokenizer, anchor bags,
full-vocabulary loss and update budget.  The tokenizer is fitted after the
sentence split, on training sentences only.  No pretrained weights, replay,
epoch, hidden validation training or model-specific decoding heuristic is used.

Example (CPU smoke):
    python tools/compare_broca_decoders.py --limit 96 --seeds 0 1 --d 16 --h 16

The JSON report deliberately separates held-out next-token metrics from surface
metrics.  Unique surfaces are not presented as evidence of semantic fluency.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

# tools/ altından doğrudan çalıştırıldığında da depo kökü import yolunda olsun.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nora.language.broca_net import N_SPECIAL, BrocaOrgan
from nora.language.voice import NF, SubwordTokenizer


class CausalAttentionNet(nn.Module):
    """Small scratch causal decoder with the same conditioning/output contract.

    ``BrocaOrgan.learn_batch`` and ``evaluate`` only need this common contract;
    the attention model is never attached to Nora's living brain.
    """

    def __init__(self, vocab: int, d: int = 32, h: int = 32, heads: int = 4):
        super().__init__()
        if d % heads:
            raise ValueError("d, attention head sayısına bölünmeli")
        self.V = vocab + N_SPECIAL
        self.pad, self.bos, self.eos = vocab, vocab + 1, vocab + 2
        self.h = h
        self.emb = nn.Embedding(self.V, d)
        self.cond = nn.Sequential(nn.Linear(NF + d, h), nn.Tanh())
        self.cond_in = nn.Linear(h, d, bias=False)
        self.pos_emb = nn.Embedding(65, d)
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d),
        )
        self.proj = nn.Linear(d, d)
        self.out_bias = None

    def encode_cond(self, state: torch.Tensor, anchor_bag: torch.Tensor) -> torch.Tensor:
        n = anchor_bag.sum(1, keepdim=True).clamp(min=1.0)
        anchor = (anchor_bag @ self.emb.weight) / n
        return self.cond(torch.cat([state, anchor], dim=1))

    def output_weight(self):
        return self.emb.weight

    def hidden(self, ids: torch.Tensor, c: torch.Tensor, h0=None, position_offset: int = 0):
        del h0  # full-prefix causal computation; no recurrent cache is hidden here
        x = self.emb(ids) + self.cond_in(c).unsqueeze(1)
        positions = torch.arange(
            position_offset, position_offset + ids.shape[1], device=ids.device,
        )
        x = x + self.pos_emb(positions).unsqueeze(0)
        causal = torch.triu(
            torch.full((ids.shape[1], ids.shape[1]), float("-inf"), device=ids.device),
            diagonal=1,
        )
        y = self.ln1(x)
        attended, _ = self.attn(y, y, y, attn_mask=causal, need_weights=False)
        x = x + attended
        x = x + self.mlp(self.ln2(x))
        return self.proj(x), None

    def forward(self, ids: torch.Tensor, c: torch.Tensor, h0=None, position_offset: int = 0):
        z, _ = self.hidden(ids, c, h0, position_offset)
        return z @ self.output_weight().t(), None


def _sentences(path: Path, limit: int) -> list[str]:
    rows = []
    seen = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = re.sub(r"\s+", " ", raw).strip()
        if len(re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)) < 4:
            continue
        if text in seen:
            continue
        seen.add(text)
        rows.append(text)
        if len(rows) >= limit:
            break
    if len(rows) < 24:
        raise ValueError("adil split için en az 24 uygun cümle gerekli")
    return rows


def _split(rows: list[str], seed: int = 2027):
    order = np.random.default_rng(seed).permutation(len(rows))
    shuffled = [rows[int(i)] for i in order]
    n_train = max(8, int(len(shuffled) * 0.8))
    n_valid = max(4, int(len(shuffled) * 0.1))
    train = shuffled[:n_train]
    valid = shuffled[n_train:n_train + n_valid]
    test = shuffled[n_train + n_valid:]
    if not test:
        test = valid[-1:]
        valid = valid[:-1]
    return train, valid, test


def _make_organ(kind: str, tok: SubwordTokenizer, seed: int, d: int, h: int,
                match_parameters: bool = True) -> BrocaOrgan:
    # Both are scratch models.  Constructing the same wrapper also gives both
    # the same stream RNG, anchor-bag procedure and optimizer hyperparameters.
    organ = BrocaOrgan(tok, seed=seed, d=d, h=h)
    organ.experiment_d = d
    if kind == "attention":
        target = sum(p.numel() for p in organ.net.parameters())
        attention_d = d
        if match_parameters:
            # Aynı token/güncelleme bütçesine ek olarak toplam parametre
            # bütçesini de yaklaşık eşitle. Aday kurulumlar RNG'yi tüketmesin.
            old_rng = torch.get_rng_state()
            choices = []
            for width in range(4, max(8, 2 * d) + 1, 4):
                torch.manual_seed(0)
                candidate = CausalAttentionNet(tok.tk.get_vocab_size(), d=width, h=h)
                choices.append((abs(sum(p.numel() for p in candidate.parameters()) - target), width))
            torch.set_rng_state(old_rng)
            attention_d = min(choices)[1]
        torch.manual_seed(seed + 10_000)
        organ.net = CausalAttentionNet(tok.tk.get_vocab_size(), d=attention_d, h=h)
        organ.experiment_d = attention_d
        organ.experiment_parameter_target = target
        organ._opt = None
    return organ


def _evaluate_fixed(organ: BrocaOrgan, rows: list[str], state: np.ndarray, seed: int):
    # evaluate() derives anchor bags with RNG.  Reset it for each model/split so
    # the held-out conditioning examples are identical, not merely similar.
    old = copy.deepcopy(organ.rng.bit_generator.state)
    organ.rng = np.random.default_rng(seed)
    try:
        return organ.evaluate(rows, state, k=5, use_anchor=True)
    finally:
        organ.rng.bit_generator.state = old


@torch.no_grad()
def _generate_shared(organ: BrocaOrgan, anchor: str, state: np.ndarray,
                     seed: int, count: int = 6) -> list[str]:
    """Same full-prefix decoder and mask for both architectures.

    This mirrors the production mask (common pieces + anchor subwords + EOS),
    including forced continuation of a split anchor.  It is intentionally
    independent of either model's private generation implementation.
    """
    net = organ.net
    net.eval()
    rng = torch.Generator().manual_seed(seed)
    anchor_ids = organ.tok.tk.encode(anchor).ids
    if not anchor_ids:
        return []
    starts = {anchor_ids[0]: anchor_ids[1:]}
    allowed = torch.zeros(net.V, dtype=torch.bool)
    if organ.common_ids is not None:
        allowed[organ.common_ids] = True
    allowed[list(starts)] = True
    allowed[net.eos] = True
    bag = torch.zeros(1, net.V)
    bag[0, anchor_ids] = 1.0
    st = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
    condition = net.encode_cond(st, bag)
    surfaces = []
    for _ in range(count):
        prefix = [net.bos]
        output = []
        queue = []
        remaining = {anchor_ids[0]}
        for _position in range(24):
            ids = torch.tensor([prefix], dtype=torch.long)
            z, _ = net.hidden(ids, condition)
            logits = z[:, -1, :] @ net.output_weight().t()
            if net.out_bias is not None:
                logits = logits + net.out_bias
            logits = logits / 0.8
            logits[:, ~allowed] = -1e9
            logits[:, net.pad] = -1e9
            logits[:, net.bos] = -1e9
            for token in remaining:
                logits[0, token] += 4.0
            for token in set(output):
                logits[0, token] -= (3.0 if token in starts else 1.5) * output.count(token)
            kth = logits.topk(min(30, net.V), dim=-1).values[:, -1:]
            logits[logits < kth] = -1e9
            nxt = int(torch.multinomial(F.softmax(logits, dim=-1), 1, generator=rng)[0, 0])
            if queue:
                nxt = queue.pop(0)
            if nxt == net.eos:
                break
            if nxt in starts and not queue:
                queue = list(starts[nxt])
            output.append(nxt)
            remaining.discard(nxt)
            prefix.append(nxt)
            if len(output) >= 24 and not queue:
                break
        text = organ.detok(output).strip()
        if len(text.split()) >= 2:
            surfaces.append(text)
    net.train()
    return surfaces


def _anchors(rows: list[str], tok: SubwordTokenizer) -> list[str]:
    counts = Counter(
        w.casefold() for row in rows
        for w in re.findall(r"[^\W\d_]+", row, flags=re.UNICODE)
        if len(w) > 2
    )
    result = []
    for word, _ in counts.most_common():
        if word not in {"için", "gibi", "kadar", "olan", "olarak", "bir", "daha"}:
            if tok.tk.encode(word).ids:
                result.append(word)
        if len(result) == 3:
            break
    return result


def _run(kind: str, seed: int, train: list[str], valid: list[str], test: list[str],
         tok: SubwordTokenizer, state: np.ndarray, args):
    organ = _make_organ(kind, tok, seed, args.d, args.h, args.match_parameters)
    started = time.perf_counter()
    losses = []
    updates = 0
    # One stream pass for each model.  No validation-driven early stopping.
    for start in range(0, len(train), args.batch):
        batch = train[start:start + args.batch]
        losses.append(organ.learn_batch(
            batch, state, loss_mode="full", anchor_dropout=args.anchor_dropout,
        ))
        updates += 1
    train_seconds = time.perf_counter() - started
    organ.update_common(min(150, organ.V))
    valid_metrics = _evaluate_fixed(organ, valid, state, 50_000 + seed)
    test_metrics = _evaluate_fixed(organ, test, state, 60_000 + seed)
    surface_rows = []
    for index, anchor in enumerate(_anchors(train, tok)):
        generated = _generate_shared(organ, anchor, state, 70_000 + seed + index)
        surface_rows.append({
            "anchor": anchor,
            "n": len(generated),
            "unique": len(set(generated)),
            "anchor_coverage": (sum(anchor.casefold() in text.casefold() for text in generated)
                                / max(1, len(generated))),
            "texts": generated,
        })
    return {
        "architecture": kind,
        "seed": seed,
        "parameters": sum(p.numel() for p in organ.net.parameters()),
        "d": int(getattr(organ, "experiment_d", args.d)),
        "parameter_target": int(getattr(organ, "experiment_parameter_target", 0) or 0),
        "updates": updates,
        "train_tokens": int(organ.seen_tokens),
        "train_seconds": train_seconds,
        "loss_first": float(losses[0]) if losses else None,
        "loss_last": float(losses[-1]) if losses else None,
        "valid": valid_metrics,
        "test": test_metrics,
        "surface": surface_rows,
    }


def _summary(runs: list[dict]):
    result = {}
    for kind in ("gru", "attention"):
        selected = [r for r in runs if r["architecture"] == kind]
        if not selected:
            continue
        result[kind] = {
            "seeds": [r["seed"] for r in selected],
            "parameters": selected[0]["parameters"],
            "train_tokens": selected[0]["train_tokens"],
            "updates": selected[0]["updates"],
            "test_nll_mean": float(np.mean([
                math.log(max(1e-12, r["test"]["ppl"])) for r in selected
            ])),
            "test_top1_mean": float(np.mean([r["test"]["top1"] for r in selected])),
            "test_top1_std": float(np.std([r["test"]["top1"] for r in selected])),
            "train_seconds_mean": float(np.mean([r["train_seconds"] for r in selected])),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="NORA_EVREN.md")
    parser.add_argument("--limit", type=int, default=384)
    parser.add_argument("--vocab", type=int, default=512)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--h", type=int, default=32)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--anchor-dropout", type=float, default=0.1)
    parser.add_argument("--match-parameters", action=argparse.BooleanOptionalAction, default=True,
                        help="attention genişliğini GRU parametre bütçesine yaklaşık eşitle")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", default="", help="JSON'u ayrıca bu yola yaz")
    args = parser.parse_args()
    if args.batch < 1 or args.limit < 24 or not args.seeds:
        raise ValueError("batch>=1, limit>=24 ve en az bir seed gerekli")
    torch.set_num_threads(args.threads)
    rows = _sentences(Path(args.data), args.limit)
    train, valid, test = _split(rows)
    # Fit only after splitting: held-out vocabulary/merges cannot leak in.
    tok = SubwordTokenizer()
    tok.train(train, vocab_size=args.vocab)
    state = np.full(NF, 0.4, dtype=np.float32)
    runs = []
    for seed in args.seeds:
        for kind in ("gru", "attention"):
            print(f"[{kind}] seed={seed} train={len(train)} ...", flush=True)
            runs.append(_run(kind, seed, train, valid, test, tok, state, args))
    report = {
        "experiment": "scratch_gru_vs_scratch_causal_attention",
        "pretrained": False,
        "one_pass": True,
        "replay": False,
        "data": str(args.data),
        "split": {"seed": 2027, "train": len(train), "valid": len(valid), "test": len(test)},
        "tokenizer": {"fit": "train_only", "vocab_requested": args.vocab,
                       "vocab_actual": tok.tk.get_vocab_size()},
        "budget": {"batch": args.batch, "d": args.d, "h": args.h,
                    "anchor_dropout": args.anchor_dropout, "loss": "full_softmax",
                    "match_parameters": bool(args.match_parameters)},
        "runs": runs,
        "summary": _summary(runs),
        "caveat": "Yüzey çeşitliliği akıcılık veya anlam doğruluğu kanıtı değildir; held-out NLL/top-k ayrı raporlanır.",
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    print(encoded)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

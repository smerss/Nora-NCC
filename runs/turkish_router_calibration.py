#!/usr/bin/env python3
"""Router eşiğini XQuAD/FAQ'a bakmadan target-dev + görülmemiş haricî akışla kalibre et."""
import json
import os

import numpy as np
import torch
from torch import nn

from nora_agi import NoraAGI
from nora_besle import DIALOGUE, SENTENCES
from nora_broca_net import BrocaOrgan
from nora_voice import state_vector
from runs.moe_domain_router import prefix_features


@torch.no_grad()
def collect(external, position, wide, router, mean, std, texts, state, position_weight):
    output = {key: [] for key in ("target_hit1", "target_hit5", "external_hit1", "external_hit5",
                                  "target_nll", "external_nll", "target_max", "external_max",
                                  "probability", "observed", "sentence")}
    sentence_offset = 0
    for start in range(0, len(texts), 64):
        enc = external.tok.tk.encode_batch(texts[start:start + 64])
        rows = [e.ids[:40] for e in enc if len(e.ids) >= 4]
        if not rows:
            continue
        length = max(map(len, rows)) + 1
        batch = len(rows)
        inp = torch.full((batch, length), external.net.pad, dtype=torch.long)
        tgt = torch.full((batch, length), -100, dtype=torch.long)
        for b, row in enumerate(rows):
            inp[b, 0] = external.net.bos
            inp[b, 1:len(row) + 1] = torch.tensor(row)
            tgt[b, :len(row)] = torch.tensor(row)
            tgt[b, len(row)] = external.net.eos
        probability = torch.sigmoid(router(prefix_features(
            external, [row[:3] for row in rows], mean, std)).squeeze(1))
        bag = torch.zeros(batch, external.net.V)
        st = torch.from_numpy(np.tile(state.astype(np.float32), (batch, 1)))
        probabilities = []
        for organ in (external, position, wide):
            logits, _ = organ.net(inp, organ.net.encode_cond(st, bag))
            probabilities.append(torch.softmax(logits.float(), dim=-1))
        mask = tgt != -100
        external_p = probabilities[0][mask]
        target_p = (position_weight * probabilities[1]
                    + (1 - position_weight) * probabilities[2])[mask]
        yt = tgt[mask]
        for name, probs in (("target", target_p), ("external", external_p)):
            top = probs.topk(5, dim=-1).indices
            output[name + "_hit1"].append((top[..., 0] == yt).numpy())
            output[name + "_hit5"].append((top == yt[..., None]).any(-1).numpy())
            output[name + "_nll"].append((-probs.clamp_min(1e-30).log().gather(
                -1, yt[..., None]).squeeze(-1)).numpy())
            output[name + "_max"].append(probs.max(dim=-1).values.numpy())
        output["probability"].append(probability[:, None].expand(batch, length)[mask].numpy())
        output["observed"].append((torch.arange(length)[None, :] >= 3).expand(batch, length)[mask].numpy())
        output["sentence"].append((torch.arange(batch)[:, None] + sentence_offset).expand(batch, length)[mask].numpy())
        sentence_offset += batch
    return {key: np.concatenate(value) for key, value in output.items()}


def attach_prefix_score(stats):
    scores = {}
    for sentence in np.unique(stats["sentence"]):
        indices = np.flatnonzero(stats["sentence"] == sentence)[:3]
        # Pozitif değer: haricî uzman ilk üç gözlenen parçayı daha iyi öngörmüş.
        scores[int(sentence)] = float((stats["target_nll"][indices]
                                       - stats["external_nll"][indices]).sum())
    stats["prefix_score"] = np.asarray([scores[int(x)] for x in stats["sentence"]])
    return stats


def evaluate(stats, ratio, threshold, gate="embedding"):
    sentence_ood = (stats["probability"] < threshold if gate == "embedding"
                    else stats["prefix_score"] > threshold)
    ood = sentence_ood & stats["observed"]
    use_target = (stats["target_max"] >= ratio * stats["external_max"]) & ~ood
    n = len(use_target)
    hit1 = int(np.where(use_target, stats["target_hit1"], stats["external_hit1"]).sum())
    hit5 = int(np.where(use_target, stats["target_hit5"], stats["external_hit5"]).sum())
    nll = float(np.where(use_target, stats["target_nll"], stats["external_nll"]).sum())
    return {"n": n, "hit1": hit1, "hit5": hit5, "top1": hit1 / n, "top5": hit5 / n,
            "ppl": float(np.exp(nll / n)), "target_fraction": float(use_target.mean()),
            "ood_sentences": int(np.unique(stats["sentence"][sentence_ood]).size)}


def main():
    torch.set_num_threads(1)
    external = BrocaOrgan.load(os.environ.get("NORA_EXTERNAL_MODEL", "data/checkpoints/turkish_external.pt"))
    position_path = os.environ.get("NORA_POSITION_MODEL", "data/checkpoints/turkish_position144.pt")
    wide_path = os.environ.get("NORA_WIDE_MODEL", "data/checkpoints/turkish_wide149.pt")
    position = BrocaOrgan.load(position_path)
    wide = BrocaOrgan.load(wide_path)
    for organ in (external, position, wide):
        organ.net.eval()
    router_path = os.environ.get("NORA_ROUTER_MODEL", "data/checkpoints/turkish_router_seed140.pt")
    saved = torch.load(router_path, weights_only=False)
    router = nn.Linear(saved["router"]["weight"].shape[1], 1)
    router.load_state_dict(saved["router"])
    mean, std = saved["mean"], saved["std"]
    weight = float(os.environ.get("NORA_POSITION_WEIGHT", saved.get("position_weight", 0.65)))
    state = state_vector(NoraAGI(seed=140, boot_thinker=False))
    target = (SENTENCES + DIALOGUE)[::5]
    # İlk 112.000 cümle haricî uzman eğitiminde, ilk 1.000'i router eğitiminde kullanıldı.
    # Son 1.000 cümle her iki kurulumun da dışındadır; XQuAD ve FAQ bu fonksiyona girmez.
    external_texts = [json.loads(line)["text"] for line in open(
        ".cache/corpora/processed/external_train.jsonl")][-1000:]
    print("collect target", flush=True)
    target_stats = attach_prefix_score(collect(
        external, position, wide, router, mean, std, target, state, weight))
    print("collect external calibration", flush=True)
    external_stats = attach_prefix_score(collect(
        external, position, wide, router, mean, std, external_texts, state, weight))
    current = float(saved["ood_threshold"])
    thresholds = sorted(set([0.0, 0.5, 0.9, 0.99, current]
                            + list(np.geomspace(current, 0.5, 24))))
    rows = []
    for ratio in (0.5, 0.6, 0.65, 0.75, 0.85, 0.95, 1.05):
        for threshold in thresholds:
            row = {"gate": "embedding", "token_ratio": ratio, "ood_threshold": float(threshold),
                   "target": evaluate(target_stats, ratio, threshold, "embedding"),
                   "external_calibration": evaluate(external_stats, ratio, threshold, "embedding")}
            rows.append(row)
        score_values = np.concatenate((target_stats["prefix_score"], external_stats["prefix_score"]))
        nll_thresholds = sorted(set([0.0] + [float(x) for x in np.quantile(
            score_values, np.linspace(0, 1, 25))]))
        for threshold in nll_thresholds:
            row = {"gate": "nll_prefix", "token_ratio": ratio, "nll_threshold": threshold,
                   "target": evaluate(target_stats, ratio, threshold, "nll_prefix"),
                   "external_calibration": evaluate(external_stats, ratio, threshold, "nll_prefix")}
            rows.append(row)
    eligible = [row for row in rows if row["target"]["hit1"] >= 177]
    if not eligible:
        raise RuntimeError("177/393 target güvenlik kısıtını sağlayan router yok")
    # Önce dış-akış top-1; eşitlikte target top-1, sonra daha düşük dış-akış PPL'si.
    selected = max(eligible, key=lambda row: (row["external_calibration"]["hit1"],
                                              row["target"]["hit1"],
                                              -row["external_calibration"]["ppl"]))
    result = {"selection": "target hit1>=177; maximize external calibration hit1/top1/ppl",
              "external_calibration_sentences": len(external_texts),
              "external_calibration_slice": "son 1000; external eğitim ve router train dışı",
              "xquad_or_faq_used": False, "position_weight": weight,
              "model_paths": {"position": position_path, "wide": wide_path, "router": router_path},
              "selected": selected, "grid": rows}
    output = os.environ.get("NORA_CALIBRATION_OUTPUT", "runs/turkish_router_calibration.json")
    with open(output, "w") as handle:
        json.dump(result, handle, indent=2)
    saved["gate"] = selected["gate"]
    if selected["gate"] == "embedding":
        saved["ood_threshold"] = selected["ood_threshold"]
    else:
        saved["nll_threshold"] = selected["nll_threshold"]
    saved["token_ratio"] = selected["token_ratio"]
    saved["position_weight"] = weight
    model_output = os.environ.get("NORA_CALIBRATED_MODEL_OUTPUT",
                                  "data/checkpoints/turkish_router_seed140_calibrated.pt")
    torch.save(saved, model_output)
    print(json.dumps(selected, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""İlk-token domain router ile target/haricî Broca uzmanları; gelecek token kullanılmaz."""
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nora_broca_net import BrocaOrgan
from nora_besle import DIALOGUE, SENTENCES
from nora_agi import NoraAGI
from nora_voice import state_vector


def first_ids(organ, texts):
    return [e.ids[0] for e in organ.tok.tk.encode_batch(texts) if len(e.ids) >= 4]


@torch.no_grad()
def routed_evaluate(router, mean, std, external, target, texts, state):
    external.net.eval()
    target.net.eval()
    hit1 = hit5 = n = routed_target = 0
    nll = 0.0
    enc = external.tok.tk.encode_batch(texts)
    for start in range(0, len(enc), 64):
        rows = [e.ids[:external.MAX_LEN] for e in enc[start:start + 64] if len(e.ids) >= 4]
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
        ids = torch.tensor([row[0] for row in rows])
        features = (external.net.emb.weight[ids] - mean) / std
        probability = torch.sigmoid(router(features)).squeeze(1)
        choose_target = probability >= 0.5
        routed_target += int(choose_target.sum())
        bag = torch.zeros(batch, external.net.V)
        st = torch.from_numpy(np.tile(state.astype(np.float32), (batch, 1)))
        external_logits, _ = external.net(inp, external.net.encode_cond(st, bag))
        target_logits, _ = target.net(inp, target.net.encode_cond(st, bag))
        use = choose_target[:, None, None].expand(-1, length, -1).clone()
        use[:, 0] = False  # ilk token henüz görülmeden haricî uzman
        logits = torch.where(use, target_logits, external_logits)
        mask = tgt != -100
        scores = logits[mask]
        yt = tgt[mask]
        top = scores.topk(5, dim=-1).indices
        hit1 += int((top[:, 0] == yt).sum())
        hit5 += int((top == yt[:, None]).any(-1).sum())
        n += len(yt)
        nll += float(F.cross_entropy(scores, yt, reduction="sum"))
    return {"n": n, "top1": hit1 / max(1, n), "top5": hit5 / max(1, n),
            "ppl": float(np.exp(nll / max(1, n))), "target_sentences": routed_target}


def main():
    torch.set_num_threads(1)
    rows = SENTENCES + DIALOGUE
    held = rows[::5]
    target_train = [x for i, x in enumerate(rows) if i % 5]
    random.Random(140).shuffle(target_train)
    external_train = []
    with open(".cache/corpora/processed/turkish_mix_train_shuffled.jsonl") as f:
        for line in f:
            text = json.loads(line)["text"]
            if text not in set(held):
                external_train.append(text)
            if len(external_train) >= 1000:
                break
    random.Random(140).shuffle(external_train)
    external_train = external_train[:len(target_train)]
    xquad = [x.strip() for x in open(".cache/corpora/processed/xquad_tr_sentences.txt") if x.strip()]
    external = BrocaOrgan.load("runs/models/external_rebuilt_best.pt")
    target = BrocaOrgan.load("runs/models/position_gated_knn.pt")
    state = state_vector(NoraAGI(seed=140, boot_thinker=False))
    target_ids = first_ids(external, target_train)
    external_ids = first_ids(external, external_train)
    train_ids = target_ids + external_ids
    labels = torch.tensor([1.] * len(target_ids) + [0.] * len(external_ids))
    raw = external.net.emb.weight[torch.tensor(train_ids)].detach()
    mean = raw.mean(0, keepdim=True)
    std = raw.std(0, keepdim=True).clamp_min(1e-3)
    features = (raw - mean) / std
    results = []
    for seed in (140, 141, 142):
        torch.manual_seed(seed)
        router = nn.Linear(features.shape[1], 1)
        optimizer = torch.optim.AdamW(router.parameters(), lr=0.01, weight_decay=0.01)
        for _ in range(300):
            loss = F.binary_cross_entropy_with_logits(router(features).squeeze(1), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_probability = torch.sigmoid(router(features)).squeeze(1)
        train_accuracy = float(((train_probability >= 0.5).float() == labels).float().mean())
        dev = routed_evaluate(router, mean, std, external, target, held, state)
        # Bağımsız dış-domain yalnız seed-140 için pahalı ölçülür; diğer tohumlar router varyansıdır.
        row = {"seed": seed, "train_loss": float(loss.detach()),
               "train_accuracy": train_accuracy, "development": dev}
        if seed == 140:
            row["xquad"] = routed_evaluate(router, mean, std, external, target, xquad, state)
        results.append(row)
        print(row, flush=True)
    with open("runs/moe_domain_router.json", "w") as f:
        json.dump(results, f, indent=2)


def prefix_ids(organ, texts, prefix=3):
    return [e.ids[:prefix] for e in organ.tok.tk.encode_batch(texts) if len(e.ids) >= 4]


def prefix_features(organ, rows, mean, std):
    raw = torch.tensor(rows)
    return ((organ.net.emb.weight[raw].detach() - mean) / std).flatten(1)


@torch.no_grad()
def prefix_probabilities(router, mean, std, external, texts, prefix=3):
    rows = prefix_ids(external, texts, prefix)
    if not rows:
        return torch.empty(0)
    return torch.sigmoid(router(prefix_features(external, rows, mean, std)).squeeze(1))


def train_prefix_router(external, target_train, external_train, prefix=3, seed=140):
    target_ids = prefix_ids(external, target_train, prefix)
    external_ids = prefix_ids(external, external_train, prefix)
    raw = torch.tensor(target_ids + external_ids)
    embedded = external.net.emb.weight[raw].detach()
    mean = embedded.mean(0, keepdim=True)
    std = embedded.std(0, keepdim=True).clamp_min(1e-3)
    features = prefix_features(external, target_ids + external_ids, mean, std)
    labels = torch.tensor([1.] * len(target_ids) + [0.] * len(external_ids))
    torch.manual_seed(seed)
    router = nn.Linear(features.shape[1], 1)
    optimizer = torch.optim.AdamW(router.parameters(), lr=0.01, weight_decay=0.01)
    for _ in range(300):
        loss = F.binary_cross_entropy_with_logits(router(features).squeeze(1), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_accuracy = float(((torch.sigmoid(router(features).squeeze(1)) >= 0.5).float()
                            == labels).float().mean())
    return router, mean, std, float(loss.detach()), train_accuracy


@torch.no_grad()
def token_confidence_evaluate(router, mean, std, external, position, wide, texts, state,
                              ood_threshold, token_ratio=0.75, prefix=3, position_weight=0.75,
                              gate="embedding", nll_threshold=0.0, batch_size=64):
    if gate not in {"embedding", "nll_prefix"}:
        raise ValueError("gate embedding veya nll_prefix olmalı")
    for organ in (external, position, wide):
        organ.net.eval()
    hit1 = hit5 = n = target_n = ood_sentences = sentence_n = 0
    nll = 0.0
    batch_size = max(1, int(batch_size))
    for start in range(0, len(texts), batch_size):
        enc = external.tok.tk.encode_batch(texts[start:start + batch_size])
        rows = [e.ids[:external.MAX_LEN] for e in enc if len(e.ids) >= 4]
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
        # Aynı satırın ilk parçalarını doğrudan kullan; metin decode/encode dönüşümü yok.
        probability = torch.sigmoid(router(prefix_features(
            external, [r[:prefix] for r in rows], mean, std)).squeeze(1))
        sentence_external = probability < float(ood_threshold)
        bag = torch.zeros(batch, external.net.V)
        st = torch.from_numpy(np.tile(state.astype(np.float32), (batch, 1)))
        probabilities, prefix_probabilities = [], []
        mask = tgt != -100
        for organ in (external, position, wide):
            logits, _ = organ.net(inp, organ.net.encode_cond(st, bag))
            probabilities.append(torch.softmax(logits[mask].float(), dim=-1))
            if gate == "nll_prefix":
                prefix_probabilities.append(torch.softmax(logits[:, :prefix].float(), dim=-1))
        if gate == "nll_prefix":
            prefix_target = tgt[:, :prefix].clamp_min(0)
            prefix_mask = tgt[:, :prefix] != -100

            def prefix_nll(probs):
                values = -probs.clamp_min(1e-30).log().gather(
                    -1, prefix_target[..., None]).squeeze(-1)
                return (values * prefix_mask).sum(1)

            target_prefix_p = (float(position_weight) * prefix_probabilities[1]
                               + (1.0 - float(position_weight)) * prefix_probabilities[2])
            prefix_score = prefix_nll(target_prefix_p) - prefix_nll(prefix_probabilities[0])
            sentence_external = prefix_score > float(nll_threshold)
        ood_sentences += int(sentence_external.sum())
        sentence_n += batch
        external_p = probabilities[0]
        target_p = float(position_weight) * probabilities[1] + (1.0 - float(position_weight)) * probabilities[2]
        use_target = (target_p.max(dim=-1).values
                      >= float(token_ratio) * external_p.max(dim=-1).values)
        # İlk üç hedef parçayı görmek, aynı parçaları tahmin ederken gelecek bilgisi
        # sızdırmasın: OOD reddi yalnız üç gerçek parça gözlendikten sonra etkinleşir.
        observed = torch.arange(length)[None, :] >= int(prefix)
        sentence_external_tokens = (sentence_external[:, None] & observed).expand(batch, length)[mask]
        use_target = use_target & ~sentence_external_tokens
        scores = torch.where(use_target[:, None], target_p, external_p) + 1e-30
        yt = tgt[mask]
        top = scores.topk(5, dim=-1).indices
        hit1 += int((top[:, 0] == yt).sum())
        hit5 += int((top == yt[:, None]).any(-1).sum())
        target_n += int(use_target.sum())
        n += len(yt)
        nll -= float(scores.log().gather(1, yt[:, None]).sum())
    return {"n": n, "top1": hit1 / max(1, n), "top5": hit5 / max(1, n),
            "ppl": float(np.exp(nll / max(1, n))),
            "target_fraction": target_n / max(1, n),
            "ood_sentences": ood_sentences, "sentences": sentence_n,
            "ood_threshold": float(ood_threshold), "token_confidence_ratio": float(token_ratio),
            "gate": gate, "nll_threshold": float(nll_threshold),
            "causal_ood_after_observed_tokens": int(prefix)}


def token_confidence_main():
    """OOD cümle reddi + token güven fallback'i; XQuAD eşik kurulumuna girmez."""
    torch.set_num_threads(1)
    seed = int(os.environ.get("NORA_ROUTER_SEED", "140"))
    output = os.environ.get("NORA_ROUTER_OUTPUT", "runs/moe_token_confidence_router.json")
    model_output = os.environ.get("NORA_ROUTER_MODEL_OUTPUT",
                                  "runs/models/moe_token_confidence_router.pt")
    token_ratio = float(os.environ.get("NORA_TOKEN_RATIO", "0.75"))
    position_weight = float(os.environ.get("NORA_POSITION_WEIGHT", "0.75"))
    gate = os.environ.get("NORA_ROUTER_GATE", "embedding")
    nll_threshold = float(os.environ.get("NORA_NLL_THRESHOLD", "0"))
    rows = SENTENCES + DIALOGUE
    held = rows[::5]
    target_train = [x for i, x in enumerate(rows) if i % 5]
    random.Random(140).shuffle(target_train)
    external_train = []
    held_set = set(held)
    external_corpus = os.environ.get(
        "NORA_EXTERNAL_CORPUS", ".cache/corpora/processed/turkish_mix_train_shuffled.jsonl")
    with open(external_corpus) as handle:
        for line in handle:
            text = json.loads(line)["text"]
            if text not in held_set:
                external_train.append(text)
            if len(external_train) >= 1000:
                break
    random.Random(140).shuffle(external_train)
    external_train = external_train[:len(target_train)]
    external = BrocaOrgan.load(os.environ.get(
        "NORA_EXTERNAL_MODEL", "runs/models/external_rebuilt_best.pt"))
    position = BrocaOrgan.load(os.environ.get(
        "NORA_POSITION_MODEL", "runs/models/position_gated_knn.pt"))
    wide = BrocaOrgan.load(os.environ.get(
        "NORA_WIDE_MODEL", "runs/models/wide143_best.pt"))
    router, mean, std, train_loss, train_accuracy = train_prefix_router(
        external, target_train, external_train, prefix=3, seed=seed)
    held_probability = prefix_probabilities(router, mean, std, external, held, prefix=3)
    # Development cümlelerinin tamamı konservatif OOD reddinin içinde kalsın.
    # Etiket/top-1 bu hesapta yok; bağımsız XQuAD yalnız aşağıda ölçülür.
    ood_threshold = 0.5 * float(held_probability.min())
    state = state_vector(NoraAGI(seed=140, boot_thinker=False))
    dev = token_confidence_evaluate(router, mean, std, external, position, wide,
                                    held, state, ood_threshold, token_ratio=token_ratio,
                                    position_weight=position_weight, gate=gate,
                                    nll_threshold=nll_threshold)
    xquad_path = os.environ.get(
        "NORA_XQUAD_CORPUS", ".cache/corpora/processed/xquad_tr_sentences.txt")
    if xquad_path.endswith(".jsonl"):
        xquad = [json.loads(line)["text"] for line in open(xquad_path) if line.strip()]
    else:
        xquad = [x.strip() for x in open(xquad_path) if x.strip()]
    confirmation = token_confidence_evaluate(router, mean, std, external, position, wide,
                                             xquad, state, ood_threshold, token_ratio=token_ratio,
                                             position_weight=position_weight, gate=gate,
                                             nll_threshold=nll_threshold)
    result = {"seed": seed, "prefix": 3, "gate": gate, "nll_threshold": nll_threshold,
              "router_train_loss": train_loss,
              "router_train_accuracy": train_accuracy,
              "development": dev, "xquad": confirmation,
              "target_ensemble": {"position": position_weight, "wide": 1.0 - position_weight},
              "model_paths": {
                  "external": os.environ.get("NORA_EXTERNAL_MODEL", "runs/models/external_rebuilt_best.pt"),
                  "position": os.environ.get("NORA_POSITION_MODEL", "runs/models/position_gated_knn.pt"),
                  "wide": os.environ.get("NORA_WIDE_MODEL", "runs/models/wide143_best.pt"),
              },
              "corpora": {"external_router_train": external_corpus, "xquad": xquad_path}}
    with open(output, "w") as handle:
        json.dump(result, handle, indent=2)
    torch.save({"router": router.state_dict(), "mean": mean, "std": std,
                "ood_threshold": ood_threshold, "prefix": 3, "token_ratio": token_ratio,
                "position_weight": position_weight, "gate": gate, "nll_threshold": nll_threshold},
               model_output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    if os.environ.get("NORA_ROUTER_MODE") == "token":
        token_confidence_main()
    else:
        main()

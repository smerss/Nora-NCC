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
import tokenizers
from torch import nn

from nora_agi import NoraAGI
from nora_besle import DIALOGUE, SENTENCES
from nora_broca_net import BrocaOrgan
from nora_voice import state_vector
from runs.turkish_confirmation_eval import probability_evaluate


def train(kind, seed, epochs, train, state, base_path="data/broca_tr.pt"):
    organ = BrocaOrgan.load(base_path)
    if kind == "wide":
        organ.net.pos_emb = None
        organ.expand_hidden(1024, seed=seed)
    elif organ.net.pos_emb is None:
        organ.net.pos_emb = nn.Embedding(65, organ.net.emb.embedding_dim)
        nn.init.zeros_(organ.net.pos_emb.weight)
    adapter = os.environ.get("NORA_TARGET_ADAPTER", "none")
    if adapter not in {"none", "temporal", "temporal_full"}:
        raise ValueError("NORA_TARGET_ADAPTER none/temporal/temporal_full olmalı")
    if adapter in {"temporal", "temporal_full"}:
        organ.add_temporal_layer(seed=seed)
        if adapter == "temporal":
            for name, parameter in organ.net.named_parameters():
                parameter.requires_grad_(name.startswith("temporal.") or
                                         name.startswith("temporal_proj.") or
                                         (kind == "position" and name.startswith("pos_emb.")))
    if bool(int(os.environ.get("NORA_TARGET_UNTIED_OUTPUT", "0"))):
        organ.add_untied_output()
    if bool(int(os.environ.get("NORA_TARGET_OUTPUT_BIAS", "0"))):
        organ.add_output_bias()
    dropout = float(os.environ.get("NORA_TARGET_DROPOUT", "0.1"))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("NORA_TARGET_DROPOUT [0,1) aralığında olmalı")
    organ.net.drop.p = dropout
    if bool(int(os.environ.get("NORA_FREEZE_EMB", "0"))):
        organ.net.emb.weight.requires_grad_(False)
    if bool(int(os.environ.get("NORA_FREEZE_COND", "0"))):
        for parameter in organ.net.cond.parameters():
            parameter.requires_grad_(False)
        organ.net.cond_in.weight.requires_grad_(False)
    organ.opt = torch.optim.Adam(organ.net.parameters(), lr=organ.lr)
    torch.manual_seed(seed)
    organ.rng = np.random.default_rng(90000 + seed)
    bpe_dropout = float(os.environ.get("NORA_TARGET_BPE_DROPOUT", "0.1"))
    deterministic_bpe = bool(int(os.environ.get("NORA_TARGET_DETERMINISTIC_BPE", "0")))
    learning_rate = float(os.environ.get("NORA_TARGET_LR", "0.0005"))
    focal_gamma = float(os.environ.get("NORA_TARGET_FOCAL_GAMMA", "0"))
    early_token_weight = float(os.environ.get("NORA_TARGET_EARLY_WEIGHT", "1"))
    early_token_count = int(os.environ.get("NORA_TARGET_EARLY_COUNT", "0"))
    frequency_alpha = float(os.environ.get("NORA_TARGET_FREQUENCY_ALPHA", "0"))
    frequency_cap = float(os.environ.get("NORA_TARGET_FREQUENCY_CAP", "8"))
    word_start_weight = float(os.environ.get("NORA_TARGET_WORD_START_WEIGHT", "1"))
    l2sp_strength = float(os.environ.get("NORA_TARGET_L2SP", "0"))
    parameter_anchor = None
    if l2sp_strength:
        parameter_anchor = {name: parameter.detach().clone()
                            for name, parameter in organ.net.named_parameters()}
    bag_fraction = float(os.environ.get("NORA_TARGET_BAG_FRACTION", "1"))
    if not 0.0 < bag_fraction <= 1.0:
        raise ValueError("NORA_TARGET_BAG_FRACTION (0,1] aralığında olmalı")
    if bag_fraction < 1.0:
        bag_size = max(32, int(round(len(train) * bag_fraction)))
        train = random.Random(seed + 313).sample(list(train), min(len(train), bag_size))
    joint_steps = int(os.environ.get("NORA_JOINT_STEPS", "0"))
    joint_target_per_batch = int(os.environ.get("NORA_JOINT_TARGET_PER_BATCH", "4"))
    if joint_steps < 0 or not 1 <= joint_target_per_batch <= 31:
        raise ValueError("NORA_JOINT_STEPS >=0 ve NORA_JOINT_TARGET_PER_BATCH 1..31 olmalı")
    if joint_steps:
        external_path = os.environ.get("NORA_JOINT_EXTERNAL", ".cache/corpora/processed/external_train.jsonl")
        external_handle = open(external_path)
        forbidden = set(SENTENCES + DIALOGUE)
        joint_source = os.environ.get("NORA_JOINT_SOURCE", "")
        target_order = list(train)
        target_cursor = 0
        target_cycle = 0
        joint_skipped = 0
        joint_lr = float(os.environ.get("NORA_JOINT_LR", str(learning_rate)))
        try:
            for step in range(joint_steps):
                external_batch = []
                while len(external_batch) < 32 - joint_target_per_batch:
                    line = external_handle.readline()
                    if not line:
                        raise RuntimeError("joint haricî akış adımlar tamamlanmadan tükendi")
                    row = json.loads(line)
                    if joint_source and row.get("source") != joint_source:
                        continue
                    text = row["text"]
                    if text in forbidden:
                        joint_skipped += 1
                        continue
                    external_batch.append(text)
                if target_cursor + joint_target_per_batch > len(target_order):
                    target_cycle += 1
                    target_order = list(train)
                    random.Random(seed + 1009 * target_cycle).shuffle(target_order)
                    target_cursor = 0
                target_batch = target_order[target_cursor:target_cursor + joint_target_per_batch]
                target_cursor += joint_target_per_batch
                batch = external_batch + target_batch
                random.Random(seed + step).shuffle(batch)
                bpe_seed = seed * 1_000_003 + step if deterministic_bpe else None
                organ.learn_batch(batch, state, lr=joint_lr, anchor_dropout=1.0,
                                  loss_mode="full", bpe_dropout=bpe_dropout, focal_gamma=focal_gamma,
                                  parameter_anchor=parameter_anchor,
                                  parameter_anchor_strength=l2sp_strength,
                                  bpe_dropout_seed=bpe_seed,
                                  early_token_weight=early_token_weight,
                                  early_token_count=early_token_count,
                                  frequency_alpha=frequency_alpha,
                                  frequency_cap=frequency_cap,
                                  word_start_weight=word_start_weight)
        finally:
            external_handle.close()
        organ.joint_stats = {"steps": joint_steps, "target_per_batch": joint_target_per_batch,
                             "external_per_batch": 32 - joint_target_per_batch,
                             "external_path": external_path, "source": joint_source or "all",
                             "target_cycles": target_cycle, "held_exact_skipped": joint_skipped}
    else:
        for epoch in range(epochs):
            for i in range(0, len(train), 32):
                bpe_seed = seed * 1_000_003 + epoch * 100_000 + i // 32 if deterministic_bpe else None
                organ.learn_batch(train[i:i + 32], state, lr=learning_rate, anchor_dropout=1.0,
                                  loss_mode="full", bpe_dropout=bpe_dropout, focal_gamma=focal_gamma,
                                  parameter_anchor=parameter_anchor,
                                  parameter_anchor_strength=l2sp_strength,
                                  bpe_dropout_seed=bpe_seed,
                                  early_token_weight=early_token_weight,
                                  early_token_count=early_token_count,
                                  frequency_alpha=frequency_alpha,
                                  frequency_cap=frequency_cap,
                                  word_start_weight=word_start_weight)
    organ.target_train_size = len(train)
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
    base_path = os.environ.get("NORA_BASE_MODEL", "data/broca_tr.pt")
    model_prefix = os.environ.get("NORA_MODEL_PREFIX", "crossfold")
    results = []
    for fold in [int(x) for x in os.environ.get("NORA_CROSSFOLDS", "1,2").split(",")]:
        if not 0 <= fold < 5 or (fold == 0 and not bool(int(os.environ.get("NORA_ALLOW_DEVELOPMENT_FOLD0", "0")))):
            raise ValueError("fold 0 yalnız açık development bayrağıyla; confirmation fold 1..4 olmalı")
        held = rows[fold::5]
        train_rows = [x for i, x in enumerate(rows) if i % 5 != fold]
        target_domain = os.environ.get("NORA_TARGET_DOMAIN", "all")
        if target_domain not in {"all", "facts", "dialogue"}:
            raise ValueError("NORA_TARGET_DOMAIN all/facts/dialogue olmalı")
        if target_domain != "all":
            allowed = set(SENTENCES if target_domain == "facts" else DIALOGUE)
            train_rows = [x for x in train_rows if x in allowed]
        started = time.monotonic()
        fold_result = {"fold": fold, "position_seed": position_seed, "wide_seed": wide_seed,
                       "position_epochs_fixed": position_epochs, "wide_epochs_fixed": wide_epochs,
                       "position_weight": weight, "base_model": base_path,
                       "model_prefix": model_prefix,
                       "target_adapter": os.environ.get("NORA_TARGET_ADAPTER", "none"),
                       "target_output_bias": bool(int(os.environ.get("NORA_TARGET_OUTPUT_BIAS", "0"))),
                       "target_untied_output": bool(int(os.environ.get("NORA_TARGET_UNTIED_OUTPUT", "0"))),
                       "freeze_emb": bool(int(os.environ.get("NORA_FREEZE_EMB", "0"))),
                       "freeze_cond": bool(int(os.environ.get("NORA_FREEZE_COND", "0"))),
                       "target_bpe_dropout": float(os.environ.get("NORA_TARGET_BPE_DROPOUT", "0.1")),
                       "target_bpe_dropout_deterministic": (not bool(float(os.environ.get("NORA_TARGET_BPE_DROPOUT", "0.1")))) or bool(int(os.environ.get("NORA_TARGET_DETERMINISTIC_BPE", "0"))),
                       "target_seeded_bpe": bool(int(os.environ.get("NORA_TARGET_DETERMINISTIC_BPE", "0"))),
                       "runtime": {"torch": torch.__version__, "tokenizers": tokenizers.__version__},
                       "target_dropout": float(os.environ.get("NORA_TARGET_DROPOUT", "0.1")),
                       "target_lr": float(os.environ.get("NORA_TARGET_LR", "0.0005")),
                       "target_focal_gamma": float(os.environ.get("NORA_TARGET_FOCAL_GAMMA", "0")),
                       "target_early_weight": float(os.environ.get("NORA_TARGET_EARLY_WEIGHT", "1")),
                       "target_early_count": int(os.environ.get("NORA_TARGET_EARLY_COUNT", "0")),
                       "target_frequency_alpha": float(os.environ.get("NORA_TARGET_FREQUENCY_ALPHA", "0")),
                       "target_frequency_cap": float(os.environ.get("NORA_TARGET_FREQUENCY_CAP", "8")),
                       "target_word_start_weight": float(os.environ.get("NORA_TARGET_WORD_START_WEIGHT", "1")),
                       "dialogue_repeats": int(os.environ.get("NORA_DIALOGUE_REPEATS", "1")),
                       "target_domain": target_domain,
                       "target_bag_fraction": float(os.environ.get("NORA_TARGET_BAG_FRACTION", "1")),
                       "target_l2sp": float(os.environ.get("NORA_TARGET_L2SP", "0"))}
        organs = []
        for kind, seed, epochs in [("position", position_seed, position_epochs),
                                   ("wide", wide_seed, wide_epochs)]:
            shuffled = list(train_rows)
            dialogue_repeats = int(os.environ.get("NORA_DIALOGUE_REPEATS", "1"))
            if dialogue_repeats < 1:
                raise ValueError("NORA_DIALOGUE_REPEATS en az 1 olmalı")
            if dialogue_repeats > 1:
                dialogue = set(DIALOGUE)
                shuffled += [x for x in train_rows if x in dialogue] * (dialogue_repeats - 1)
            random.Random(seed).shuffle(shuffled)
            organ = train(kind, seed, epochs, shuffled, state, base_path=base_path)
            fold_result[f"{kind}_train_sentences"] = organ.target_train_size
            if hasattr(organ, "joint_stats"):
                fold_result[f"{kind}_joint"] = organ.joint_stats
            organ.save(f"runs/models/{model_prefix}_fold{fold}_{kind}.pt")
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

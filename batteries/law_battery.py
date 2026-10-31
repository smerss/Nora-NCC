#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""law_battery.py — yeni koşullu temsilin held-out ve planlama sınavı."""
from __future__ import annotations

import sys, os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


import argparse
import copy
import base64
import collections
import hashlib
import itertools
import json
import math
import os
import random
import re
import time
import tracemalloc
import urllib.parse
import urllib.request

import numpy as np

from nora_causal import CausalTransitionModel
from nora_laws import SymbolicLaw, SymbolicTransitionModel
from nora_planning import AdaptivePlanner


def oracle(x, action):
    return action if x < 0 else -action


def records(rows):
    return [{"state": [float(x)], "action": float(a),
             "next": [float(x + oracle(x, a))]} for x, a in rows]


TRAIN = [(-3, .5), (-2, 1.), (-1, 2.), (0, .5), (1, 1.), (2, 2.)]
VALID = [(-4, 1.5), (3, .5)]
AUDIT = [(-5, 2.5), (4, 1.5)]


def conditional_expression() -> tuple[bool, str]:
    fn = lambda rows: [{"args": [x, a], "expect": oracle(x, a)} for x, a in rows]
    law = SymbolicLaw(["f0", "action"], max_depth=3, max_exprs=12000)
    result = law.fit(fn(TRAIN), fn(VALID), fn(AUDIT))
    fresh = [(-6, 3.), (-2.5, 1.25), (0., 2.), (5., 4.)]
    ok = (result["status"] == "accepted" and result.get("form") == "conditional"
          and all(abs(float(law.predict([x, a])) - oracle(x, a)) < 1e-8 for x, a in fresh))
    return ok, f"status={result['status']} form={result.get('form')} expr={result.get('expression')}"


def learned_law_plans() -> tuple[bool, str]:
    model = SymbolicTransitionModel(["f0"], max_depth=3, max_exprs=12000)
    fit = model.fit(records(TRAIN), records(VALID), records(AUDIT))
    if fit["status"] != "accepted":
        return False, f"fit={fit}"
    planner = AdaptivePlanner(model, beam=8, max_horizon=5)
    result = planner.plan([-3.0], [1.0], {"f0": {"target": 0.0, "tol": 1e-8}})
    # Bağımsız dünya: modelin kendi states listesini başarı kanıtı sayma.
    actual = -3.0
    for action in result.get("actions", []):
        actual += oracle(actual, action)
    restored = SymbolicTransitionModel(["f0"])
    restored.load_state_dict(model.state_dict())
    same = restored.predict([-3.0], 1.0) == model.predict([-3.0], 1.0)
    ok = result["status"] == "ok" and abs(actual) < 1e-8 and same
    return ok, f"fit={fit['status']} path={result.get('actions')} independent_state={actual} checkpoint={same}"


def linear_ablation() -> tuple[bool, str]:
    linear = CausalTransitionModel(["f0"])
    for row in records(TRAIN):
        linear.observe(row["state"], {"name": "a", "features": [row["action"]]},
                       row["next"], intervened=True)
    symbolic = SymbolicTransitionModel(["f0"], max_depth=3, max_exprs=12000)
    fit = symbolic.fit(records(TRAIN), records(VALID), records(AUDIT))
    if fit["status"] != "accepted":
        return False, "symbolic model kabul edilmedi"
    lin_err, sym_err = [], []
    for x, a in VALID + AUDIT + [(-6, 3.), (5, 4.)]:
        expected = oracle(x, a)
        lp = linear.predict([x], {"name": "a", "features": [a]})
        sp = symbolic.predict([x], a)
        lin_err.append(abs((lp[0]["delta"] if lp else 0.0) - expected))
        sym_err.append(abs((sp[0]["delta"] if sp else 0.0) - expected))
    ok = float(np.mean(sym_err)) < float(np.mean(lin_err))
    return ok, f"held-out MAE symbolic={np.mean(sym_err):.3f} linear={np.mean(lin_err):.3f}"


def audit_rejects_and_atomic_fit() -> tuple[bool, str]:
    law = SymbolicLaw(["f0", "action"], max_depth=3, max_exprs=6000)
    fn = lambda rows: [{"args": [x, a], "expect": oracle(x, a)} for x, a in rows]
    bad = law.fit(fn(TRAIN), fn(VALID), [{"args": [9., 1.], "expect": 999.}])
    unresolved = bad["status"] == "unresolved" and law.expression is None
    model = SymbolicTransitionModel(["f0"], max_depth=3, max_exprs=12000)
    good = model.fit(records(TRAIN), records(VALID), records(AUDIT))
    before = model.state_dict()
    # Aynı modelin ikinci fit'i bozuk audit ile eski kabulü silmemeli.
    bad2 = model.fit(records(TRAIN), records(VALID), [{"state": [9.], "action": 1., "next": [999.]}])
    atomic = good["status"] == "accepted" and bad2["status"] == "unresolved" and model.state_dict() == before
    return unresolved and atomic, f"bad-law={bad['status']} atomic={atomic}"


def nora_induce_plan_checkpoint() -> tuple[bool, str]:
    import os
    import tempfile
    from nora_agi import NoraAGI
    agi = NoraAGI(seed=9192, boot_thinker=False)
    fit = agi.induce_transition_law(["f0"], train=records(TRAIN),
                                    validation=records(VALID), audit=records(AUDIT))
    first = agi.plan_actions([-3.0], [1.0], {"f0": 0.0}, use_symbolic=True)
    prefix = tempfile.mktemp(prefix="nora_law_live_", dir="/tmp")
    try:
        agi.save(prefix)
        restored = NoraAGI.load(prefix)
        second = restored.plan_actions([-3.0], [1.0], {"f0": 0.0}, use_symbolic=True)
        ok = (fit["status"] == "accepted" and first["status"] == "ok" and
              second["status"] == "ok" and first["actions"] == second["actions"])
        return ok, f"fit={fit['status']} plan={first['status']} loaded={second['status']} path={second.get('actions')}"
    finally:
        for ext in (".agi.json", ".agi_mb.npz", ".mind.npz", ".voice.pkl.gz", ".broca.pt"):
            try:
                os.remove(prefix + ext)
            except FileNotFoundError:
                pass


def latent_representation_opens_only_when_needed() -> tuple[bool, str]:
    """Eski akış ifadeleri yetmezse gizli durumlu temsil aç; held-out'ta sına."""
    from nora_semantic_code import CompositionalCodeLearner

    def oracle(text):
        parity = 0
        output = []
        for character in text:
            if character == "a":
                parity = 1 - parity
            output.append(str(parity))
        return "".join(output)

    def pairs(texts):
        return [(text, oracle(text)) for text in texts]

    learner = CompositionalCodeLearner()
    learner.load_primer("data/python_semantics_primer.md")
    before = learner.representation_kinds()
    train = pairs(["abca", "cba", "bac", "cca", "abcabc", "cbacba"])
    validation = pairs(["acbb", "ccaa", "baca"])
    result = learner.invent_representation(train, validation)
    heldout = ["cabca", "bbaca", "ccabca"]
    outputs = [learner.apply_representation(result.get("id"), text) for text in heldout]
    expected = [oracle(text) for text in heldout]
    saved = learner.state_dict()
    if result.get("id") in learner.representations:
        learner.representations[result["id"]]["transitions"] = []
    ablated = learner.apply_representation(result.get("id"), heldout[0])
    learner.load_state_dict(saved)
    restored = learner.apply_representation(result.get("id"), heldout[0])

    control = CompositionalCodeLearner()
    control.load_primer("data/python_semantics_primer.md")
    existing = control.invent_representation(
        [("ab", "ab"), ("xy", "xy")], [("pq", "pq")],
    )
    # Modulo-6 regresyonu: iki minimal çözüm exhaustive sayılmalı, yalnız
    # recurrent aday açık tercihle seçilmeli ve held-out tam geçmelidir.
    hard_task = _counter_task(6)
    hard_learner = _prior_learner()
    hard = hard_learner.invent_representation(
        hard_task["train"], hard_task["validation"], max_states=6,
        max_expansions=SEARCH_EXPANSION_LIMIT,
    )
    hard_name = str(hard.get("id") or "")
    hard_correct = sum(
        hard_learner.apply_representation(hard_name, source) == expected
        for source, expected in hard_task["heldout"]
    ) if hard_name else 0
    hard_ok = (hard.get("status") == "invented"
               and hard.get("selection") == "unique-minimal-recurrent"
               and hard.get("candidates") == 2
               and hard.get("recurrent_candidates") == 1
               and hard_correct == len(hard_task["heldout"]))
    ok = (result.get("status") == "invented"
          and result.get("kind") not in before and result.get("states", 0) > 1
          and outputs == expected and ablated is None and restored == expected[0]
          and existing.get("status") == "existing" and not control.representations
          and hard_ok)
    detail = (f"durum={result.get('status')} tür={result.get('kind')} durum-sayısı={result.get('states')} | "
              f"held-out={outputs} beklenen={expected} | ablasyon={ablated} restore={restored} | "
              f"eski-temsil-kontrolü={existing.get('status')} | "
              f"mod6={hard_correct}/{len(hard_task['heldout'])}:{hard.get('selection')}")
    return ok, detail


def learned_representation_personal_application() -> tuple[bool, str]:
    """Kalıcı temsil doğal istekle yeni girdiye uygulanır ve gerçekten çalışır."""
    from nora_code import CodeSynthesisCortex

    def oracle(text):
        parity = 0
        output = []
        for character in text:
            if character == "a":
                parity = 1 - parity
            output.append(str(parity))
        return "".join(output)

    pairs = lambda texts: [(text, oracle(text)) for text in texts]
    cortex = CodeSynthesisCortex(seed=731, pretrain=False)
    cortex.semantic.operators.update({"bind", "iterate", "concatenate"})
    learned = cortex.semantic.invent_representation(
        pairs(["abca", "cba", "bac", "cca", "abcabc", "cbacba"]),
        pairs(["acbb", "ccaa", "baca"]), max_states=2,
    )
    state = json.loads(json.dumps(cortex.semantic.state_dict()))
    restored = CodeSynthesisCortex(seed=732, pretrain=False)
    restored.semantic.load_state_dict(state)
    source = "cabca"
    request = f"{source} metnini öğrendiğin işlemle dönüştürüp çıktısını yazdır"
    result = restored.handle(request)
    unknown = restored.handle("cabc2 metnini öğrendiğin işlemle dönüştürüp çıktısını yazdır")
    expected = oracle(source) + "\n"
    # Genel operatör-sentez kanalı, kalıcı temsilin "uygula" parafrazını
    # kod kanalından önce kapmamalı. Bu ayrı AGI ve ayrı state ile sınanır.
    from nora_agi import NoraAGI
    agi = NoraAGI(seed=733, boot_thinker=False)
    agi.code_cortex.semantic.load_state_dict(state)
    agi.converse(f'"{source}" için aynı öğrendiğin işlemi uygulayıp çıktıyı yazdır.')
    agi_code = dict(agi.last_code or {})
    anaphoric_ok = bool(agi_code.get("ok") and agi_code.get("verified")
                        and agi_code.get("stdout") == expected)
    # Alıntı ve nominal işaret olmadan, iki nokta sonrası payload da aynı genel
    # kompozisyon kanalına girmeli. Beklenenler yukarıdaki bağımsız oracle'dan
    # gelir; isteklerin hazır stdout veya görev-özel şablonu yoktur.
    freeform_rows = []
    for free_source, free_request in (
        ("ccaba", "Daha önce kendi bulduğun işlemi şu yeni dizgiye aynen uygula: ccaba"),
        ("babcca", "Aynı öğrendiğin işlemi bu diziye de uygulayıp sonucu yazar mısın: babcca"),
    ):
        agi.converse(free_request)
        free_code = dict(agi.last_code or {})
        freeform_rows.append(bool(
            free_code.get("ok") and free_code.get("verified")
            and free_code.get("path") == "composed"
            and free_code.get("stdout") == oracle(free_source) + "\n"
        ))
    agi.converse("Aynı öğrendiğin işlemi bu diziye de uygulayıp sonucu yazar mısın: cab2a")
    free_unknown = dict(agi.last_code or {})
    control_goal = restored.semantic.request("Sonucu yazar mısın: ccaba")
    freeform_ok = all(freeform_rows)
    freeform_unknown_ok = (not free_unknown.get("ok")
                           and free_unknown.get("reason") == "representation_input_gap")
    colon_control_ok = getattr(control_goal, "operation", None) != "representation"
    ok = (learned.get("status") == "invented" and result.get("ok")
          and result.get("verified") and result.get("stdout") == expected
          and not unknown.get("ok") and unknown.get("reason") == "representation_input_gap"
          and anaphoric_ok and freeform_ok and freeform_unknown_ok and colon_control_ok)
    detail = (f"öğrenme={learned.get('status')} yol={result.get('path')} "
              f"stdout={result.get('stdout')!r} beklenen={expected!r} "
              f"doğrulama={result.get('verified')} alfabe-dışı={unknown.get('reason')} "
              f"AGI-anaför={anaphoric_ok} serbest={freeform_rows} "
              f"serbest-alfabe-dışı={free_unknown.get('reason')} "
              f"anaförsüz-kontrol={colon_control_ok}")
    return ok, detail


MISSION_PROMPT = (
    "Her örnekte soldaki ikili diziyi sağdaki ikili diziye dönüştüren tek bir "
    "deterministik operatör bul. Yalnız train ve validation çiftlerini kullan; "
    "bilmiyorsan bunu söyle ve genel bilgi araştır. Operatörü hiç görülmemiş "
    "girdilere genelle. Çözüm kuralı, kaynak program ve arama adresi verilmiyor."
)
MISSION_MODULUS = 5
MISSION_SEED = 20261028
SEARCH_EXPANSION_LIMIT = 100000
AUTONOMOUS_STATE_CAP = 8  # Görev modulusu değil, bütün görevlerde genel kaynak tavanı.


def _counter_oracle(source: str, modulus: int) -> str:
    """Deney sahibinin bağımsız oracle'ı; Nora'ya fonksiyon olarak verilmez."""
    state = 0
    output = []
    for symbol in source:
        state = (state + int(symbol)) % int(modulus)
        output.append(str(int(state == modulus - 1)))
    return "".join(output)


def _bits(length: int) -> list[str]:
    return ["".join(row) for row in itertools.product("01", repeat=int(length))]


def _counter_task(modulus: int, seed: int = MISSION_SEED) -> dict:
    train_sources = _bits(modulus + 1)
    validation_sources = _bits(modulus + 2) + _bits(modulus + 3)
    heldout_sources = _bits(modulus + 4)
    rng = random.Random(seed + modulus)
    for _ in range(64):
        length = rng.randint(modulus + 5, modulus + 12)
        heldout_sources.append("".join(rng.choice("01") for _ in range(length)))
    pairs = lambda sources: [(source, _counter_oracle(source, modulus)) for source in sources]
    return {
        "modulus": int(modulus), "train": pairs(train_sources),
        "validation": pairs(validation_sources), "heldout": pairs(heldout_sources),
    }


def _symbol_strings(alphabet: str, length: int) -> list[str]:
    return ["".join(row) for row in itertools.product(alphabet, repeat=int(length))]


def _family_oracle(source: str, modulus: int, family: str) -> str:
    """Bağımsız OOD oracle'ı; sentez yalnız string çiftlerini görür."""
    state = 0
    output = []
    for symbol in source:
        if family == "counter":
            state = (state + int(symbol)) % modulus
        elif family == "reset":
            state = (state + 1) % modulus if symbol == "a" else 0
        elif family == "affine":
            state = (state * 2 + int(symbol)) % modulus
        elif family == "ternary_reset":
            state = 0 if symbol == "c" else (state + (1 if symbol == "a" else 2)) % modulus
        elif family == "ternary_weighted":
            state = (state + (1, 2, 3)[int(symbol)]) % modulus
        else:
            raise ValueError("unknown_family")
        output.append(str(int(state == modulus - 1)))
    return "".join(output)


def _independent_family_oracle(source: str, modulus: int, family: str) -> str:
    """İkinci kanal için durum döngüsünü paylaşmayan prefix-aritmetik referansı."""
    output = []
    for end in range(1, len(source) + 1):
        prefix = source[:end]
        if family == "counter":
            state = sum(int(symbol) for symbol in prefix) % modulus
        elif family == "reset":
            state = len(prefix.rsplit("b", 1)[-1]) % modulus
        elif family == "affine":
            state = int(prefix, 2) % modulus
        elif family == "ternary_reset":
            tail = prefix.rsplit("c", 1)[-1]
            state = sum({"a": 1, "b": 2}[symbol] for symbol in tail) % modulus
        elif family == "ternary_weighted":
            state = sum(int(symbol) + 1 for symbol in prefix) % modulus
        else:
            raise ValueError("unknown_family")
        output.append(str(int(state == modulus - 1)))
    return "".join(output)


def _family_task(modulus: int, family: str, seed: int = MISSION_SEED) -> dict:
    """Dondurulmuş binary ve bounded-ternary veri protokolü."""
    if family == "counter":
        task = _counter_task(modulus, seed)
        task["family"] = family
        return task
    if family in {"reset", "affine"}:
        alphabet = "ab" if family == "reset" else "01"
        train = _bits(modulus + 1)
        validation = _bits(modulus + 2) + _bits(modulus + 3)
        heldout = _bits(modulus + 4)
        if family == "reset":
            convert = lambda rows: ["".join("ab"[int(x)] for x in source) for source in rows]
            train, validation, heldout = convert(train), convert(validation), convert(heldout)
    elif family in {"ternary_reset", "ternary_weighted"}:
        alphabet = "abc" if family == "ternary_reset" else "012"
        train = sum((_symbol_strings(alphabet, length) for length in range(4)), [])
        validation = _symbol_strings(alphabet, 4) + _symbol_strings(alphabet, 5)
        heldout = _symbol_strings(alphabet, 6)
    else:
        raise ValueError("unknown_family")
    rng = random.Random(seed + modulus + sum(map(ord, family)))
    for _ in range(64):
        length = rng.randint(modulus + 5, modulus + 12)
        heldout.append("".join(rng.choice(alphabet) for _ in range(length)))
    pairs = lambda sources: [(source, _family_oracle(source, modulus, family))
                             for source in sources]
    return {
        "modulus": int(modulus), "family": family,
        "train": pairs(train), "validation": pairs(validation),
        "heldout": pairs(heldout),
    }


def _rows_hash(rows) -> str:
    raw = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _prior_learner():
    """Hedefe özel belge/tablo değil, yalnız önceki genel akış ilkelleri."""
    from nora_semantic_code import CompositionalCodeLearner
    learner = CompositionalCodeLearner()
    learner.operators.update({"bind", "iterate", "concatenate"})
    return learner


ACTIVE_QUERY_BUDGET = 64
ACTIVE_POOL_SIZE = 1024
ACTIVE_RECOVERY_QUERIES = 24
ACTIVE_CANDIDATE_SAMPLE = 32


def _run_finite_candidate(candidate, source: str) -> str | None:
    """Kanonik temsil veya iç aday tablosunu bağımsız oracle'a karşı yürüt."""
    if isinstance(candidate, dict) and "transitions" in candidate:
        table = {(int(state), symbol): (int(target), output)
                 for state, symbol, target, output in candidate["transitions"]}
    else:
        table = candidate
    state = 0
    output = []
    for symbol in source:
        transition = table.get((state, symbol))
        if transition is None:
            return None
        state, emitted = transition
        output.append(emitted)
    return "".join(output)


def _active_query_domain(family: str, modulus: int, alphabet: str,
                         pool_size: int = ACTIVE_POOL_SIZE):
    """Bütün orijinal splitlerden uzunluğuyla ayrışan etiketsiz sorgu alanı."""
    if family.startswith("ternary"):
        # Dondurulmuş ternary held-out |x|=6 ve uzun rastgeleler |x|>=m+5.
        lengths = list(range(7, int(modulus) + 5))
    else:
        # Binary held-out |x|=m+4 ve uzun rastgeleler en çok m+12.
        lengths = [int(modulus) + 13, int(modulus) + 14, int(modulus) + 15]
    rng = random.Random(MISSION_SEED + 100003 * int(modulus) + sum(map(ord, family)))
    pool = set()
    for length in lengths:
        pool.add(alphabet[0] * length)
        pool.add(alphabet[-1] * length)
        pool.add("".join(alphabet[i % len(alphabet)] for i in range(length)))
        pool.add("".join(alphabet[-1 - (i % len(alphabet))] for i in range(length)))
    while len(pool) < int(pool_size):
        length = lengths[len(pool) % len(lengths)]
        pool.add("".join(rng.choice(alphabet) for _ in range(length)))
    return sorted(pool), lengths


def _active_representation(train, validation, pool, oracle, *, max_states: int,
                           alphabet: str, lengths, budget: int = ACTIVE_QUERY_BUDGET,
                           cost_budget: float | None = None, query_cost=1.0,
                           repetitions: int = 1, adaptive_voting: bool = False,
                           trusted_oracle: bool = True, secondary_oracle=None,
                           secondary_trusted_oracle: bool = False):
    """Üretim learner'ının aktif protokolünü battery tuple'ına uyarla."""
    learner = _prior_learner()
    detail = learner.learn_representation_active(
        train, validation, pool, oracle, max_states=max_states,
        alphabet=alphabet, lengths=lengths, budget=budget,
        cost_budget=cost_budget, query_cost=query_cost,
        repetitions=repetitions, adaptive_voting=adaptive_voting,
        trusted_oracle=trusted_oracle, secondary_oracle=secondary_oracle,
        secondary_trusted_oracle=secondary_trusted_oracle,
        candidate_sample=ACTIVE_CANDIDATE_SAMPLE, recovery_batch=ACTIVE_RECOVERY_QUERIES,
        max_expansions=SEARCH_EXPANSION_LIMIT,
    )
    return detail.get("representation"), detail


def _active_measure(family: str, modulus: int,
                    budget: int = ACTIVE_QUERY_BUDGET, *,
                    cost_budget: float | None = None, query_cost=1.0,
                    repetitions: int = 1, adaptive_voting: bool = False,
                    trusted_oracle: bool = True, oracle=None,
                    secondary_oracle=None, secondary_trusted_oracle: bool = False) -> dict:
    task = _family_task(modulus, family)
    alphabet = "".join(sorted({symbol for source, _expected in
                               task["train"] + task["validation"] for symbol in source}))
    pool, lengths = _active_query_domain(family, modulus, alphabet)
    original_sources = {source for split in ("train", "validation", "heldout")
                        for source, _expected in task[split]}
    if set(pool) & original_sources:
        raise AssertionError("active_pool_overlaps_frozen_splits")
    started = time.perf_counter()
    selected, detail = _active_representation(
        task["train"], task["validation"], pool,
        oracle or (lambda source: _family_oracle(source, modulus, family)),
        max_states=modulus, alphabet=alphabet, lengths=lengths, budget=budget,
        cost_budget=cost_budget, query_cost=query_cost, repetitions=repetitions,
        adaptive_voting=adaptive_voting, trusted_oracle=trusted_oracle,
        secondary_oracle=secondary_oracle,
        secondary_trusted_oracle=secondary_trusted_oracle,
    )
    # Bu noktadan önce held-out, sorgu seçicinin fonksiyon kapsamına hiç girmez.
    correct = (sum(_run_finite_candidate(selected, source) == expected
                   for source, expected in task["heldout"])
               if selected is not None else 0)
    queries = detail["queries"]
    secondary_queries = detail.get("secondary_queries", [])
    if (any(query["source"] in original_sources for query in queries + secondary_queries)
            or any(len(query["source"]) not in lengths for query in queries + secondary_queries)
            or len({query["source"] for query in queries}) != len(queries)):
        raise AssertionError("active_query_protocol_violation")
    total = len(task["heldout"])
    reason_counts = collections.Counter(query["reason"] for query in queries)
    return {
        "family": family, "modulus": int(modulus), "status": detail["stop"],
        "technical_status": detail.get("technical_stop", detail["stop"]),
        "trusted_oracle": bool(detail.get("trusted_oracle", trusted_oracle)),
        "secondary_trusted_oracle": bool(detail.get("secondary_trusted_oracle",
                                                    secondary_trusted_oracle)),
        "accepted": bool(detail.get("accepted")),
        "provisional": bool(detail.get("provisional")),
        "quarantined": bool(detail.get("quarantined")),
        "quarantined_queries": int(detail.get("quarantined_queries", 0)),
        "persistent_writes": int(detail.get("persistent_writes", 0)),
        "candidate_recovered": bool(detail.get("candidate_recovered")),
        "secondary_check_ran": bool(detail.get("secondary_check_ran")),
        "secondary_complete": bool(detail.get("secondary_complete")),
        "secondary_mismatches": int(detail.get("secondary_mismatches", 0)),
        "secondary_primary_mismatches": int(detail.get("secondary_primary_mismatches", 0)),
        "secondary_errors": int(detail.get("secondary_errors", 0)),
        "secondary_query_count": int(detail.get("secondary_query_count", 0)),
        "secondary_oracle_calls": int(detail.get("secondary_oracle_calls", 0)),
        "secondary_total_cost": float(detail.get("secondary_total_cost", 0.0)),
        "combined_oracle_calls": int(detail.get("combined_oracle_calls",
                                                detail.get("oracle_calls", 0))),
        "combined_total_cost": float(detail.get("combined_total_cost",
                                                detail.get("total_cost", 0.0))),
        "secondary_queries": secondary_queries,
        "conformance_complete": detail["stop"] == "unique-conformance-complete",
        "resolved": selected is not None, "correct": correct, "total": total,
        "accuracy": correct / total, "exact": correct == total,
        "exact_but_unverified": correct == total and not bool(detail.get("accepted")),
        "seconds": time.perf_counter() - started,
        "query_count": len(queries),
        "oracle_calls": int(detail.get("oracle_calls", len(queries))),
        "accounted_oracle_calls": sum(int(query.get("calls", 1)) for query in queries),
        "total_cost": float(detail.get("total_cost", len(queries))),
        "oracle_seconds": float(detail.get("oracle_seconds", 0.0)),
        "cost_budget": cost_budget, "repetitions": int(repetitions),
        "adaptive_voting": bool(adaptive_voting),
        "disagreement_queries": reason_counts["disagreement"],
        "conformance_queries": reason_counts["conformance"],
        "recovery_queries": reason_counts["recovery-conformance"],
        "synthesis_count": len(detail["syntheses"]),
        "synthesis_expansions": sum(int(row.get("expansions", 0))
                                    for row in detail["syntheses"]),
        "filter_count": len(detail["filters"]),
        "filter_expansions": sum(int(row["expansions"]) for row in detail["filters"]),
        "max_candidates_before_query": max(
            [int(query["candidates_before"]) for query in queries] or [0],
        ),
        "final_states": int(selected.get("states", 0)) if selected else 0,
        "conformance_probe_count": detail["conformance_probe_count"],
        "active_lengths": lengths, "pool_size": len(pool),
        "pool_disjoint_from_all_original_splits": True,
        "queries_not_in_any_prior_split": True,
        "heldout_passed_to_active_learner": False,
        "train_sha256": _rows_hash(task["train"]),
        "validation_sha256": _rows_hash(task["validation"]),
        "heldout_sha256": _rows_hash(task["heldout"]),
        "active_queries_sha256": _rows_hash([
            (query["source"], query["output"]) for query in queries
        ]),
        "queries": queries, "syntheses": detail["syntheses"], "filters": detail["filters"],
    }


def active_representation_generalization() -> dict:
    """Dondurulmuş 12 OOD görevinde ayrı aktif-sorgu ölçümü."""
    rows = [_active_measure(family, modulus)
            for modulus in (5, 7, 8)
            for family in ("reset", "affine", "ternary_reset", "ternary_weighted")]
    complete = (len(rows) == 12 and all(row["status"] for row in rows)
                and all(row["queries_not_in_any_prior_split"] for row in rows))
    return {
        "protocol": {
            "query_budget": ACTIVE_QUERY_BUDGET, "pool_size": ACTIVE_POOL_SIZE,
            "candidate_sample": ACTIVE_CANDIDATE_SAMPLE,
            "recovery_batch": ACTIVE_RECOVERY_QUERIES,
            "query_selection": "tiered max-distinct/entropy disagreement; W-method extra-state conformance",
            "heldout_passed_to_active_learner": False,
            "heldout_used_for_query_selection": False,
            "acceptance": "only unique-conformance-complete is persistent/accepted",
            "provisional_rule": "budget/cost/oracle stops may be scored held-out but are not accepted",
            "query_cost": "one cost unit per oracle call in this perfect-oracle baseline",
            "axes": {
                "observational_identifiability": "active disagreement/conformance queries",
                "synthesis_scaling": "separate --scaling measurement; active learning does not claim to solve it",
                "oracle_robustness": "separate --oracle-robustness measurement; perfect baseline says nothing about noise",
            },
            "baseline_comparison": "run representation_generalization separately; frozen rows are not rewritten",
        },
        "rows": rows,
        "findings": {
            "exact_rows": sum(row["exact"] for row in rows), "total_rows": len(rows),
            "accepted_rows": sum(row["accepted"] for row in rows),
            "provisional_exact_rows": sum(row["exact_but_unverified"] for row in rows),
            "conformance_complete_rows": sum(row["conformance_complete"] for row in rows),
            "budget_exhausted_rows": [
                f"{row['family']}-{row['modulus']}:{row['status']}" for row in rows
                if row["status"] in {"budget", "cost-budget"}
            ],
            "nonexact": [f"{row['family']}-{row['modulus']}:{row['status']}"
                         for row in rows if not row["exact"]],
            "total_queries": sum(row["query_count"] for row in rows),
            "total_oracle_calls": sum(row["oracle_calls"] for row in rows),
            "total_cost": sum(row["total_cost"] for row in rows),
            "total_disagreement_queries": sum(row["disagreement_queries"] for row in rows),
            "total_conformance_queries": sum(row["conformance_queries"] for row in rows),
            "total_recovery_queries": sum(row["recovery_queries"] for row in rows),
            "all_oracle_calls_accounted": all(
                row["oracle_calls"] == row["accounted_oracle_calls"] for row in rows
            ),
            "primary_limit": "held-out exact is not conformance proof; budget/cost stops remain provisional",
        },
        "measurement_complete": complete,
        "not_claimed": ["conformance_complete_on_budget_rows",
                        "noise_or_cost_robustness_from_this_perfect_baseline",
                        "synthesis_scaling_solved"],
    }


def active_query_protocol_audit() -> tuple[bool, str]:
    """Ambiguity ve unique-minimal yanlış genellemeyi bağımsız aktif testte yakala."""
    reset = _active_measure("reset", 5)
    weighted = _active_measure("ternary_weighted", 8)
    ok = (reset["exact"] and weighted["exact"]
          and reset["disagreement_queries"] > 0
          and any(query["reason"] == "conformance" and query["prediction_matched"] is False
                  for query in weighted["queries"]))
    detail = (f"reset5={reset['correct']}/{reset['total']} q={reset['query_count']} "
              f"weighted8={weighted['correct']}/{weighted['total']} q={weighted['query_count']} "
              f"weighted_conformance_reject={any(q['prediction_matched'] is False for q in weighted['queries'])}")
    return ok, detail


def _active_query_price(source: str, attempt: int) -> float:
    """Açık örnek fiyat: çağrı başına 1 + her kaynak karakteri için 0,01."""
    return 1.0 + 0.01 * len(source)


def _noisy_family_oracle(family: str, modulus: int, seed: int, rate: float = 0.15,
                         base_oracle=None):
    """Bağımsız hash gürültüsü; ilk attempt iki protokolde de aynı kalır."""
    attempts, stats = {}, {"calls": 0, "corrupted": 0}
    reference = base_oracle or (lambda source: _family_oracle(source, modulus, family))

    def ask(source: str) -> str:
        attempt = attempts.get(source, 0)
        attempts[source] = attempt + 1
        truth = reference(source)
        stats["calls"] += 1
        digest = hashlib.sha256(
            f"{int(seed)}|{family}|{int(modulus)}|{source}|{attempt}".encode("utf-8")
        ).digest()
        draw = int.from_bytes(digest[:8], "big") / 2**64
        if draw >= float(rate):
            return truth
        position = int.from_bytes(digest[8:16], "big") % len(source)
        stats["corrupted"] += 1
        replacement = "1" if truth[position] == "0" else "0"
        return truth[:position] + replacement + truth[position + 1:]

    return ask, stats


def _correlated_family_oracle(family: str, modulus: int, period: int = 20,
                              width: int = 3, base_oracle=None):
    """Aileye bağlı konum sapması; aynı kaynakta her çağrı aynı yanlışı verir."""
    phase = sum(map(ord, family)) % int(period)
    stats = {"calls": 0, "corrupted_calls": 0, "corrupted_labels": 0}
    reference = base_oracle or (lambda source: _family_oracle(source, modulus, family))

    def ask(source: str) -> str:
        truth = reference(source)
        emitted = []
        flipped = 0
        for position, character in enumerate(truth):
            corrupt = (position + phase) % int(period) < int(width)
            if corrupt:
                flipped += 1
                character = "1" if character == "0" else "0"
            emitted.append(character)
        stats["calls"] += 1
        stats["corrupted_calls"] += int(bool(flipped))
        stats["corrupted_labels"] += flipped
        return "".join(emitted)

    return ask, stats


def correlated_oracle_robustness() -> dict:
    """Aynı 12 OOD görevinde tekrarlar arasında tam korele aile sapması."""
    rows = []
    for modulus in (5, 7, 8):
        for family in ("reset", "affine", "ternary_reset", "ternary_weighted"):
            for protocol, adaptive in (("vote3", False), ("adaptive3", True)):
                oracle, noise = _correlated_family_oracle(family, modulus)
                row = _active_measure(
                    family, modulus, 3 * ACTIVE_QUERY_BUDGET,
                    repetitions=3, adaptive_voting=adaptive, trusted_oracle=False,
                    oracle=oracle, query_cost=_active_query_price,
                )
                wrong_answers = sum(
                    response != _family_oracle(query["source"], modulus, family)
                    for query in row["queries"] for response in query.get("responses", [])
                )
                consensus_errors = sum(
                    query.get("output") is not None
                    and query["output"] != _family_oracle(query["source"], modulus, family)
                    for query in row["queries"]
                )
                used_consensus_errors = sum(
                    query.get("output") is not None
                    and query["output"] != _family_oracle(query["source"], modulus, family)
                    and query.get("used_for_learning", False)
                    for query in row["queries"]
                )
                error_records = []
                for query_index, query in enumerate(row["queries"]):
                    if (query.get("output") is None
                            or query["output"] == _family_oracle(
                                query["source"], modulus, family)):
                        continue
                    before = query.get("candidates_before")
                    after = query.get("candidates_after")
                    error_records.append({
                        "source": query["source"], "query_index": query_index,
                        "reason": query["reason"],
                        "used_for_learning": bool(query.get("used_for_learning")),
                        "later_queries": len(row["queries"]) - query_index - 1,
                        "prediction_matched": query.get("prediction_matched"),
                        "candidates_before": before, "candidates_after": after,
                        "candidate_eliminations": (
                            int(before) - int(after)
                            if before is not None and after is not None else 0
                        ),
                        "evidence_status": query.get("evidence_status"),
                    })
                repeated_same = all(
                    len(set(query.get("responses") or [])) <= 1 for query in row["queries"]
                )
                rows.append({
                    "family": family, "modulus": modulus, "protocol": protocol,
                    "status": row["status"], "technical_status": row["technical_status"],
                    "accepted": row["accepted"],
                    "provisional": row["provisional"], "exact": row["exact"],
                    "correct": row["correct"], "total": row["total"],
                    "query_sources": row["query_count"], "oracle_calls": row["oracle_calls"],
                    "total_cost": row["total_cost"], "seconds": row["seconds"],
                    "wrong_oracle_answers": wrong_answers,
                    "reported_corrupted_calls": noise["corrupted_calls"],
                    "corrupted_labels": noise["corrupted_labels"],
                    "consensus_errors": consensus_errors,
                    "consensus_error_records": error_records,
                    "used_consensus_errors": used_consensus_errors,
                    "unused_consensus_errors": consensus_errors - used_consensus_errors,
                    "quarantined_consensus_errors": (
                        consensus_errors if not row["accepted"] else 0
                    ),
                    "accepted_consensus_errors": consensus_errors if row["accepted"] else 0,
                    "persistent_writes": row["persistent_writes"],
                    "candidate_recovered": row["candidate_recovered"],
                    "all_repetitions_identical": repeated_same,
                    "oracle_inconclusive": row["status"] == "oracle-inconclusive",
                    "active_queries_sha256": row["active_queries_sha256"],
                })
    summaries = {}
    for protocol in ("vote3", "adaptive3"):
        group = [row for row in rows if row["protocol"] == protocol]
        summaries[protocol] = {
            "rows": len(group), "exact_rows": sum(row["exact"] for row in group),
            "accepted_rows": sum(row["accepted"] for row in group),
            "accepted_nonexact_rows": sum(row["accepted"] and not row["exact"] for row in group),
            "oracle_inconclusive_rows": sum(row["oracle_inconclusive"] for row in group),
            "wrong_oracle_answers": sum(row["wrong_oracle_answers"] for row in group),
            "consensus_errors": sum(row["consensus_errors"] for row in group),
            "used_consensus_errors": sum(row["used_consensus_errors"] for row in group),
            "unused_consensus_errors": sum(row["unused_consensus_errors"] for row in group),
            "quarantined_consensus_errors": sum(
                row["quarantined_consensus_errors"] for row in group
            ),
            "accepted_consensus_errors": sum(row["accepted_consensus_errors"] for row in group),
            "persistent_writes": sum(row["persistent_writes"] for row in group),
            "recovered_candidate_rows": sum(row["candidate_recovered"] for row in group),
            "consensus_error_reasons": dict(collections.Counter(
                record["reason"] for row in group for record in row["consensus_error_records"]
            )),
            "consensus_errors_with_later_queries": sum(
                record["later_queries"] > 0 for row in group
                for record in row["consensus_error_records"]
            ),
            "terminal_consensus_errors": sum(
                record["later_queries"] == 0 for row in group
                for record in row["consensus_error_records"]
            ),
            "consensus_conformance_rejections": sum(
                record["prediction_matched"] is False for row in group
                for record in row["consensus_error_records"]
            ),
            "consensus_conformance_confirmations": sum(
                record["prediction_matched"] is True for row in group
                for record in row["consensus_error_records"]
            ),
            "consensus_disagreement_filters": sum(
                record["reason"] == "disagreement" for row in group
                for record in row["consensus_error_records"]
            ),
            "consensus_candidate_eliminations": sum(
                record["candidate_eliminations"] for row in group
                for record in row["consensus_error_records"]
            ),
            "quarantined_error_records": sum(
                record["evidence_status"] == "quarantined" for row in group
                for record in row["consensus_error_records"]
            ),
            "consensus_errors_by_family": {
                family: sum(row["consensus_errors"] for row in group if row["family"] == family)
                for family in sorted({row["family"] for row in group})
            },
            "consensus_errors_by_modulus": {
                str(modulus): sum(row["consensus_errors"] for row in group
                                  if row["modulus"] == modulus)
                for modulus in sorted({row["modulus"] for row in group})
            },
            "consensus_error_tasks": [
                {"family": row["family"], "modulus": row["modulus"],
                 "count": row["consensus_errors"], "accepted": row["accepted"]}
                for row in group if row["consensus_errors"]
            ],
            "query_sources": sum(row["query_sources"] for row in group),
            "oracle_calls": sum(row["oracle_calls"] for row in group),
            "total_cost": sum(row["total_cost"] for row in group),
            "stop_counts": dict(collections.Counter(row["status"] for row in group)),
            "technical_stop_counts": dict(collections.Counter(
                row["technical_status"] for row in group
            )),
        }
    complete = bool(
        len(rows) == 24 and all(row["all_repetitions_identical"] for row in rows)
        and all(row["wrong_oracle_answers"] == row["reported_corrupted_calls"] for row in rows)
        and all(row["used_consensus_errors"] + row["unused_consensus_errors"]
                == row["consensus_errors"] for row in rows)
        and all(row["accepted_consensus_errors"] + row["quarantined_consensus_errors"]
                == row["consensus_errors"] for row in rows)
        and all(len(row["consensus_error_records"]) == row["consensus_errors"]
                for row in rows)
    )
    return {
        "protocol": {
            "tasks": "the same 12 OOD tasks as active-generalization",
            "bias": "family-specific phase; positions (position+phase)%20<3 flipped 0<->1",
            "rate": "3/20 = 15% systematic output-position bias",
            "correlation": "same source receives the same wrong answer on every repetition",
            "vote3": "always three calls per source",
            "adaptive3": "two calls when they agree everywhere, third only on disagreement",
            "oracle_trust": "untrusted: no persistent acceptance without independent evidence",
            "diagnosis": "technical stop preserved; unresolved/state-cap after active labels becomes oracle-inconclusive",
            "price": "1 + 0.01 * len(source) per call",
            "heldout_used_for_learning": False,
        },
        "rows": rows, "findings": summaries, "measurement_complete": complete,
        "not_claimed": ["correlated_noise_detection_without_independent_evidence",
                        "voting_removes_systematic_bias", "accepted_equals_true_under_bad_oracle"],
    }


def adaptive_oracle_cost_comparison() -> dict:
    """Bağımsız gürültüde uniform-3 ile sıralı adaptive-3 maliyet karşılaştırması."""
    seeds = [20262023, 20262024, 20262025]
    tasks = [("reset", 5), ("affine", 8),
             ("ternary_reset", 7), ("ternary_weighted", 8)]
    rows = []
    for family, modulus in tasks:
        for seed in seeds:
            for protocol, adaptive in (("vote3", False), ("adaptive3", True)):
                oracle, noise = _noisy_family_oracle(family, modulus, seed)
                row = _active_measure(
                    family, modulus, 3 * ACTIVE_QUERY_BUDGET,
                    repetitions=3, adaptive_voting=adaptive, trusted_oracle=True,
                    oracle=oracle, query_cost=_active_query_price,
                )
                wrong_answers = sum(
                    response != _family_oracle(query["source"], modulus, family)
                    for query in row["queries"] for response in query.get("responses", [])
                )
                consensus_errors = sum(
                    query.get("output") is not None
                    and query["output"] != _family_oracle(query["source"], modulus, family)
                    for query in row["queries"]
                )
                used_consensus_errors = sum(
                    query.get("output") is not None
                    and query["output"] != _family_oracle(query["source"], modulus, family)
                    and query.get("used_for_learning", False)
                    for query in row["queries"]
                )
                call_histogram = collections.Counter(
                    int(query.get("calls", 1)) for query in row["queries"]
                )
                rows.append({
                    "family": family, "modulus": modulus, "seed": seed,
                    "protocol": protocol, "status": row["status"],
                    "accepted": row["accepted"], "exact": row["exact"],
                    "accepted_nonexact": row["accepted"] and not row["exact"],
                    "candidate_recovered": row["candidate_recovered"],
                    "persistent_writes": row["persistent_writes"],
                    "correct": row["correct"], "total": row["total"],
                    "query_sources": row["query_count"], "oracle_calls": row["oracle_calls"],
                    "total_cost": row["total_cost"], "seconds": row["seconds"],
                    "wrong_oracle_answers": wrong_answers,
                    "reported_corrupted_calls": noise["corrupted"],
                    "consensus_errors": consensus_errors,
                    "used_consensus_errors": used_consensus_errors,
                    "unused_consensus_errors": consensus_errors - used_consensus_errors,
                    "accepted_consensus_errors": consensus_errors if row["accepted"] else 0,
                    "quarantined_consensus_errors": (
                        consensus_errors if not row["accepted"] else 0
                    ),
                    "accepted_with_wrong_consensus": row["accepted"] and consensus_errors > 0,
                    "two_call_queries": call_histogram[2],
                    "three_call_queries": call_histogram[3],
                    "first_source": row["queries"][0]["source"] if row["queries"] else "",
                    "first_response": (row["queries"][0].get("responses") or [""])[0]
                    if row["queries"] else "",
                    "active_queries_sha256": row["active_queries_sha256"],
                })
    summaries = {}
    for protocol in ("vote3", "adaptive3"):
        group = [row for row in rows if row["protocol"] == protocol]
        summaries[protocol] = {
            "rows": len(group), "exact_rows": sum(row["exact"] for row in group),
            "accepted_rows": sum(row["accepted"] for row in group),
            "accepted_nonexact_rows": sum(row["accepted_nonexact"] for row in group),
            "accepted_with_wrong_consensus_rows": sum(
                row["accepted_with_wrong_consensus"] for row in group
            ),
            "recovered_candidate_rows": sum(row["candidate_recovered"] for row in group),
            "persistent_writes": sum(row["persistent_writes"] for row in group),
            "query_sources": sum(row["query_sources"] for row in group),
            "oracle_calls": sum(row["oracle_calls"] for row in group),
            "total_cost": sum(row["total_cost"] for row in group),
            "wrong_oracle_answers": sum(row["wrong_oracle_answers"] for row in group),
            "consensus_errors": sum(row["consensus_errors"] for row in group),
            "used_consensus_errors": sum(row["used_consensus_errors"] for row in group),
            "unused_consensus_errors": sum(row["unused_consensus_errors"] for row in group),
            "accepted_consensus_errors": sum(row["accepted_consensus_errors"] for row in group),
            "quarantined_consensus_errors": sum(
                row["quarantined_consensus_errors"] for row in group
            ),
            "two_call_queries": sum(row["two_call_queries"] for row in group),
            "three_call_queries": sum(row["three_call_queries"] for row in group),
            "stop_counts": dict(collections.Counter(row["status"] for row in group)),
        }
    uniform, adaptive = summaries["vote3"], summaries["adaptive3"]
    comparison = {
        "call_savings_fraction": 1 - adaptive["oracle_calls"] / max(1, uniform["oracle_calls"]),
        "cost_savings_fraction": 1 - adaptive["total_cost"] / max(1e-12, uniform["total_cost"]),
        "exact_delta": adaptive["exact_rows"] - uniform["exact_rows"],
        "accepted_delta": adaptive["accepted_rows"] - uniform["accepted_rows"],
        "accepted_nonexact_delta": (adaptive["accepted_nonexact_rows"]
                                    - uniform["accepted_nonexact_rows"]),
        "accepted_wrong_consensus_delta": (
            adaptive["accepted_with_wrong_consensus_rows"]
            - uniform["accepted_with_wrong_consensus_rows"]
        ),
        "consensus_error_delta": adaptive["consensus_errors"] - uniform["consensus_errors"],
    }
    paired = 0
    for family, modulus in tasks:
        for seed in seeds:
            pair = [row for row in rows if row["family"] == family
                    and row["modulus"] == modulus and row["seed"] == seed]
            paired += int(len(pair) == 2 and pair[0]["first_source"]
                          and pair[0]["first_source"] == pair[1]["first_source"]
                          and pair[0]["first_response"] == pair[1]["first_response"])
    complete = bool(
        len(rows) == 24 and paired == len(tasks) * len(seeds)
        and all(row["wrong_oracle_answers"] == row["reported_corrupted_calls"] for row in rows)
        and all(row["two_call_queries"] == 0 for row in rows if row["protocol"] == "vote3")
        and all(row["three_call_queries"] == row["query_sources"] for row in rows
                if row["protocol"] == "vote3")
        and all(row["used_consensus_errors"] + row["unused_consensus_errors"]
                == row["consensus_errors"] for row in rows)
        and all(row["accepted_consensus_errors"] + row["quarantined_consensus_errors"]
                == row["consensus_errors"] for row in rows)
    )
    return {
        "protocol": {
            "noise": "independent SHA-256 one-character answer corruption, rate 0.15",
            "fresh_noise_seeds": seeds, "source_budget_per_task": ACTIVE_QUERY_BUDGET,
            "vote3": "exactly three calls per source",
            "adaptive3": "two calls if unanimous at every position; otherwise a third call",
            "trusted_oracle": True,
            "price": "1 + 0.01 * len(source) per call",
            "paired_first_answers": paired,
            "scope": "cost/accuracy comparison, not correlated-bias robustness",
        },
        "rows": rows, "findings": summaries, "comparison": comparison,
        "measurement_complete": complete,
        "not_claimed": ["adaptive_voting_detects_correlated_bias",
                        "cost_saving_without_accuracy_tradeoff", "untrusted_oracle_acceptance"],
    }


def dual_oracle_acceptance() -> dict:
    """Bağımsız ikinci oracle ile kabul, uyuşmazlık ve güvenli abstention."""
    tasks = [(family, modulus) for modulus in (5, 7, 8)
             for family in ("reset", "affine", "ternary_reset", "ternary_weighted")]
    scenarios = ("independent_clean", "secondary_correlated",
                 "primary_correlated", "partially_reliable_untrusted")
    reference_checks = 0
    reference_mismatches = []
    for family, modulus in tasks:
        alphabet = {"reset": "ab", "affine": "01", "ternary_reset": "abc",
                    "ternary_weighted": "012"}[family]
        for length in range(5):
            for source in _symbol_strings(alphabet, length):
                reference_checks += 1
                if (_family_oracle(source, modulus, family)
                        != _independent_family_oracle(source, modulus, family)):
                    reference_mismatches.append((family, modulus, source))
    rows = []
    for family, modulus in tasks:
        for scenario in scenarios:
            primary_clean = lambda source, f=family, m=modulus: _family_oracle(source, m, f)
            independent = lambda source, f=family, m=modulus: _independent_family_oracle(source, m, f)
            primary = primary_clean
            secondary = independent
            secondary_trusted = scenario != "partially_reliable_untrusted"
            noise = None
            if scenario == "secondary_correlated":
                secondary, _stats = _correlated_family_oracle(
                    family, modulus, base_oracle=independent,
                )
            elif scenario == "primary_correlated":
                primary, _stats = _correlated_family_oracle(family, modulus)
            elif scenario == "partially_reliable_untrusted":
                seed = 20263000 + 100 * modulus + sum(map(ord, family))
                secondary, noise = _noisy_family_oracle(
                    family, modulus, seed, base_oracle=independent,
                )
            row = _active_measure(
                family, modulus, trusted_oracle=False, oracle=primary,
                secondary_oracle=secondary, secondary_trusted_oracle=secondary_trusted,
                query_cost=_active_query_price,
            )
            wrong_secondary_answers = sum(
                query.get("output") is not None
                and query["output"] != _family_oracle(query["source"], modulus, family)
                for query in row["secondary_queries"]
            )
            rows.append({
                "family": family, "modulus": modulus, "scenario": scenario,
                "status": row["status"], "technical_status": row["technical_status"],
                "accepted": row["accepted"], "provisional": row["provisional"],
                "exact": row["exact"], "correct": row["correct"], "total": row["total"],
                "persistent_writes": row["persistent_writes"],
                "candidate_recovered": row["candidate_recovered"],
                "secondary_trusted": row["secondary_trusted_oracle"],
                "secondary_check_ran": row["secondary_check_ran"],
                "secondary_complete": row["secondary_complete"],
                "secondary_mismatches": row["secondary_mismatches"],
                "secondary_primary_mismatches": row["secondary_primary_mismatches"],
                "secondary_errors": row["secondary_errors"],
                "secondary_queries": row["secondary_query_count"],
                "secondary_calls": row["secondary_oracle_calls"],
                "secondary_cost": row["secondary_total_cost"],
                "combined_calls": row["combined_oracle_calls"],
                "combined_cost": row["combined_total_cost"],
                "wrong_secondary_answers": wrong_secondary_answers,
                "reported_corrupted_secondary_calls": (
                    noise["corrupted"] if noise is not None else 0
                ),
                "accounted_secondary_calls": sum(
                    int(query.get("calls", 0)) for query in row["secondary_queries"]
                ),
                "secondary_learning_uses": sum(
                    bool(query.get("used_for_learning")) for query in row["secondary_queries"]
                ),
                "secondary_evidence_statuses": sorted({
                    query.get("evidence_status") for query in row["secondary_queries"]
                }),
                "active_queries_sha256": row["active_queries_sha256"],
            })

    def unavailable(_source):
        raise RuntimeError("secondary-offline")

    offline = _active_measure(
        "reset", 5, trusted_oracle=False,
        oracle=lambda source: _family_oracle(source, 5, "reset"),
        secondary_oracle=unavailable, secondary_trusted_oracle=True,
        query_cost=_active_query_price,
    )
    summaries = {}
    for scenario in scenarios:
        group = [row for row in rows if row["scenario"] == scenario]
        summaries[scenario] = {
            "rows": len(group), "exact_rows": sum(row["exact"] for row in group),
            "accepted_rows": sum(row["accepted"] for row in group),
            "accepted_nonexact_rows": sum(
                row["accepted"] and not row["exact"] for row in group
            ),
            "persistent_writes": sum(row["persistent_writes"] for row in group),
            "oracle_inconclusive_rows": sum(
                row["status"] == "oracle-inconclusive" for row in group
            ),
            "secondary_checked_rows": sum(row["secondary_check_ran"] for row in group),
            "candidate_mismatch_rows": sum(row["secondary_mismatches"] > 0 for row in group),
            "primary_secondary_mismatch_rows": sum(
                row["secondary_primary_mismatches"] > 0 for row in group
            ),
            "secondary_mismatches": sum(row["secondary_mismatches"] for row in group),
            "secondary_primary_mismatches": sum(
                row["secondary_primary_mismatches"] for row in group
            ),
            "untrusted_agreement_abstentions": sum(
                not row["secondary_trusted"] and not row["accepted"]
                and not row["secondary_mismatches"]
                and not row["secondary_primary_mismatches"] for row in group
            ),
            "wrong_secondary_answers": sum(row["wrong_secondary_answers"] for row in group),
            "secondary_calls": sum(row["secondary_calls"] for row in group),
            "secondary_cost": sum(row["secondary_cost"] for row in group),
            "combined_calls": sum(row["combined_calls"] for row in group),
            "combined_cost": sum(row["combined_cost"] for row in group),
            "stop_counts": dict(collections.Counter(row["status"] for row in group)),
        }
    clean = summaries["independent_clean"]
    complete = bool(
        reference_checks > 0 and not reference_mismatches
        and len(rows) == len(tasks) * len(scenarios)
        and clean["exact_rows"] == len(tasks) and clean["accepted_rows"] > 0
        and clean["accepted_nonexact_rows"] == 0
        and all(summaries[name]["accepted_rows"] == 0 for name in scenarios[1:])
        and summaries["secondary_correlated"]["candidate_mismatch_rows"] > 0
        and summaries["primary_correlated"]["primary_secondary_mismatch_rows"] > 0
        and all(row["persistent_writes"] == int(row["accepted"]) for row in rows)
        and all(row["secondary_calls"] == row["accounted_secondary_calls"] for row in rows)
        and all(row["secondary_learning_uses"] == 0 for row in rows)
        and all(row["wrong_secondary_answers"] == row["reported_corrupted_secondary_calls"]
                for row in rows if row["scenario"] == "partially_reliable_untrusted")
        and not offline["accepted"] and offline["persistent_writes"] == 0
        and offline["secondary_errors"] > 0
        and offline["status"] == "oracle-inconclusive"
    )
    return {
        "protocol": {
            "tasks": "the same 12 OOD tasks as active-generalization",
            "primary_trust": "untrusted in every scenario",
            "acceptance": "unique conformance + trusted secondary + zero candidate/primary mismatch + no secondary error",
            "partially_reliable": "independent 15% noise; secondary_trusted=False, so agreement never accepts",
            "price": "1 + 0.01 * len(source), charged separately to primary and secondary",
            "secondary_learning_uses": 0,
            "secondary_implementation": "separate prefix-arithmetic reference; primary state-loop is not called",
            "heldout_used_for_learning_or_acceptance": False,
        },
        "reference_cross_check": {
            "checks": reference_checks, "mismatches": reference_mismatches,
            "domain": "all family-alphabet strings of lengths 0 through 4 for moduli 5,7,8",
        },
        "rows": rows, "findings": summaries,
        "unavailable_control": {
            "status": offline["status"], "accepted": offline["accepted"],
            "secondary_errors": offline["secondary_errors"],
            "secondary_calls": offline["secondary_oracle_calls"],
            "secondary_cost": offline["secondary_total_cost"],
            "persistent_writes": offline["persistent_writes"],
        },
        "measurement_complete": complete,
        "not_claimed": [
            "shared_correlated_bias_detection", "agreement_implies_truth",
            "free_independent_validation", "probabilistic_noise_is_a_proof",
        ],
    }


def active_oracle_robustness() -> dict:
    """Maliyet, gürültü ve oracle-yok ölçümü; hiçbiri held-out seçimine girmez."""
    # Bu tohumlar pilot koşusundan farklıdır; noise×repetition karşılaştırması
    # yeni gürültü örneklerinde yapılır. İlk attempt ortak hash sayesinde eşlidir.
    seeds = [20262020, 20262021, 20262022]
    tasks = [("reset", 5), ("affine", 8),
             ("ternary_reset", 7), ("ternary_weighted", 8)]
    rows = []
    for family, modulus in tasks:
        for seed in seeds:
            for protocol, repetitions, call_budget in (
                ("single", 1, ACTIVE_QUERY_BUDGET),
                ("vote3", 3, 3 * ACTIVE_QUERY_BUDGET),
            ):
                oracle, noise = _noisy_family_oracle(family, modulus, seed)
                row = _active_measure(
                    family, modulus, call_budget, repetitions=repetitions,
                    oracle=oracle, query_cost=_active_query_price,
                )
                wrong_answers = sum(
                    response != _family_oracle(query["source"], modulus, family)
                    for query in row["queries"] for response in query.get("responses", [])
                )
                consensus_errors = sum(
                    query.get("output") is not None
                    and query["output"] != _family_oracle(query["source"], modulus, family)
                    for query in row["queries"]
                )
                rows.append({
                    "family": family, "modulus": modulus, "seed": seed,
                    "protocol": protocol, "status": row["status"],
                    "accepted": row["accepted"], "provisional": row["provisional"],
                    "exact": row["exact"], "correct": row["correct"], "total": row["total"],
                    "query_sources": row["query_count"], "oracle_calls": row["oracle_calls"],
                    "total_cost": row["total_cost"], "seconds": row["seconds"],
                    "wrong_oracle_answers": wrong_answers,
                    "reported_corrupted_calls": noise["corrupted"],
                    "consensus_errors": consensus_errors,
                    "active_queries_sha256": row["active_queries_sha256"],
                    "first_source": row["queries"][0]["source"] if row["queries"] else "",
                    "first_response": (row["queries"][0].get("responses") or [""])[0]
                    if row["queries"] else "",
                })

    summaries = {}
    for protocol in ("single", "vote3"):
        group = [row for row in rows if row["protocol"] == protocol]
        summaries[protocol] = {
            "rows": len(group), "exact_rows": sum(row["exact"] for row in group),
            "accepted_rows": sum(row["accepted"] for row in group),
            "provisional_exact_rows": sum(row["exact"] and not row["accepted"] for row in group),
            "oracle_calls": sum(row["oracle_calls"] for row in group),
            "query_sources": sum(row["query_sources"] for row in group),
            "total_cost": sum(row["total_cost"] for row in group),
            "wrong_oracle_answers": sum(row["wrong_oracle_answers"] for row in group),
            "consensus_errors": sum(row["consensus_errors"] for row in group),
            "stop_counts": dict(collections.Counter(row["status"] for row in group)),
        }
    paired = 0
    for family, modulus in tasks:
        for seed in seeds:
            single = next(row for row in rows if row["family"] == family
                          and row["modulus"] == modulus and row["seed"] == seed
                          and row["protocol"] == "single")
            vote = next(row for row in rows if row["family"] == family
                        and row["modulus"] == modulus and row["seed"] == seed
                        and row["protocol"] == "vote3")
            paired += int(bool(single["first_source"])
                          and single["first_source"] == vote["first_source"]
                          and single["first_response"] == vote["first_response"])

    priced_rows = [
        _active_measure(family, modulus, query_cost=_active_query_price)
        for modulus in (5, 7, 8)
        for family in ("reset", "affine", "ternary_reset", "ternary_weighted")
    ]
    priced = {
        "rows": len(priced_rows), "exact_rows": sum(row["exact"] for row in priced_rows),
        "accepted_rows": sum(row["accepted"] for row in priced_rows),
        "provisional_exact_rows": sum(row["exact_but_unverified"] for row in priced_rows),
        "oracle_calls": sum(row["oracle_calls"] for row in priced_rows),
        "total_cost": sum(row["total_cost"] for row in priced_rows),
        "price_rule": "1 + 0.01 * len(source) per call",
        "row_costs": [{"family": row["family"], "modulus": row["modulus"],
                       "calls": row["oracle_calls"], "cost": row["total_cost"],
                       "accepted": row["accepted"], "exact": row["exact"]}
                      for row in priced_rows],
    }

    # Fiyat üst sınırı ve oracle kesintisi başarı değil, açık durma davranışıdır.
    capped = _active_measure(
        "reset", 5, cost_budget=5.0, query_cost=_active_query_price,
    )
    unavailable_calls = {"count": 0}

    def unavailable(source):
        unavailable_calls["count"] += 1
        if unavailable_calls["count"] > 3:
            raise RuntimeError("oracle-offline")
        return _family_oracle(source, 5, "reset")

    unavailable_row = _active_measure("reset", 5, oracle=unavailable)
    controls = {
        "cost_cap": {"status": capped["status"], "accepted": capped["accepted"],
                     "cost": capped["total_cost"], "calls": capped["oracle_calls"],
                     "budget": 5.0, "within_budget": capped["total_cost"] <= 5.0},
        "unavailable": {"status": unavailable_row["status"],
                        "accepted": unavailable_row["accepted"],
                        "calls": unavailable_row["oracle_calls"],
                        "external_calls": unavailable_calls["count"],
                        "error_recorded": any(query.get("errors") for query in unavailable_row["queries"])},
    }
    complete = bool(
        len(rows) == len(tasks) * len(seeds) * 2 and paired == len(tasks) * len(seeds)
        and all(row["wrong_oracle_answers"] == row["reported_corrupted_calls"] for row in rows)
        and capped["status"] == "cost-budget" and not capped["accepted"]
        and capped["total_cost"] <= 5.0
        and unavailable_row["status"] == "oracle-unavailable"
        and not unavailable_row["accepted"]
        and unavailable_row["oracle_calls"] == unavailable_calls["count"] == 4
        and controls["unavailable"]["error_recorded"]
        and priced["oracle_calls"] == sum(row["query_count"] for row in priced_rows)
    )
    return {
        "protocol": {
            "noise": "independent SHA-256 draws; each corrupted answer flips one output character",
            "noise_rate": 0.15, "fresh_noise_seeds": seeds,
            "source_budget_per_task": ACTIVE_QUERY_BUDGET,
            "single_call_budget": ACTIVE_QUERY_BUDGET,
            "vote3_call_budget": 3 * ACTIVE_QUERY_BUDGET,
            "vote_rule": "per-position majority; model prediction never chooses the answer",
            "price": "1 + 0.01 * len(source) per oracle call",
            "acceptance": "only unique-conformance-complete",
            "heldout_used_for_queries_or_noise": False,
            "scope": "stochastic one-character answer noise; not adversarial/oracle-free identifiability",
        },
        "noise_rows": rows, "noise_findings": summaries,
        "paired_first_answers": paired, "priced_clean": priced, "controls": controls,
        "measurement_complete": complete,
        "not_claimed": ["adversarial_noise_robustness", "free_oracle",
                        "all_budget_stops_are_accepted", "synthesis_scaling_solved"],
    }


def active_oracle_boundary_audit() -> tuple[bool, str]:
    """Üretimde kabul/abstention, kalıcılık ve kontrollü gürültü sınırı."""
    task = _family_task(5, "reset")
    alphabet = "ab"
    pool, lengths = _active_query_domain("reset", 5, alphabet)

    def run(learner, oracle, **kwargs):
        return learner.learn_representation_active(
            task["train"], task["validation"], pool, oracle,
            max_states=5, alphabet=alphabet, lengths=lengths,
            candidate_sample=ACTIVE_CANDIDATE_SAMPLE,
            recovery_batch=ACTIVE_RECOVERY_QUERIES,
            max_expansions=SEARCH_EXPANSION_LIMIT, **kwargs,
        )

    clean_learner = _prior_learner()
    clean = run(clean_learner, lambda source: _family_oracle(source, 5, "reset"))
    clean_name = str(clean.get("id") or "")
    clean_correct = sum(
        clean_learner.apply_representation(clean_name, source) == expected
        for source, expected in task["heldout"]
    ) if clean_name else 0
    restored_clean = _prior_learner()
    restored_clean.load_state_dict(json.loads(json.dumps(clean_learner.state_dict())))
    restored_correct = sum(
        restored_clean.apply_representation(clean_name, source) == expected
        for source, expected in task["heldout"]
    ) if clean_name else 0
    clean_ok = (clean["accepted"] and clean["stop"] == "unique-conformance-complete"
                and clean_correct == len(task["heldout"])
                and restored_correct == len(task["heldout"])
                and bool(clean_learner.representations))

    capped_learner = _prior_learner()
    capped = run(capped_learner, lambda source: _family_oracle(source, 5, "reset"),
                 cost_budget=5.0, query_cost=_active_query_price)
    capped_ok = (capped["stop"] == "cost-budget" and not capped["accepted"]
                 and capped["total_cost"] <= 5.0 and not capped_learner.representations)

    attempts = {"count": 0}

    def offline(source):
        attempts["count"] += 1
        if attempts["count"] > 3:
            raise RuntimeError("oracle-offline")
        return _family_oracle(source, 5, "reset")

    offline_learner = _prior_learner()
    offline_result = run(offline_learner, offline)
    offline_ok = (offline_result["stop"] == "oracle-unavailable"
                  and not offline_result["accepted"] and not offline_learner.representations
                  and offline_result["oracle_calls"] == attempts["count"] == 4)

    # Her kaynağın ilk cevabı bozuk, sonraki iki bağımsız cevabı temizdir.
    # Böylece tek çağrı ile üçlü oylama aynı kontrollü gürültüde karşılaştırılır.
    def first_bad_oracle():
        seen = {}

        def ask(source):
            attempt = seen.get(source, 0)
            seen[source] = attempt + 1
            truth = _family_oracle(source, 5, "reset")
            if attempt:
                return truth
            position = 0
            return ("1" if truth[position] == "0" else "0") + truth[1:]

        return ask

    single_learner = _prior_learner()
    single = run(single_learner, first_bad_oracle())
    vote_learner = _prior_learner()
    vote = run(vote_learner, first_bad_oracle(), repetitions=3,
               budget=3 * ACTIVE_QUERY_BUDGET)
    vote_name = str(vote.get("id") or "")
    vote_correct = sum(
        vote_learner.apply_representation(vote_name, source) == expected
        for source, expected in task["heldout"]
    ) if vote_name else 0
    vote_ok = (not single["accepted"] and vote["accepted"]
               and vote["oracle_calls"] == 3 * vote["query_count"]
               and vote_correct == len(task["heldout"])
               and all(query["consensus_agreement"] >= 2 / 3 for query in vote["queries"]))

    inconclusive_learner = _prior_learner()
    answers = iter(("0", "1", "2"))

    def inconclusive(source):
        try:
            character = next(answers)
        except StopIteration:
            character = "0"
        return character * len(source)

    inconclusive_result = run(inconclusive_learner, inconclusive, repetitions=3)
    inconclusive_ok = (inconclusive_result["stop"] == "oracle-inconclusive"
                       and not inconclusive_result["accepted"]
                       and not inconclusive_learner.representations)
    ok = all((clean_ok, capped_ok, offline_ok, vote_ok, inconclusive_ok))
    detail = (f"clean={clean_correct}/{len(task['heldout'])}:{clean['stop']} "
              f"restore={restored_correct}/{len(task['heldout'])} "
              f"cap={capped['stop']}:{capped['total_cost']:.2f} "
              f"offline={offline_result['stop']}:{offline_result['oracle_calls']} "
              f"single={single['stop']} vote3={vote_correct}/{len(task['heldout'])}:{vote['stop']} "
              f"no-majority={inconclusive_result['stop']}")
    return ok, detail


def adaptive_oracle_boundary_audit() -> tuple[bool, str]:
    """Sıralı oy maliyeti ve güvenilmez korele oracle abstention regresyonu."""
    task = _family_task(5, "reset")
    alphabet = "ab"
    pool, lengths = _active_query_domain("reset", 5, alphabet)

    def run(learner, oracle, **kwargs):
        return learner.learn_representation_active(
            task["train"], task["validation"], pool, oracle,
            max_states=5, alphabet=alphabet, lengths=lengths,
            candidate_sample=ACTIVE_CANDIDATE_SAMPLE,
            recovery_batch=ACTIVE_RECOVERY_QUERIES,
            max_expansions=SEARCH_EXPANSION_LIMIT, **kwargs,
        )

    clean_oracle = lambda source: _family_oracle(source, 5, "reset")
    uniform_learner = _prior_learner()
    uniform = run(uniform_learner, clean_oracle, repetitions=3,
                  budget=3 * ACTIVE_QUERY_BUDGET)
    adaptive_learner = _prior_learner()
    adaptive = run(adaptive_learner, clean_oracle, repetitions=3,
                   adaptive_voting=True, budget=3 * ACTIVE_QUERY_BUDGET)
    clean_ok = (uniform["accepted"] and adaptive["accepted"]
                and uniform["oracle_calls"] == 3 * uniform["query_count"]
                and adaptive["oracle_calls"] == 2 * adaptive["query_count"]
                and adaptive["oracle_calls"] < uniform["oracle_calls"]
                and all(query["consensus_agreement"] == 1.0 for query in adaptive["queries"]))

    seen = {}

    def first_bad(source):
        attempt = seen.get(source, 0)
        seen[source] = attempt + 1
        truth = _family_oracle(source, 5, "reset")
        if attempt:
            return truth
        return ("1" if truth[0] == "0" else "0") + truth[1:]

    disagreement_learner = _prior_learner()
    disagreement = run(disagreement_learner, first_bad, repetitions=3,
                       adaptive_voting=True, budget=3 * ACTIVE_QUERY_BUDGET)
    disagreement_ok = (disagreement["accepted"]
                       and disagreement["oracle_calls"] == 3 * disagreement["query_count"]
                       and all(query["calls"] == 3 for query in disagreement["queries"]))

    correlated_learner = _prior_learner()
    correlated_oracle, _noise = _correlated_family_oracle("reset", 5)
    correlated = run(correlated_learner, correlated_oracle, repetitions=3,
                     adaptive_voting=True, trusted_oracle=False,
                     budget=3 * ACTIVE_QUERY_BUDGET)
    correlated_ok = (correlated["stop"] == "oracle-inconclusive"
                     and correlated["technical_stop"] == "state-cap"
                     and not correlated["accepted"] and not correlated_learner.representations
                     and correlated["oracle_calls"] == 2)

    untrusted_learner = _prior_learner()
    untrusted = run(untrusted_learner, clean_oracle, repetitions=3,
                    adaptive_voting=True, trusted_oracle=False,
                    budget=3 * ACTIVE_QUERY_BUDGET)
    untrusted_ok = (untrusted["stop"] == "oracle-inconclusive"
                    and untrusted["technical_stop"] == "unique-conformance-complete"
                    and not untrusted["accepted"] and not untrusted_learner.representations)
    ok = all((clean_ok, disagreement_ok, correlated_ok, untrusted_ok))
    detail = (f"clean-uniform={uniform['oracle_calls']} adaptive={adaptive['oracle_calls']} "
              f"disagreement={disagreement['oracle_calls']} "
              f"correlated={correlated['stop']}:{correlated['technical_stop']} "
              f"clean-untrusted={untrusted['stop']}:{untrusted['technical_stop']}")
    return ok, detail


def adaptive_candidate_recovery_audit() -> tuple[bool, str]:
    """Kayıp adaptive satırında yanlış ret, adayın held-out değerini silmemeli."""
    oracle, _noise = _noisy_family_oracle("ternary_weighted", 8, 20262023)
    row = _active_measure(
        "ternary_weighted", 8, 3 * ACTIVE_QUERY_BUDGET,
        repetitions=3, adaptive_voting=True, trusted_oracle=True,
        oracle=oracle, query_cost=_active_query_price,
    )
    wrong = [query for query in row["queries"]
             if query.get("output") is not None
             and query["output"] != _family_oracle(query["source"], 8, "ternary_weighted")]
    third_oracle, _third_noise = _noisy_family_oracle("ternary_weighted", 8, 20262023)
    third_responses = []
    if wrong:
        third_responses = [third_oracle(wrong[-1]["source"]) for _ in range(3)]
    third_also_wrong = bool(third_responses) and all(
        response != _family_oracle(wrong[-1]["source"], 8, "ternary_weighted")
        for response in third_responses
    )
    ok = bool(
        row["exact"] and row["correct"] == row["total"]
        and row["technical_status"] == "state-cap" and row["candidate_recovered"]
        and not row["accepted"] and row["quarantined"]
        and row["persistent_writes"] == 0 and wrong
        and all(query["used_for_learning"] for query in wrong)
        and all(query["evidence_status"] == "quarantined" for query in wrong)
        and third_also_wrong
    )
    return ok, (f"exact={row['correct']}/{row['total']} stop={row['status']} "
                f"recovered={row['candidate_recovered']} accepted={row['accepted']} "
                f"wrong-consensus={len(wrong)} third-also-wrong={third_also_wrong}")


def dual_oracle_boundary_audit() -> tuple[bool, str]:
    """Bağımsız acceptance; agreement veya erişilebilirlik tek başına kanıt değil."""
    clean = lambda source: _family_oracle(source, 5, "reset")
    independent = lambda source: _independent_family_oracle(source, 5, "reset")
    biased, _noise = _correlated_family_oracle("reset", 5, base_oracle=independent)
    accepted = _active_measure(
        "reset", 5, trusted_oracle=False, oracle=clean,
        secondary_oracle=independent, secondary_trusted_oracle=True,
    )
    untrusted = _active_measure(
        "reset", 5, trusted_oracle=False, oracle=clean,
        secondary_oracle=independent, secondary_trusted_oracle=False,
    )
    mismatch = _active_measure(
        "reset", 5, trusted_oracle=False, oracle=clean,
        secondary_oracle=biased, secondary_trusted_oracle=True,
    )

    def offline(_source):
        raise RuntimeError("secondary-offline")

    unavailable = _active_measure(
        "reset", 5, trusted_oracle=False, oracle=clean,
        secondary_oracle=offline, secondary_trusted_oracle=True,
    )
    rows = (accepted, untrusted, mismatch, unavailable)
    ok = bool(
        accepted["accepted"] and accepted["exact"] and accepted["persistent_writes"] == 1
        and not untrusted["accepted"] and untrusted["secondary_mismatches"] == 0
        and untrusted["status"] == "oracle-inconclusive"
        and not mismatch["accepted"] and mismatch["secondary_mismatches"] > 0
        and mismatch["status"] == "oracle-inconclusive"
        and not unavailable["accepted"] and unavailable["secondary_errors"] > 0
        and unavailable["status"] == "oracle-inconclusive"
        and all(not query["used_for_learning"] for row in rows
                for query in row["secondary_queries"])
    )
    return ok, (f"clean={accepted['accepted']} untrusted-agreement={untrusted['status']} "
                f"mismatch={mismatch['status']}:{mismatch['secondary_mismatches']} "
                f"offline={unavailable['status']}:{unavailable['secondary_errors']}")


def _scan_old_repertoire(train, validation) -> dict:
    """Eski akış ifadelerinin 288 kompozisyonunu train+validation'da tara."""
    learner = _prior_learner()
    expressions = learner._expression_space()
    attempted = train_matches = validated = 0
    for update in expressions:
        for emit in expressions:
            for after in (False, True):
                construction = {
                    "initial": ("const", ""), "update": update,
                    "emit": emit, "after": after,
                }
                if not learner._construction_uses(construction).issubset(learner.operators):
                    continue
                attempted += 1
                if not all(learner._run_stream(construction, source) == list(expected)
                           for source, expected in train):
                    continue
                train_matches += 1
                if all(learner._run_stream(construction, source) == list(expected)
                       for source, expected in validation):
                    validated += 1
    return {
        "attempted": attempted, "train_consistent": train_matches,
        "validation_consistent": validated, "accessible": bool(validated),
        "invention_requires_no_old_composition": not bool(validated),
    }


class _RestrictedRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, allowed_hosts):
        self.allowed_hosts = frozenset(allowed_hosts)

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlparse(newurl)
        if (parsed.scheme != "https" or parsed.hostname not in self.allowed_hosts
                or parsed.port not in (None, 443)):
            raise ValueError("outside_allowlist_redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class RestrictedGitHubResearch:
    """Yalnız HTTPS GitHub API/HTML; metin okur, indirilen kodu çalıştırmaz."""

    ALLOWED_HOSTS = frozenset({"api.github.com", "github.com"})

    def __init__(self, max_requests: int = 5, max_bytes: int = 400000):
        self.max_requests = int(max_requests)
        self.max_bytes = int(max_bytes)
        self.log: list[dict] = []

    def _get(self, url: str, purpose: str) -> bytes | None:
        entry = {"url": url, "purpose": purpose, "time": time.time()}
        parsed = urllib.parse.urlparse(url)
        if (parsed.scheme != "https" or parsed.hostname not in self.ALLOWED_HOSTS
                or parsed.port not in (None, 443)):
            entry.update({"blocked": True, "error": "outside_allowlist"})
            self.log.append(entry)
            return None
        if len([row for row in self.log if not row.get("blocked")]) >= self.max_requests:
            entry.update({"blocked": True, "error": "request_budget"})
            self.log.append(entry)
            return None
        started = time.perf_counter()
        try:
            request = urllib.request.Request(url, headers={
                "User-Agent": "Nora-Restricted-Research/1.0",
                "Accept": "application/vnd.github+json, text/plain;q=0.5",
            })
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({}),
                _RestrictedRedirect(self.ALLOWED_HOSTS),
            )
            with opener.open(request, timeout=15) as response:
                raw = response.read(self.max_bytes + 1)
                status = int(response.status)
                final_url = str(response.geturl())
            if len(raw) > self.max_bytes:
                raise ValueError("document_too_large")
            entry.update({
                "status": status, "bytes": len(raw), "final_url": final_url,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "seconds": round(time.perf_counter() - started, 4),
            })
            self.log.append(entry)
            return raw
        except Exception as exc:
            entry.update({"error": type(exc).__name__, "detail": str(exc)[:240],
                          "seconds": round(time.perf_counter() - started, 4)})
            self.log.append(entry)
            return None

    def research(self, query: str) -> dict:
        tokens = [token for token in re.findall(r"[a-z0-9]+", query.lower()) if token]
        url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode({
            "q": query, "per_page": 4,
        })
        raw = self._get(url, "repository_search")
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeError, json.JSONDecodeError):
            payload = {}
        items = payload.get("items") or []
        sources = []
        families: list[set[str]] = []
        for item in items[:4]:
            full_name = str(item.get("full_name") or "")
            if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", full_name):
                continue
            readme_url = "https://api.github.com/repos/" + full_name + "/readme"
            document = self._get(readme_url, "general_document")
            if not document:
                continue
            try:
                readme = json.loads(document.decode("utf-8"))
                content = readme.get("content") or ""
                if str(readme.get("encoding") or "").lower() == "base64":
                    text = base64.b64decode(content).decode("utf-8", errors="replace")
                else:
                    text = str(content)
            except (ValueError, UnicodeError, json.JSONDecodeError):
                continue
            low = text.lower()
            coverage = [token for token in tokens if token in low]
            vocabulary = set(re.findall(r"[a-z0-9]+", low))
            family = None
            for index, other in enumerate(families):
                union = vocabulary | other
                similarity = len(vocabulary & other) / max(1, len(union))
                if similarity >= 0.80:
                    family = index
                    break
            if family is None:
                family = len(families)
                families.append(vocabulary)
            paragraphs = re.split(r"\n\s*\n", text)
            ranked = sorted(paragraphs, key=lambda part: (
                -sum(token in part.lower() for token in tokens), len(part), part,
            ))
            excerpt = re.sub(r"\s+", " ", ranked[0]).strip()[:700] if ranked else ""
            sources.append({
                "repository": full_name,
                "citation": str(readme.get("html_url") or item.get("html_url") or ""),
                "coverage": coverage, "supports_family": len(coverage) == len(tokens),
                "independence_group": family,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "excerpt": excerpt, "code_executed": False,
            })
        supporting = [row["repository"] for row in sources if row["supports_family"]]
        independent = sorted({row["independence_group"] for row in sources
                              if row["supports_family"]})
        return {
            "query": query, "sources": sources, "supporting_sources": supporting,
            "supporting_independent_groups": independent,
            "independence_method": "README token Jaccard >= 0.80 collapsed (conservative)",
            "corroborated": len(independent) >= 2,
            "learned_concepts": tokens if len(independent) >= 2 else [],
            "used_for": "external family provenance only; never a synthesis or budget gate",
            "network_log": self.log, "allowed_hosts": sorted(self.ALLOWED_HOSTS),
            "downloaded_code_executed": False,
        }


def restricted_research_boundary() -> tuple[bool, str]:
    """Alan, şema/port, yönlendirme ve istek bütçesi ağ açılmadan reddedilir."""
    client = RestrictedGitHubResearch(max_requests=0)
    outside = client._get("https://example.com/document", "boundary_probe")
    budget = client._get("https://api.github.com/", "budget_probe")
    redirect_rejected = False
    try:
        _RestrictedRedirect(client.ALLOWED_HOSTS).redirect_request(
            urllib.request.Request("https://api.github.com/"), None, 302, "Found", {},
            "https://raw.githubusercontent.com/example/project/main/README.md",
        )
    except ValueError:
        redirect_rejected = True
    errors = [row.get("error") for row in client.log]
    ok = (outside is None and budget is None and redirect_rejected
          and errors == ["outside_allowlist", "request_budget"])
    return ok, f"alan-dışı={errors} yönlendirme-reddi={redirect_rejected} ağ-isteği=0"


def autonomous_invention_mission(*, network: bool = True,
                                 checkpoint: str = "/tmp/nora_autonomous_skill.json") -> dict:
    """Nora'ya yalnız görev+örnekler; kural/oracle ve arama tarifi yok."""
    task = _counter_task(MISSION_MODULUS)
    train, validation, heldout = task["train"], task["validation"], task["heldout"]
    old_scan = _scan_old_repertoire(train, validation)
    learner = _prior_learner()
    initial = learner.invent_representation(
        train, validation, max_states=2, max_expansions=SEARCH_EXPANSION_LIMIT,
    )
    initial_response = (
        "Bilmiyorum: bu ilişkiyi açıklayan doğrulanmış bir operatörüm yok."
        if not initial.get("id") else "Mevcut operatörlerden doğrulanmış bir aday buldum."
    )
    query = str(initial.get("kind") or "").replace("_", " ").strip()
    if network and query and not initial.get("id"):
        research = RestrictedGitHubResearch().research(query)
    else:
        research = {"query": query, "corroborated": False, "sources": [],
                    "network_log": [], "reason": "research_not_run"}
    report = {
        "protocol": {
            "prompt": MISSION_PROMPT, "task_family_after_unsealing": "hidden_counter_transducer",
            "modulus_after_unsealing": MISSION_MODULUS,
            "train": len(train), "validation": len(validation), "heldout": len(heldout),
            "train_sha256": _rows_hash(train), "validation_sha256": _rows_hash(validation),
            "heldout_sha256": _rows_hash(heldout),
            "autonomous_state_cap": AUTONOMOUS_STATE_CAP,
            "target_modulus_used_as_search_budget": False,
            "oracle_passed_to_learner": False, "heldout_passed_to_learner": False,
            "source_program_or_search_url_given": False,
        },
        "initial": {"response": initial_response, "result": initial},
        "old_repertoire_scan": old_scan, "research": research,
    }
    if initial.get("id"):
        result = initial
    else:
        # Araştırma artık bir hesaplama/bütçe kapısı değildir. Genel dış
        # kaynaklar provenance sağlar; sentez aynı veri ve tavanla her zaman
        # çalışır. Aşağıdaki zorlanmış kontrol bunun nedensel etkisini ölçer.
        result = learner.invent_representation(
            train, validation, max_states=AUTONOMOUS_STATE_CAP,
            max_expansions=SEARCH_EXPANSION_LIMIT,
        )
    research["role"] = "external_provenance_not_a_synthesis_gate"
    report["synthesis"] = result
    name = str(result.get("id") or "")
    correct = 0
    mismatches = []
    if name:
        for source, expected in heldout:
            predicted = learner.apply_representation(name, source)
            correct += int(predicted == expected)
            if predicted != expected and len(mismatches) < 3:
                mismatches.append({"source": source, "predicted": predicted, "expected": expected})
    total = len(heldout)
    exact = bool(name) and correct == total
    report["heldout"] = {"correct": correct, "total": total,
                         "accuracy": correct / total, "exact": exact,
                         "first_mismatches": mismatches}
    research_control = {"run": False, "reason": "research_forced_off_for_whole_mission"}
    if network and not initial.get("id"):
        control_learner = _prior_learner()
        control_result = control_learner.invent_representation(
            train, validation, max_states=AUTONOMOUS_STATE_CAP,
            max_expansions=SEARCH_EXPANSION_LIMIT,
        )
        control_name = str(control_result.get("id") or "")
        control_correct = sum(
            control_learner.apply_representation(control_name, source) == expected
            for source, expected in heldout
        ) if control_name else 0
        same_operator = bool(name and control_name) and (
            learner.representations[name]["transitions"]
            == control_learner.representations[control_name]["transitions"]
        )
        same_search = result.get("expansions") == control_result.get("expansions")
        research_control = {
            "run": True, "research": "forced_off_synthesis_only",
            "same_data": True, "same_state_cap": AUTONOMOUS_STATE_CAP,
            "status": control_result.get("status"),
            "heldout_correct": control_correct, "heldout_total": total,
            "exact": bool(control_name) and control_correct == total,
            "operator_identical": same_operator,
            "search_expansions": control_result.get("expansions", 0),
            "same_search_expansions": same_search,
            "solution_causal_effect_detected": not (
                same_operator and same_search and control_correct == correct
            ),
            "conclusion": (
                "Bu ölçümde araştırmanın operatör, held-out veya arama üzerinde nedensel "
                "etkisi yok; rolü dış provenance ile sınırlı."
                if same_operator and same_search and control_correct == correct else
                "Kontrol ayrıştı; araştırma etkisi hakkında üstünlük iddiası kurulmaz."
            ),
        }
    report["research_causal_control"] = research_control
    if old_scan["accessible"] or result.get("status") == "existing":
        classification = "KESIF"
    elif result.get("status") == "invented" and exact:
        classification = "ICAT_OPERATOR"
    elif result.get("status") == "invented":
        classification = "ICAT_ADAYI_HELDOUT_KALDI"
    else:
        classification = "COZULEMEDI"
    report["classification"] = classification

    ablation_ok = False
    if name:
        saved = learner.state_dict()
        learner.representations[name]["transitions"] = []
        ablated = learner.apply_representation(name, heldout[0][0])
        learner.load_state_dict(saved)
        restored = learner.apply_representation(name, heldout[0][0])
        ablation_ok = ablated is None and restored == heldout[0][1]
    report["causal_ablation"] = {"passed": ablation_ok}

    persistence = {"promoted": False, "checkpoint": None, "roundtrip_exact": False}
    if exact and ablation_ok:
        pending = checkpoint + ".pending"
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint)), exist_ok=True)
        with open(pending, "w", encoding="utf-8") as handle:
            json.dump(learner.state_dict(), handle, ensure_ascii=False, sort_keys=True)
        restored_learner = _prior_learner()
        with open(pending, "r", encoding="utf-8") as handle:
            restored_learner.load_state_dict(json.load(handle))
        restored_correct = sum(
            restored_learner.apply_representation(name, source) == expected
            for source, expected in heldout
        )
        persistence["roundtrip_exact"] = restored_correct == total
        persistence["restored_correct"] = restored_correct
        if persistence["roundtrip_exact"]:
            os.replace(pending, checkpoint)
            with open(checkpoint, "rb") as handle:
                checkpoint_hash = hashlib.sha256(handle.read()).hexdigest()
            persistence.update({"promoted": True, "checkpoint": checkpoint,
                                "sha256": checkpoint_hash})
    report["persistence"] = persistence
    research_control_ok = (not research_control["run"] or
                           (research_control.get("exact")
                            and research_control.get("operator_identical")
                            and research_control.get("same_search_expansions")))
    report["success"] = bool(classification == "ICAT_OPERATOR" and ablation_ok
                             and persistence["promoted"] and research_control_ok)
    report["not_claimed"] = [
        "general_intelligence", "phenomenal_consciousness", "better_than_LLMs",
        "research_is_computationally_necessary",
    ]
    return report


def _attempt_once(task: dict, max_states: int) -> tuple[dict, int]:
    learner = _prior_learner()
    result = learner.invent_representation(
        task["train"], task["validation"], max_states=max_states,
        max_expansions=SEARCH_EXPANSION_LIMIT,
    )
    name = str(result.get("id") or "")
    correct = sum(
        learner.apply_representation(name, source) == expected
        for source, expected in task["heldout"]
    ) if name else 0
    return result, correct


def _measured_attempt(task: dict, max_states: int) -> dict:
    # Süre, bellek izleyicisinin yükünü içermez. Bellek ikinci ve deterministik
    # bir koşuda ölçülür; iki koşu ayrışırsa ölçüm başarısızdır.
    wall = time.perf_counter()
    cpu = time.process_time()
    result, correct = _attempt_once(task, max_states)
    seconds = time.perf_counter() - wall
    cpu_seconds = time.process_time() - cpu
    tracemalloc.start()
    try:
        confirmed_result, confirmed_correct = _attempt_once(task, max_states)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    if result != confirmed_result or correct != confirmed_correct:
        raise RuntimeError("nondeterministic_measurement")
    total = len(task["heldout"])
    return {
        "status": result.get("status"), "max_states": int(max_states),
        "learned_states": result.get("states", 0), "candidates": result.get("candidates", 0),
        "correct": correct, "total": total, "accuracy": correct / total,
        "exact": bool(result.get("id")) and correct == total,
        "seconds": seconds, "cpu_seconds": cpu_seconds,
        "peak_traced_python_mib": peak / (1024 * 1024),
        "timing_profile": "untraced", "memory_profile": "second_tracemalloc_run",
        "search_expansions": result.get("expansions", 0),
        "peak_search_stack": result.get("peak_stack", 0),
        "algorithm": result.get("algorithm", "canonical-dfs-fallback"),
        "selection": result.get("selection", ""),
        "recurrent_candidates": result.get("recurrent_candidates", 0),
        "constraint_checks": result.get("constraint_checks", 0),
        "proof_completions": result.get("proof_completions", 0),
        "proof_labelings": result.get("proof_permutations", 0),
        "labeling_space": result.get("labeling_space", ""),
        "anchor_expansions": result.get("anchor_expansions", 0),
        "anchor_color_visits": result.get("anchor_color_visits", 0),
        "mdl_bits_per_candidate": result.get("mdl_bits_per_candidate", 0.0),
    }


def representation_scaling() -> dict:
    """Aynı görevde kapasite ve aynı ailede zorluk; held-out her zaman ayrı."""
    _prior_learner()  # İlk ölçülen satıra modül-import maliyeti karışmasın.
    primary = _counter_task(MISSION_MODULUS)
    state_budget = []
    for states in range(1, MISSION_MODULUS + 1):
        row = _measured_attempt(primary, states)
        row["research"] = "forced_off_state_sweep"
        state_budget.append(row)
    difficulty = []
    for modulus in range(2, 9):
        row = _measured_attempt(_counter_task(modulus), modulus)
        row["modulus"] = modulus
        difficulty.append(row)
    forced_control = _measured_attempt(primary, AUTONOMOUS_STATE_CAP)
    forced_control["research"] = "forced_off_same_data_and_autonomous_cap"
    return {
        "expansion_limit_per_state": SEARCH_EXPANSION_LIMIT,
        "autonomous_state_cap": AUTONOMOUS_STATE_CAP,
        "same_task_state_budget": state_budget,
        "task_family_difficulty": difficulty,
        "forced_no_research_control": {
            **forced_control,
            "conclusion": (
                "Araştırma hesaplama için zorunlu değil; sentez kapısı kaldırıldı, rolü dış provenance."
                if forced_control["exact"] else
                "Araştırmasız zorlanan kontrol çözemedi; bu tek sonuç nedensellik kanıtı değil."
            ),
        },
        "llm_baseline": {"run": False, "comparison_claimed": False,
                         "reason": "same-task same-data same-budget LLM run not available"},
    }


def representation_generalization() -> dict:
    """Mod7/8, OOD FST aileleri ve açık seçim-prior ablasyonları."""
    tasks: dict[tuple[str, int], dict] = {}
    measurements: dict[tuple[str, int, str, int], dict] = {}

    def measure(family: str, modulus: int, strategy="recurrent", seed=0) -> dict:
        task_key = (family, int(modulus))
        if task_key not in tasks:
            tasks[task_key] = _family_task(modulus, family)
        task = tasks[task_key]
        key = (family, int(modulus), strategy, int(seed))
        if key in measurements:
            return measurements[key]
        learner = _prior_learner()
        started = time.perf_counter()
        result = learner.invent_representation(
            task["train"], task["validation"], max_states=modulus,
            max_expansions=SEARCH_EXPANSION_LIMIT,
            selection_strategy=strategy, selection_seed=seed,
        )
        seconds = time.perf_counter() - started
        name = str(result.get("id") or "")
        correct = sum(
            learner.apply_representation(name, source) == expected
            for source, expected in task["heldout"]
        ) if name else 0
        total = len(task["heldout"])
        row = {
            "family": family, "modulus": int(modulus),
            "selection_strategy": strategy, "selection_seed": int(seed),
            "status": result.get("status"), "resolved": bool(name),
            "correct": correct, "total": total, "accuracy": correct / total,
            "exact": bool(name) and correct == total,
            "seconds": seconds, "algorithm": result.get("algorithm", "canonical-dfs-fallback"),
            "learned_states": result.get("states", 0),
            "candidates": result.get("candidates", 0),
            "recurrent_candidates": result.get("recurrent_candidates", 0),
            "selection": result.get("selection", ""),
            "search_expansions": result.get("expansions", 0),
            "proof_labelings": result.get("proof_permutations", 0),
            "proof_completions": result.get("proof_completions", 0),
            "anchor_expansions": result.get("anchor_expansions", 0),
            "anchor_color_visits": result.get("anchor_color_visits", 0),
            "labeling_space": result.get("labeling_space", ""),
            "mdl_bits_per_candidate": result.get("mdl_bits_per_candidate", 0.0),
            "train_sha256": _rows_hash(task["train"]),
            "validation_sha256": _rows_hash(task["validation"]),
            "heldout_sha256": _rows_hash(task["heldout"]),
        }
        measurements[key] = row
        return row

    counter = [dict(measure("counter", modulus)) for modulus in range(2, 9)]
    ood = [dict(measure(family, modulus))
           for modulus in (5, 7, 8)
           for family in ("reset", "affine", "ternary_reset", "ternary_weighted")]
    tie_break = []
    tie_specs = [
        ("counter", 6, (0, 1, 2)), ("counter", 7, (0, 1, 2)),
        ("counter", 8, (0, 1, 2)), ("reset", 5, (0, 1, 2)),
        ("ternary_reset", 7, (0,)), ("ternary_reset", 8, (0,)),
    ]
    for family, modulus, seeds in tie_specs:
        strategies = [("recurrent", 0)]
        strategies.extend(("uniform-random", seed) for seed in seeds)
        strategies.extend([("lexicographic", 0), ("mdl", 0)])
        for strategy, seed in strategies:
            tie_break.append(dict(measure(family, modulus, strategy, seed)))

    mdl_rows = [row for row in tie_break if row["selection_strategy"] == "mdl"]
    unique_failures = [row["family"] + "-" + str(row["modulus"]) for row in ood
                       if row["selection"] == "unique-minimal" and not row["exact"]]
    complete = (len(counter) == 7 and len(ood) == 12 and len(tie_break) == 32
                and all(row["status"] for row in counter + ood + tie_break))
    return {
        "protocol": {
            "binary": "train |x|=m+1; validation |x|=m+2,m+3; heldout |x|=m+4 + 64 long",
            "ternary": "train |x|<=3; validation 4,5; heldout 6 + 64 long",
            "expansion_limit": SEARCH_EXPANSION_LIMIT,
            "heldout_passed_to_learner": False,
            "oracle_passed_to_learner": False,
            "families": ["counter", "reset", "affine", "ternary_reset", "ternary_weighted"],
        },
        "counter_mod2_8": counter,
        "ood_families": ood,
        "selection_ablation": tie_break,
        "findings": {
            "counter_all_exact": all(row["exact"] for row in counter),
            "ood_exact_rows": sum(row["exact"] for row in ood),
            "ood_total_rows": len(ood),
            "ood_nonexact": [row["family"] + "-" + str(row["modulus"]) + ":" + row["status"]
                             for row in ood if not row["exact"]],
            "unique_minimal_but_nonexact": unique_failures,
            "recurrent_prior": "selects only when exactly one recurrent minimal candidate exists",
            "uniform_and_lexicographic": "resolve ambiguity by preference; exactness is task/seed dependent",
            "fixed_width_mdl_all_multi_candidate_ties": all(
                row["candidates"] <= 1 or (not row["resolved"] and row["selection"] == "mdl-tie")
                for row in mdl_rows
            ),
        },
        "measurement_complete": complete,
        "llm_baseline": {"run": False, "comparison_claimed": False,
                         "reason": "same-task same-data same-budget LLM run not available"},
        "not_claimed": ["general_FST_identifiability", "unique_minimal_implies_OOD_exact",
                        "recurrent_prior_is_mathematical_uniqueness", "better_than_LLMs"],
    }


def sealed_checkpoint_audit(checkpoint: str = "/tmp/nora_autonomous_skill.json") -> dict:
    """Terfi sonrası yeni mühür; önceki hiçbir veri bölümüyle girdi çakışmaz."""
    prior = _counter_task(MISSION_MODULUS)
    previous = {source for key in ("train", "validation", "heldout")
                for source, _expected in prior[key]}
    sources = [source for source in _bits(MISSION_MODULUS + 5) if source not in previous]
    rng = random.Random(MISSION_SEED + 999)
    random_sources = []
    while len(random_sources) < 64:
        length = rng.randint(33, 40)
        source = "".join(rng.choice("01") for _ in range(length))
        if source not in previous and source not in random_sources:
            random_sources.append(source)
    sources += random_sources
    with open(checkpoint, "rb") as handle:
        raw = handle.read()
    learner = _prior_learner()
    learner.load_state_dict(json.loads(raw.decode("utf-8")))
    if len(learner.representations) != 1:
        return {"exact": False, "reason": "ambiguous_checkpoint"}
    name = next(iter(learner.representations))
    correct = sum(learner.apply_representation(name, source) == _counter_oracle(source, MISSION_MODULUS)
                  for source in sources)
    return {
        "correct": correct, "total": len(sources), "accuracy": correct / len(sources),
        "exact": correct == len(sources),
        "sources_not_in_any_prior_split": all(source not in previous for source in sources),
        "checkpoint_sha256": hashlib.sha256(raw).hexdigest(),
        "audit_sha256": _rows_hash(sources), "checkpoint_modified_by_audit": False,
        "used_for_promotion_or_tuning": False,
    }


def selection_and_labeling_audit() -> tuple[bool, str]:
    """Kanonik kök etiketleme, tam N! ile aynı adayları üretmeli."""
    from nora_semantic_code import _FiniteObservationEvidence

    task = _counter_task(6)
    learner = _prior_learner()
    result = learner.invent_representation(
        task["train"], task["validation"], max_states=6,
        max_expansions=SEARCH_EXPANSION_LIMIT,
    )
    name = str(result.get("id") or "")
    if not name:
        return False, "default recurrent selection failed"
    representation = learner.representations[name]
    table = {(state, symbol): (target, output)
             for state, symbol, target, output in representation["transitions"]}
    evidence = _FiniteObservationEvidence(task["train"] + task["validation"], ["0", "1"])
    reduced = evidence.prove(table, SEARCH_EXPANSION_LIMIT)
    exhaustive = evidence.prove(table, SEARCH_EXPANSION_LIMIT, exhaustive_labelings=True)
    same_candidates = (
        reduced.get("status") == exhaustive.get("status") == "proved"
        and reduced.get("candidates") == exhaustive.get("candidates")
        and reduced.get("recurrent_candidates") == exhaustive.get("recurrent_candidates")
        and reduced.get("table") == exhaustive.get("table")
    )
    exact = all(learner.apply_representation(name, source) == expected
                for source, expected in task["heldout"])
    alternatives = {}
    for strategy in ("uniform-random", "lexicographic", "mdl"):
        other = _prior_learner()
        row = other.invent_representation(
            task["train"], task["validation"], max_states=6,
            max_expansions=SEARCH_EXPANSION_LIMIT, selection_strategy=strategy,
        )
        alternatives[strategy] = row
    alternative_ok = (
        alternatives["mdl"].get("status") == "ambiguous"
        and alternatives["mdl"].get("selection") == "mdl-tie"
        and all(alternatives[s].get("candidates") == result.get("candidates")
                for s in alternatives)
        and all(alternatives[s].get("id") for s in ("uniform-random", "lexicographic"))
    )
    ok = (same_candidates and exact and alternative_ok
          and reduced.get("expansions", 0) < exhaustive.get("expansions", 0))
    detail = (f"N!-canonical candidates={exhaustive.get('candidates')}/"
              f"{reduced.get('candidates')} expansions={exhaustive.get('expansions')}/"
              f"{reduced.get('expansions')} exact={exact} mdl={alternatives['mdl'].get('status')}")
    return ok, detail


def _technical_independent_oracles() -> tuple[bool, bool]:
    """Dil testinin evet/hayır beklentisi üretim metninden değil makine sayımından gelir."""
    alphabet = ("0", "1")

    def run(table, source):
        state, output = 0, []
        for symbol in source:
            state, emitted = table[(state, symbol)]
            output.append(emitted)
        return "".join(output)

    # Aynı sonlu gözlem, iki farklı tek-durumlu recurrent etiketlemeye açıktır.
    observed_forms = []
    for unseen_output in alphabet:
        table = {(0, "0"): (0, "0"), (0, "1"): (0, unseen_output)}
        if run(table, "0") == "0":
            observed_forms.append(tuple(sorted(table.items())))
    finite_observation_nonunique = len(set(observed_forms)) == 2

    # Bağımsız tam-davranış kontrolü: bütün iki-durumlu deterministik Mealy
    # makinelerini tara; hedefle eşdeğer olanların kanonik minimal biçimlerini say.
    target = {(0, "0"): (0, "0"), (0, "1"): (1, "1"),
              (1, "0"): (1, "1"), (1, "1"): (0, "0")}
    sources = [""]
    for length in range(1, 6):
        sources.extend("".join(bits) for bits in itertools.product(alphabet, repeat=length))

    def behavior(table):
        return tuple(run(table, source) for source in sources)

    def canonical(table):
        mapping, queue = {0: 0}, [0]
        for state in queue:
            for symbol in alphabet:
                nxt, _output = table[(state, symbol)]
                if nxt not in mapping:
                    mapping[nxt] = len(mapping)
                    queue.append(nxt)
        if len(mapping) != 2:
            return None
        return tuple((mapping[state], symbol, mapping[table[(state, symbol)][0]],
                      table[(state, symbol)][1])
                     for state in queue for symbol in alphabet)

    wanted = behavior(target)
    equivalent_forms = set()
    keys = tuple(target)
    for encoded in itertools.product(range(4), repeat=4):
        table = {key: (value // 2, str(value % 2)) for key, value in zip(keys, encoded)}
        if behavior(table) == wanted:
            form = canonical(table)
            if form is not None:
                equivalent_forms.add(form)
    fixed_behavior_unique = len(equivalent_forms) == 1 and canonical(target) in equivalent_forms
    return finite_observation_nonunique, fixed_behavior_unique


def technical_epistemic_boundary() -> tuple[bool, str]:
    """Teknik niyet, parafraz, negasyon ve bibliyografik kanıtın uçtan uca sınavı."""
    from nora_agi import NoraAGI

    finite_nonunique, fixed_unique = _technical_independent_oracles()
    checks = []

    def record(name, ok, detail=""):
        checks.append((name, bool(ok), str(detail)))

    record("independent_no_oracle", finite_nonunique)
    record("independent_yes_oracle", fixed_unique)

    def fresh(seed=41):
        return NoraAGI(seed=seed, boot_thinker=False)

    no_questions = [
        "minimal recurrent form her zaman tek midir? Evet ya da hayır de ve nedenini açıkla; emin değilsen bilmiyorum de.",
        "Bir FST’nin minimal recurrent biçimi her koşulda tekil midir? Neden?",
        "Does a strongly connected finite-state transducer always have a unique minimal form?",
    ]
    for i, question in enumerate(no_questions):
        agi = fresh()
        response = agi.converse(question)
        low = response.lower()
        ok = (finite_nonunique and low.startswith("hayır;") and "sonlu gözlem" in low
              and "izomorfizm" in low and "; ancak aynı tam davranış" in low
              and agi.episteme.guard.audit_response(question, response)["safe"])
        record(f"no_paraphrase_{i}", ok, response)

    positive = "Deterministik ve tam bir Mealy makinesinin minimal biçimi izomorfizmaya kadar tek midir?"
    agi = fresh()
    response = agi.converse(positive)
    record("fixed_behavior_positive", fixed_unique and response.lower().startswith("evet;")
           and "; ancak aynı tam davranış" in response, response)

    negated = [
        "Deterministik değil ama tam bir Mealy makinesinin minimal biçimi tek midir?",
        "Deterministik olmayan, tam bir Mealy makinesinin minimal biçimi tek midir?",
        "Deterministic not: is a complete Mealy machine's minimal form unique?",
    ]
    for i, question in enumerate(negated):
        agi = fresh()
        query = agi.episteme.technical.parse(question)
        response = agi.converse(question)
        ok = bool(query and "deterministic" in query.negated
                  and "complete" not in query.negated and response.lower().startswith("hayır;"))
        record(f"negation_direction_{i}", ok, response)

    negative_polarity = "Deterministik ve tam bir Mealy makinesinin minimal biçimi tek değil mi?"
    agi = fresh()
    response = agi.converse(negative_polarity)
    record("negative_question_polarity", response.lower().startswith("hayır;"), response)

    bibliographic = [
        "2014 Computational Linguistics, “Halvorsen & Erdoğan, Recurrent Minimization of Weighted Transducers via Prefix-Incompatibility Graphs” kaydındaki algoritmanın adı ve temel fikri nedir?",
        "Halvorsen ve Erdoğan’ın 2014’teki Recurrent Minimization of Weighted Transducers via Prefix-Incompatibility Graphs çalışmasında kullanılan algoritma ne, temel fikri ne?",
    ]
    for i, question in enumerate(bibliographic):
        agi = fresh()
        response = agi.converse(question)
        low = response.lower()
        refusal = "doğrulamadım" in low and "bilmiyorum" in low
        safe = agi.episteme.guard.audit_response(question, response)["safe"]
        hallucination = ("Algoritmanın adı Prefix-Incompatibility Minimization; "
                         "temel fikri seq2seq hata optimizasyonudur.")
        caught = not agi.episteme.guard.audit_response(question, hallucination)["safe"]
        mixed = "Bilmiyorum. Algoritmanın adı Prefix-Incompatibility Minimization."
        mixed_caught = not agi.episteme.guard.audit_response(question, mixed)["safe"]
        record(f"unknown_source_{i}", refusal and safe and caught and mixed_caught, response)

    title = "Recurrent Minimization of Weighted Transducers via Prefix-Incompatibility Graphs"
    document = ("Algorithm: QV-Bridge\n"
                "Basic idea: eşlenmeyen önekleri ayrı tutup geçişleri birleştirmek\n")
    metadata = {"title": title, "year": 2014, "venue": "Computational Linguistics",
                "authors": ["A. Researcher"]}
    agi = fresh()
    stored = agi.episteme.technical.register_document("local:verified", document, metadata, verified=True)
    response = agi.converse(bibliographic[0])
    fields = dict(line.split(":", 1) for line in document.strip().splitlines())
    documented = (stored["status"] == "verified" and fields["Algorithm"].strip() in response
                  and fields["Basic idea"].strip() in response
                  and agi.episteme.guard.audit_response(bibliographic[0], response)["safe"])
    record("verified_document_positive", documented, response)

    agi = fresh()
    unverified = agi.episteme.technical.register_document("local:unverified", document, metadata, verified=False)
    response = agi.converse(bibliographic[0])
    record("unverified_document_rejected", unverified["status"] == "unverified"
           and "bilmiyorum" in response and fields["Algorithm"] not in response, response)

    wrong_year = bibliographic[0].replace("2014", "2015")
    agi = fresh()
    agi.episteme.technical.register_document("local:verified", document, metadata, verified=True)
    response = agi.converse(wrong_year)
    record("bibliographic_year_mismatch", "bilmiyorum" in response and fields["Algorithm"] not in response, response)

    agi = fresh()
    compound = "Prefix-incompatibility graph nedir?"
    response = agi.converse(compound)
    record("compound_without_invention", "bilmiyorum" in response and "önek uyumsuzluğu" in response
           and "çizge" in response, response)
    record("nontechnical_control", agi.episteme.technical.answer("Kediler neden mırlar?") is None
           and agi.episteme.technical.answer("2 + 2 kaç eder?") is None)

    passed = sum(ok for _name, ok, _detail in checks)
    failed = [name for name, ok, _detail in checks if not ok]
    detail = f"{passed}/{len(checks)}"
    if failed:
        detail += " failed=" + ",".join(failed)
    return passed == len(checks), detail


def _bounded_family_task(family: str, modulus: int, *, train_n: int = 64,
                         validation_n: int = 16, heldout_n: int = 64,
                         seed: int = MISSION_SEED) -> dict:
    """Büyük modlarda exhaustive evreni şişirmeden ayrı ve yeniden üretilebilir split.

    Beklenenler iki bağımsız oracle'ın uzlaşısıyla hesaplanır. Kaynaklar ve
    uzunluk aralıkları train, validation ve held-out'ta ayrıdır; pozitif-olay
    katmanı çoğunluk tabanının sahte başarısını önler. Hiçbir kaynak/çıktı çifti
    öğrenici dışında modele hazır tablo olarak verilmez.
    """
    if family in {"counter", "affine"}:
        alphabet = "01"
    elif family == "reset":
        alphabet = "ab"
    elif family == "ternary_reset":
        alphabet = "abc"
    elif family == "ternary_weighted":
        alphabet = "012"
    else:
        raise ValueError("unknown_family")
    rng = random.Random(seed + 1009 * modulus + sum(map(ord, family)))
    used: set[str] = set()

    def make(length: int, positive: bool) -> str:
        """Bir sınıfı yok sayan çoğunluk modelinin geçemeyeceği diziler üret.

        Özellikle binary counter'da mod-1 olayına rastgele ulaşma olasılığı mod
        büyüdükçe üstel olarak azalır. Önceki düzgün-rastgele örneklemde mod24
        bütün splitleri yalnız '0' çıkarıyor, bir-durumlu hep-sıfır FST'si 64/64
        görünüyordu. Burada yalnız ÖRNEK SEÇİMİ katmanlaşır; etiket yine bağımsız
        oracle'dan hesaplanır ve modele pozitiflik bilgisi ayrıca verilmez.
        """
        for _attempt in range(50000):
            length = max(1, int(length))
            if positive and family in {"counter", "reset"}:
                # Eşik civarında pozitif olayları erişilebilir kılan karışım;
                # hedef çıktı dizisi veya çözüm tablosu üretilmez.
                hot = "1" if family == "counter" else "a"
                cold = "0" if family == "counter" else "b"
                source = "".join(hot if rng.random() < 0.9 else cold
                                 for _ in range(length))
            elif positive and family == "ternary_reset":
                source = "".join(rng.choices(alphabet, weights=(0.65, 0.25, 0.10),
                                             k=length))
            else:
                source = "".join(rng.choice(alphabet) for _ in range(length))
            if source in used:
                continue
            expected = _family_oracle(source, modulus, family)
            if ("1" in expected) != bool(positive):
                continue
            # İki ayrı algoritma aynı etiketi vermiyorsa veri üretimi durur;
            # tutarsız split hangi modelin lehine olursa olsun kullanılmaz.
            if expected != _independent_family_oracle(source, modulus, family):
                raise AssertionError("family oracle disagreement")
            used.add(source)
            return source
        raise RuntimeError(f"stratified-source-exhausted:{family}:mod{modulus}:{positive}")

    def sources(count: int, lengths: tuple[int, ...]) -> list[str]:
        out = []
        count = int(count)
        positive_quota = max(1, count // 4)
        stride = max(1, count // positive_quota)
        positive_seen = negative_seen = 0
        for index in range(count):
            positive = index % stride == 0
            # Pozitif ve negatif sınıflar AYNI uzunluk havuzunu kendi içlerinde
            # dengeli dolaşır. Böylece model 'uzunsa pozitif' kısayolunu kullanamaz.
            # Kısa counter/reset dizilerinde yalnız birkaç pozitif kaynak olduğu
            # için havuz mod+2'den başlar; split aralıkları yine ayrışır.
            class_index = positive_seen if positive else negative_seen
            length = lengths[class_index % len(lengths)]
            if positive:
                positive_seen += 1
            else:
                negative_seen += 1
            # Yaklaşık dörtte bir pozitif-olay içeren dizi; kalanlar tamamen
            # negatif. İki sınıfın ayrı held-out doğruluğu aşağıda ölçülür.
            out.append(make(length, positive=positive))
        return out

    train_sources = sources(train_n, tuple(range(modulus + 2, modulus + 6)))
    validation_sources = sources(validation_n, tuple(range(modulus + 6, modulus + 10)))
    heldout_sources = sources(heldout_n, tuple(range(modulus + 10, modulus + 18)))
    pairs = lambda rows: [(source, _family_oracle(source, modulus, family))
                          for source in rows]
    task = {
        "family": family, "modulus": int(modulus),
        "train": pairs(train_sources), "validation": pairs(validation_sources),
        "heldout": pairs(heldout_sources),
    }
    assert not ({s for s, _ in task["train"]} & {s for s, _ in task["heldout"]})
    assert not ({s for s, _ in task["validation"]} & {s for s, _ in task["heldout"]})
    for split in ("train", "validation", "heldout"):
        labels = [expected for _source, expected in task[split]]
        assert any("1" in expected for expected in labels), f"positive-free:{split}"
        assert any("1" not in expected for expected in labels), f"negative-free:{split}"
    return task


def _heldout_prediction_metrics(predictions, heldout) -> dict:
    """Toplam doğruluğun içinde kaybolan pozitif/negatif sınıfları ayır."""
    rows = list(zip(predictions, heldout))
    positive = [(prediction == expected) for prediction, (_source, expected) in rows
                if "1" in expected]
    negative = [(prediction == expected) for prediction, (_source, expected) in rows
                if "1" not in expected]
    p_correct, n_correct = sum(positive), sum(negative)
    p_total, n_total = len(positive), len(negative)
    return {
        "positive_correct": p_correct, "positive_total": p_total,
        "negative_correct": n_correct, "negative_total": n_total,
        "positive_accuracy": p_correct / max(1, p_total),
        "negative_accuracy": n_correct / max(1, n_total),
        "balanced_accuracy": 0.5 * (p_correct / max(1, p_total) +
                                     n_correct / max(1, n_total)),
        # Bu satır modelin skoru değil, aynı splitteki hep-negatif tabanın
        # kaç dizi tutturacağını gösteren ölçüm hijyenidir.
        "majority_baseline_correct": max(p_total, n_total),
    }


def _continuous_measured_attempt(task: dict, *, hidden: int = 64,
                                 max_epochs: int = 240, seed: int = 0,
                                 class_balance: bool = True,
                                 prefix_augmentation: int = 0,
                                 prefix_warmup_epochs: int = 0,
                                 checkpoint_selection: str = "train_first",
                                 causal_summary: bool | str = False) -> tuple[dict, object]:
    from nora_semantic_code import CompositionalCodeLearner
    learner = CompositionalCodeLearner()
    started = time.perf_counter()
    result = learner.invent_continuous_representation(
        task["train"], task["validation"], hidden=hidden, seed=seed,
        max_epochs=max_epochs, batch_size=16, patience=60,
        class_balance=class_balance, prefix_augmentation=prefix_augmentation,
        prefix_warmup_epochs=prefix_warmup_epochs,
        checkpoint_selection=checkpoint_selection,
        causal_summary=causal_summary,
    )
    seconds = time.perf_counter() - started
    name = str(result.get("id") or "")
    predictions = [learner.apply_representation(name, source, allow_provisional=True)
                   for source, _expected in task["heldout"]]
    correct = sum(prediction == expected for prediction, (_source, expected)
                  in zip(predictions, task["heldout"]))
    total = len(task["heldout"])

    # Ağırlıklar çıkarıldığında sonuç kaybolmalı; save/load aynı tahminleri geri
    # getirmeli. Provisional model normal üretim kapısından geçemez.
    saved = learner.state_dict()
    ablated = None
    if name:
        learner.representations[name]["weights"] = {}
        learner._continuous_models.clear()
        ablated = learner.apply_representation(
            name, task["heldout"][0][0], allow_provisional=True,
        )
    learner.load_state_dict(saved)
    restored = [learner.apply_representation(name, source, allow_provisional=True)
                for source, _expected in task["heldout"]]
    production_gate = learner.apply_representation(name, task["heldout"][0][0])
    gate_ok = ((result.get("acceptance") == "accepted" and production_gate == predictions[0])
               or (result.get("acceptance") == "provisional" and production_gate is None))
    row = {
        **{key: value for key, value in result.items() if key not in {"seconds"}},
        "family": task["family"], "modulus": task["modulus"],
        "seconds": seconds, "correct": correct, "total": total,
        "accuracy": correct / max(1, total), "exact": correct == total,
        "states": result.get("hidden", 0),
        "transitions": "continuous",
        "candidates": "distributed-next-symbol-distribution",
        "search_expansions": result.get("gradient_steps", 0),
        "selected_search_expansions": result.get(
            "selected_gradient_steps", result.get("gradient_steps", 0)),
        "budget": int(max_epochs),
        "budget_unit": "epochs",
        "ablation_passed": bool(name) and ablated is None,
        "roundtrip_exact_predictions": restored == predictions,
        "production_gate_passed": gate_ok,
        "train_sha256": _rows_hash(task["train"]),
        "validation_sha256": _rows_hash(task["validation"]),
        "heldout_sha256": _rows_hash(task["heldout"]),
        **_heldout_prediction_metrics(predictions, task["heldout"]),
    }
    return row, learner


class _BenchmarkAttention:
    """Yalnız geçmiş ve şimdiki sembole bakan standart çok-kafalı dikkat.

    Sınıfı doğrudan ``torch.nn.Module``den türetmek yerine burada bir
    ``new`` yardımcısı kullanmak dış checkpoint'in alan adlarını aynen
    karşılamamızı sağlar. Böylece pretrained GPT için başka bir gizli
    uygulama veya API gerekmez.
    """

    def __new__(cls, d_model: int, n_heads: int, dropout: float = 0.0):
        import torch

        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.n_heads = int(n_heads)
                self.head_dim = int(d_model) // int(n_heads)
                self.q_proj = torch.nn.Linear(d_model, d_model)
                self.k_proj = torch.nn.Linear(d_model, d_model)
                self.v_proj = torch.nn.Linear(d_model, d_model)
                self.out_proj = torch.nn.Linear(d_model, d_model)
                self.resid_dropout = torch.nn.Dropout(float(dropout))

            def forward(self, xs):
                batch, width, dim = xs.shape
                q = self.q_proj(xs).view(batch, width, self.n_heads, self.head_dim).transpose(1, 2)
                k = self.k_proj(xs).view(batch, width, self.n_heads, self.head_dim).transpose(1, 2)
                v = self.v_proj(xs).view(batch, width, self.n_heads, self.head_dim).transpose(1, 2)
                scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
                causal = torch.tril(torch.ones(width, width, dtype=torch.bool, device=xs.device))
                scores = scores.masked_fill(~causal, float("-inf"))
                weights = torch.nn.functional.softmax(scores, dim=-1)
                weights = torch.nn.functional.dropout(
                    weights, p=self.resid_dropout.p, training=self.training)
                out = (weights @ v).transpose(1, 2).contiguous().view(batch, width, dim)
                return self.resid_dropout(self.out_proj(out))

        return Attention()


class _BenchmarkTransformer:
    """Küçük decoder-only GPT; dış MiniGPT checkpoint'iyle de uyumlu.

    Bu sınıf Nora'nın üretim modeli değildir. Aynı train/validation/held-out
    çiftlerinde bağımsız Transformer tabanı olarak ``law_battery`` içindedir.
    ``model_state`` alan adları indirilen MiniGPT ile birebir eşleşir; ağırlık
    yüklemesi ``strict=True`` olduğundan sessiz eksik/yanlış katman olamaz.
    """

    def __new__(cls, *, vocab_size: int, context_length: int = 128,
                d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                ffn_dim: int = 256, dropout: float = 0.0):
        import torch

        class FeedForward(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(d_model, ffn_dim)
                self.act = torch.nn.GELU()
                self.fc2 = torch.nn.Linear(ffn_dim, d_model)
                self.dropout = torch.nn.Dropout(float(dropout))

            def forward(self, xs):
                return self.dropout(self.fc2(self.act(self.fc1(xs))))

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ln1 = torch.nn.LayerNorm(d_model)
                self.attn = _BenchmarkAttention(d_model, n_heads, dropout)
                self.ln2 = torch.nn.LayerNorm(d_model)
                self.ffn = FeedForward()

            def forward(self, xs):
                xs = xs + self.attn(self.ln1(xs))
                return xs + self.ffn(self.ln2(xs))

        class GPT(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.d_model = int(d_model)
                self.context_length = int(context_length)
                self.token_emb = torch.nn.Embedding(vocab_size, d_model)
                self.blocks = torch.nn.ModuleList(Block() for _ in range(n_layers))
                self.ln_f = torch.nn.LayerNorm(d_model)
                self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)
                self.drop = torch.nn.Dropout(float(dropout))
                position = torch.arange(context_length, dtype=torch.float32).unsqueeze(1)
                index = torch.arange(0, d_model, 2, dtype=torch.float32)
                frequency = torch.exp(-math.log(10000.0) * index / d_model)
                encoding = torch.zeros(context_length, d_model)
                encoding[:, 0::2] = torch.sin(position * frequency)
                encoding[:, 1::2] = torch.cos(position * frequency)
                self.register_buffer("pe", encoding, persistent=False)
                self.apply(self._initialize)

            @staticmethod
            def _initialize(module):
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, torch.nn.Embedding):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=d_model ** -0.5)

            def forward(self, ids):
                width = int(ids.shape[1])
                if width > self.context_length:
                    raise ValueError("transformer-context-length")
                xs = self.token_emb(ids) * math.sqrt(self.d_model) + self.pe[:width]
                xs = self.drop(xs)
                for block in self.blocks:
                    xs = block(xs)
                return self.lm_head(self.ln_f(xs))

        return GPT()


def _transformer_vocab_and_model(task: dict, *, seed: int,
                                 pretrained_checkpoint: str = ""):
    """Görev sözlüğünü train'den kur; varsa gerçek GPT ağırlıklarını yükle.

    Pretrained tokenizer'da bulunmayan 0/1/2 gibi semboller için yeni gömme ve
    çıktı satırları açılır. Eski Transformer gövdesi korunur. Görevin output
    sınıfları yine yalnız train'den gelir; bütün 65 harf serbest bırakılıp
    pretrained modele yapay bir hata/avantaj kanalı açılmaz.
    """
    import torch

    train = task["train"]
    input_symbols = sorted({s for source, _ in train for s in source})
    output_symbols = sorted({s for _source, out in train for s in out})
    provenance = {"pretrained": False}
    if pretrained_checkpoint:
        checkpoint = torch.load(pretrained_checkpoint, map_location="cpu", weights_only=True)
        config = dict(checkpoint["model_config"])
        model = _BenchmarkTransformer(
            vocab_size=int(config["vocab_size"]),
            context_length=int(config["context_length"]),
            d_model=int(config["d_model"]), n_heads=int(config["n_heads"]),
            n_layers=int(config["n_layers"]), ffn_dim=int(config["ffn_dim"]),
            dropout=float(config.get("dropout", 0.0)),
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        characters = [str(s) for s in checkpoint["tokenizer"]["chars"]]
        symbol_id = {symbol: index for index, symbol in enumerate(characters)}
        provenance = {
            "pretrained": True,
            "checkpoint_sha256": hashlib.sha256(
                open(pretrained_checkpoint, "rb").read()).hexdigest(),
            "checkpoint_bytes": os.path.getsize(pretrained_checkpoint),
            "source_model": "MiniGPT-TinyShakespeare",
            "source_repository": "NaveedShk/miniGPT_Assignment-",
            "source_commit": "5063ea23d04f596ee6ddbb258cd2b0bbb245d0b3",
            "pretraining_data": "Tiny Shakespeare; ayrı ve ek ön-eğitim",
            "pretraining_steps_recorded": int((checkpoint.get("train_config") or {})
                                               .get("train", {}).get("max_steps", 0)),
        }
    else:
        characters = []
        symbol_id = {}
        model = None
    torch.manual_seed(int(seed))
    missing = sorted((set(input_symbols) | set(output_symbols)) - set(symbol_id))
    old_vocab = len(symbol_id)
    if missing:
        if old_vocab:
            dimension = int(model.token_emb.weight.shape[1])
            new_embedding = torch.nn.Embedding(old_vocab + len(missing), dimension)
            new_head = torch.nn.Linear(dimension, old_vocab + len(missing), bias=False)
            with torch.no_grad():
                new_embedding.weight[:old_vocab].copy_(model.token_emb.weight)
                new_head.weight[:old_vocab].copy_(model.lm_head.weight)
            model.token_emb = new_embedding
            model.lm_head = new_head
        for offset, symbol in enumerate(missing):
            symbol_id[symbol] = old_vocab + offset
        characters.extend(missing)
    if not symbol_id:
        raise ValueError("empty transformer vocabulary")
    if not pretrained_checkpoint:
        # vocab_size=0 ile kurulan geçici model yerine aynı mimariyi sıfırdan,
        # gerçek sözlükle başlat. Tohum ağırlık inisiyalizasyonundan önce sabit.
        torch.manual_seed(int(seed))
        model = _BenchmarkTransformer(vocab_size=len(symbol_id))
        characters = [symbol for symbol, _index in
                      sorted(symbol_id.items(), key=lambda row: row[1])]
    provenance["new_task_symbols"] = missing
    provenance["original_vocab_size"] = old_vocab
    provenance["task_vocab_size"] = len(symbol_id)
    return model, symbol_id, characters, input_symbols, output_symbols, provenance


def _transformer_batch(model, rows, symbol_id, output_symbols, *, class_weights=None):
    import torch

    output_id = {symbol: index for index, symbol in enumerate(output_symbols)}
    allowed = torch.tensor([symbol_id[symbol] for symbol in output_symbols], dtype=torch.long)
    width = max(len(source) for source, _expected in rows)
    xs = torch.zeros((len(rows), width), dtype=torch.long)
    ys = torch.zeros((len(rows), width), dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.bool)
    for index, (source, expected) in enumerate(rows):
        xs[index, :len(source)] = torch.tensor([symbol_id[s] for s in source])
        ys[index, :len(expected)] = torch.tensor([output_id[s] for s in expected])
        mask[index, :len(source)] = True
    logits = model(xs)[..., allowed]
    loss = torch.nn.functional.cross_entropy(
        logits[mask], ys[mask], reduction="mean", weight=class_weights)
    return loss, logits, ys, mask


def _transformer_evaluate(model, rows, symbol_id, characters, output_symbols,
                          *, batch_size: int = 32) -> dict:
    import torch

    exact = correct = total = 0
    class_correct = [0 for _ in output_symbols]
    class_total = [0 for _ in output_symbols]
    predictions = []
    losses = []
    with torch.inference_mode():
        model.eval()
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            loss, logits, ys, mask = _transformer_batch(model, chunk, symbol_id, output_symbols)
            predicted = logits.argmax(-1)
            correct += int(((predicted == ys) & mask).sum())
            total += int(mask.sum())
            exact += sum(int(bool((predicted[i] == ys[i])[mask[i]].all()))
                         for i in range(len(chunk)))
            for class_index in range(len(output_symbols)):
                selected = mask & (ys == class_index)
                class_total[class_index] += int(selected.sum())
                class_correct[class_index] += int(((predicted == ys) & selected).sum())
            losses.append(float(loss))
            for index, (source, _expected) in enumerate(chunk):
                ids = predicted[index, :len(source)].tolist()
                predictions.append("".join(output_symbols[i] for i in ids))
    class_accuracy = {symbol: class_correct[index] / class_total[index]
                      for index, symbol in enumerate(output_symbols) if class_total[index]}
    return {
        "token_accuracy": correct / max(1, total),
        "macro_token_accuracy": sum(class_accuracy.values()) / max(1, len(class_accuracy)),
        "class_token_accuracy": class_accuracy,
        "exact_sequences": exact, "total_sequences": len(rows),
        "exact": exact == len(rows), "loss": sum(losses) / max(1, len(losses)),
        "predictions": predictions,
    }


def _transformer_causality_probe() -> bool:
    """Gelecekteki iki girdi değişince önceki logits aynı kalmalı."""
    import torch

    torch.manual_seed(771)
    model = _BenchmarkTransformer(vocab_size=3, d_model=16, n_heads=4,
                                  n_layers=2, ffn_dim=32, dropout=0.0)
    left = torch.tensor([[0, 1, 2, 0, 1]])
    right = torch.tensor([[0, 1, 0, 1, 2]])
    with torch.inference_mode():
        model.eval()
        a = model(left)
        b = model(right)
    return bool(torch.allclose(a[:, :2], b[:, :2], atol=1e-6, rtol=0.0)
                and not torch.allclose(a[:, 2:], b[:, 2:], atol=1e-6, rtol=0.0))


def _transformer_measured_attempt(task: dict, *, example_budget: int,
                                  seed: int, learning_rate: float = 0.001,
                                  batch_size: int = 16, patience: int = 60,
                                  class_balance: bool = True,
                                  pretrained_checkpoint: str = "") -> dict:
    """CCG'nin toplam örnek/gradyan bütçesini aşmayan Transformer ölçümü."""
    import torch

    started = time.perf_counter()
    model, symbol_id, characters, input_symbols, output_symbols, provenance = \
        _transformer_vocab_and_model(
            task, seed=seed, pretrained_checkpoint=pretrained_checkpoint)
    train = list(task["train"])
    validation = list(task["validation"])
    if any(s not in symbol_id for source, _ in validation for s in source):
        raise ValueError("validation-alphabet")
    torch.manual_seed(int(seed) + 19)
    np.random.seed(int(seed) % (2 ** 32 - 1))
    output_id = {symbol: index for index, symbol in enumerate(output_symbols)}
    counts = torch.zeros(len(output_symbols), dtype=torch.float64)
    for _source, expected in train:
        for symbol in expected:
            counts[output_id[symbol]] += 1
    weights = torch.zeros_like(counts, dtype=torch.float32)
    present = counts > 0
    weights[present] = (counts[present].sum() / (present.sum() * counts[present])).float()
    if not class_balance:
        weights.fill_(1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    rng = np.random.default_rng(int(seed) + 91_733)
    max_epochs = max(1, int(example_budget) // max(1, len(train)))
    best_score = (-1, -1, -float("inf"), -float("inf"), -float("inf"), -float("inf"))
    best_weights = None
    best_epoch = stale = gradient_steps = presentations = 0
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        for start in range(0, len(order), int(batch_size)):
            rows = [train[int(i)] for i in order[start:start + batch_size]]
            optimizer.zero_grad()
            loss, _logits, _ys, _mask = _transformer_batch(
                model, rows, symbol_id, output_symbols, class_weights=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            gradient_steps += 1
        presentations += len(train)
        if epoch == 1 or epoch % 5 == 0 or epoch == max_epochs:
            valid = _transformer_evaluate(model, validation, symbol_id, characters, output_symbols)
            trained = _transformer_evaluate(model, train, symbol_id, characters, output_symbols)
            score = (int(valid["exact"]), int(trained["exact"]),
                     float(valid["macro_token_accuracy"]), float(trained["macro_token_accuracy"]),
                     float(valid["token_accuracy"]), float(trained["token_accuracy"]))
            if score > best_score:
                best_score = score
                best_weights = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                stale = 0
            else:
                stale += 5
            if (valid["exact"] and trained["exact"]) or stale >= int(patience):
                break
    if best_weights is not None:
        model.load_state_dict(best_weights)
    valid = _transformer_evaluate(model, validation, symbol_id, characters, output_symbols)
    trained = _transformer_evaluate(model, train, symbol_id, characters, output_symbols)
    held = _transformer_evaluate(model, task["heldout"], symbol_id, characters, output_symbols)
    predictions = held.pop("predictions")
    valid.pop("predictions")
    trained.pop("predictions")
    correct = sum(prediction == expected for prediction, (_source, expected)
                  in zip(predictions, task["heldout"]))
    total = len(task["heldout"])
    seconds = time.perf_counter() - started
    return {
        "status": "fitted", "kind": "pretrained_mini_gpt" if provenance["pretrained"]
                  else "transformer_from_scratch",
        "family": task["family"], "modulus": task["modulus"],
        "seconds": seconds, "correct": correct, "total": total,
        "accuracy": correct / max(1, total), "exact": correct == total,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "layers": len(model.blocks), "heads": model.blocks[0].attn.n_heads,
        "d_model": model.d_model, "context_length": model.context_length,
        "epochs": epoch, "best_epoch": best_epoch, "gradient_steps": gradient_steps,
        "example_presentations": presentations,
        "example_budget_cap": int(example_budget),
        "budget": int(example_budget), "budget_unit": "example_presentations",
        "within_example_budget": presentations <= int(example_budget),
        "learning_rate": float(learning_rate), "batch_size": int(batch_size),
        "class_balance": bool(class_balance),
        "class_weights": {symbol: float(weights[index]) for index, symbol in enumerate(output_symbols)},
        "train_token_accuracy": trained["token_accuracy"],
        "train_exact_sequences": trained["exact_sequences"],
        "validation_token_accuracy": valid["token_accuracy"],
        "validation_macro_token_accuracy": valid["macro_token_accuracy"],
        "validation_exact_sequences": valid["exact_sequences"],
        "heldout_token_accuracy": held["token_accuracy"],
        "heldout_macro_token_accuracy": held["macro_token_accuracy"],
        "heldout_exact_sequences": held["exact_sequences"],
        "heldout_passed_to_learner": False,
        "train_sha256": _rows_hash(task["train"]),
        "validation_sha256": _rows_hash(task["validation"]),
        "heldout_sha256": _rows_hash(task["heldout"]),
        "causal_architecture": True, "causality_probe": _transformer_causality_probe(),
        **provenance,
        **_heldout_prediction_metrics(predictions, task["heldout"]),
    }


def transformer_representation_comparison(*, moduli=(12, 16, 24), train_n: int = 64,
                                          heldout_n: int = 64, max_epochs: int = 240,
                                          seed: int = MISSION_SEED,
                                          data_seed: int = MISSION_SEED,
                                          include_other_families: bool = True,
                                          pretrained_checkpoint: str = "") -> dict:
    """CCG ↔ sıfırdan Transformer ↔ isteğe bağlı pretrained MiniGPT.

    Transformer'lara CCG auto'nun üç adayının toplam örnek sunumu kadar üst
    sınır verilir. Böylece CCG'nin mimari seçim bütçesi gizlenip Transformer'a
    yalnız tek-aday bütçesi verilmez. Erken duran model daha az tüketebilir;
    hiçbiri üst sınırı aşamaz. Adım sayısı eşitlenebilir, FLOP ve duvar süresi
    eşit değildir; bunlar ayrı raporlanır.
    """
    specs = [("counter", int(modulus)) for modulus in moduli]
    if include_other_families:
        specs.extend((family, 12) for family in
                     ("reset", "affine", "ternary_reset", "ternary_weighted"))
    rows = []
    for index, (family, modulus) in enumerate(specs):
        task = _bounded_family_task(family, modulus, train_n=train_n,
                                    heldout_n=heldout_n, seed=data_seed)
        continuous, _learner = _continuous_measured_attempt(
            task, max_epochs=max_epochs, seed=seed + index,
            class_balance=True, causal_summary="auto")
        budget = int(continuous["example_presentations"])
        transformer = _transformer_measured_attempt(
            task, example_budget=budget, seed=seed + index,
            learning_rate=0.001, class_balance=True)
        row = {"family": family, "modulus": modulus,
               "train_examples": train_n, "heldout_examples": heldout_n,
               "continuous_causal_graph": continuous,
               "transformer_from_scratch": transformer}
        if pretrained_checkpoint:
            row["pretrained_mini_gpt"] = _transformer_measured_attempt(
                task, example_budget=budget, seed=seed + index,
                learning_rate=0.0003, class_balance=True,
                pretrained_checkpoint=pretrained_checkpoint)
        rows.append(row)
        message = (f"TR {family} mod{modulus}: CCG={continuous['correct']}/{continuous['total']} "
                   f"P={continuous['positive_correct']}/{continuous['positive_total']} | "
                   f"Transformer={transformer['correct']}/{transformer['total']} "
                   f"P={transformer['positive_correct']}/{transformer['positive_total']} "
                   f"steps={transformer['gradient_steps']}/{budget // 16}")
        if pretrained_checkpoint:
            pretrained = row["pretrained_mini_gpt"]
            message += (f" | pretrained={pretrained['correct']}/{pretrained['total']} "
                        f"P={pretrained['positive_correct']}/{pretrained['positive_total']}")
        print(message, flush=True)
    model_names = ["continuous_causal_graph", "transformer_from_scratch"]
    if pretrained_checkpoint:
        model_names.append("pretrained_mini_gpt")
    complete = all(row[model].get("status") and
                   row[model].get("positive_total", 0) > 0 and
                   row[model].get("negative_total", 0) > 0 and
                   row[model].get("within_example_budget", True) and
                   row[model].get("causality_probe", True)
                   for row in rows for model in model_names)
    leaders = []
    for row in rows:
        scores = {model: row[model]["accuracy"] for model in model_names}
        best = max(scores.values())
        leaders.append({"family": row["family"], "modulus": row["modulus"],
                        "heldout_leaders": sorted(model for model, score in scores.items()
                                                  if score == best),
                        "accuracies": scores})
    return {
        "protocol": {
            "same_task_data_for_all": True, "heldout_passed_to_learner": False,
            "heldout_used_for_checkpoint_or_budget": False,
            "train_examples": train_n, "validation_examples": 16,
            "heldout_examples": heldout_n,
            "budget_rule": "each Transformer <= CCG auto total example presentations",
            "gradient_step_batch_size": 16,
            "budgets_not_flops_or_seconds": True,
            "ccg_max_epochs_per_candidate": max_epochs,
            "transformer_learning_rate": 0.001,
            "pretrained_finetune_learning_rate": 0.0003,
            "data_seed": data_seed, "model_seed_base": seed,
            "causality_probe": _transformer_causality_probe(),
            "pretrained_checkpoint_supplied": bool(pretrained_checkpoint),
            "measurement_version": "v1-transformer-equal-example-step-cap",
        },
        "scale_and_ood": rows, "heldout_leaders_descriptive": leaders,
        "measurement_complete": bool(complete),
        "not_claimed": ["general_intelligence", "pretrained_tiny_gpt_is_general_llm",
                        "gradient_steps_equal_flops", "single_seed_universality",
                        "heldout_used_for_tuning"],
    }


def _fst_measured_attempt(task: dict, expansion_limit: int) -> dict:
    learner = _prior_learner()
    started = time.perf_counter()
    result = learner.invent_representation(
        task["train"], task["validation"], max_states=task["modulus"],
        max_expansions=expansion_limit,
    )
    seconds = time.perf_counter() - started
    name = str(result.get("id") or "")
    predictions = [learner.apply_representation(name, source) if name else None
                   for source, _expected in task["heldout"]]
    correct = sum(prediction == expected for prediction, (_source, expected)
                  in zip(predictions, task["heldout"]))
    total = len(task["heldout"])
    return {
        **{key: value for key, value in result.items() if key not in {"candidate_representations"}},
        "family": task["family"], "modulus": task["modulus"],
        "seconds": seconds, "correct": correct, "total": total,
        "accuracy": correct / max(1, total), "exact": bool(name) and correct == total,
        "transitions": (len(learner.representations[name].get("transitions", []))
                        if name else 0),
        "budget": int(expansion_limit), "budget_unit": "search_expansions",
        "train_sha256": _rows_hash(task["train"]),
        "validation_sha256": _rows_hash(task["validation"]),
        "heldout_sha256": _rows_hash(task["heldout"]),
        **_heldout_prediction_metrics(predictions, task["heldout"]),
    }


def continuous_representation_comparison(*, moduli=(12, 16, 24), train_n: int = 64,
                                         heldout_n: int = 64,
                                         expansion_limit: int = 1_000_000,
                                         max_epochs: int = 240,
                                         class_balance: bool = True,
                                         prefix_augmentation: int = 0,
                                         prefix_warmup_epochs: int = 0,
                                         checkpoint_selection: str = "train_first",
                                         causal_summary: bool | str = "auto") -> dict:
    """FST ve sürekli nedensel grafik: aynı OOD verisi, ayrı kaynak birimleri.

    ``measurement_complete`` yalnız bütün satırların gerçekten koştuğunu söyler;
    iki modelin de exact olması veya birinin üstün çıkması için eşik gevşetilmez.
    """
    specs = [("counter", int(modulus)) for modulus in moduli]
    # Mod12'de aynı görev biçimi dışındaki dört aile de ortak OOD sınavına girer.
    specs.extend((family, 12) for family in
                 ("reset", "affine", "ternary_reset", "ternary_weighted"))
    rows = []
    for index, (family, modulus) in enumerate(specs):
        task = _bounded_family_task(family, modulus, train_n=train_n,
                                    heldout_n=heldout_n)
        fst = _fst_measured_attempt(task, expansion_limit)
        continuous, _learner = _continuous_measured_attempt(
            task, max_epochs=max_epochs, seed=MISSION_SEED + index,
            class_balance=class_balance, prefix_augmentation=prefix_augmentation,
            prefix_warmup_epochs=prefix_warmup_epochs,
            checkpoint_selection=checkpoint_selection, causal_summary=causal_summary,
        )
        rows.append({"family": family, "modulus": modulus,
                     "train_examples": train_n, "heldout_examples": heldout_n,
                     "finite_state_transducer": fst,
                     "continuous_causal_graph": continuous})
        print(f"CCG {family} mod{modulus}: FST={fst['correct']}/{fst['total']} "
              f"P={fst['positive_correct']}/{fst['positive_total']} {fst.get('status')} | "
              f"sürekli={continuous['correct']}/{continuous['total']} "
              f"P={continuous['positive_correct']}/{continuous['positive_total']} "
              f"{continuous.get('status')}", flush=True)

    sample_efficiency = []
    modulus = int(moduli[0])
    for count in (8, 16, 32, 64):
        task = _bounded_family_task("counter", modulus, train_n=count,
                                    heldout_n=heldout_n, seed=MISSION_SEED + count)
        fst = _fst_measured_attempt(task, expansion_limit)
        continuous, _learner = _continuous_measured_attempt(
            task, max_epochs=max_epochs, seed=MISSION_SEED + count,
            class_balance=class_balance, prefix_augmentation=prefix_augmentation,
            prefix_warmup_epochs=prefix_warmup_epochs,
            checkpoint_selection=checkpoint_selection, causal_summary=causal_summary,
        )
        sample_efficiency.append({"train_examples": count,
                                  "finite_state_transducer": fst,
                                  "continuous_causal_graph": continuous})
        print(f"CCG örnek={count}: FST={fst['correct']}/{fst['total']} "
              f"P={fst['positive_correct']}/{fst['positive_total']} | "
              f"sürekli={continuous['correct']}/{continuous['total']} "
              f"P={continuous['positive_correct']}/{continuous['positive_total']}", flush=True)

    complete = all(row[model].get("status") and
                   row[model].get("positive_total", 0) > 0 and
                   row[model].get("negative_total", 0) > 0 and
                   row[model].get("ablation_passed", True) and
                   row[model].get("roundtrip_exact_predictions", True) and
                   row[model].get("production_gate_passed", True)
                   for row in [*rows, *sample_efficiency] for model in
                   ("finite_state_transducer", "continuous_causal_graph"))
    winners = []
    for row in rows:
        a = row["finite_state_transducer"]["accuracy"]
        b = row["continuous_causal_graph"]["accuracy"]
        winners.append({"family": row["family"], "modulus": row["modulus"],
                        "heldout_leader": "tie" if a == b else
                        ("finite_state_transducer" if a > b else "continuous_causal_graph")})
    return {
        "protocol": {
            "same_task_data_for_both": True, "heldout_passed_to_learner": False,
            "oracle_passed_to_learner": False, "train_examples": train_n,
            "validation_examples": 16, "heldout_examples": heldout_n,
            "fst_expansion_budget": expansion_limit,
            "fst_expansion_budget_scope": "her arama evresi; raporlanan expansions evrelerin toplamı",
            "continuous_epoch_budget": max_epochs,
            "continuous_class_balance": bool(class_balance),
            "continuous_prefix_augmentation_per_train_row": int(prefix_augmentation),
            "continuous_prefix_warmup_epochs": int(prefix_warmup_epochs),
            "continuous_checkpoint_selection": str(checkpoint_selection),
            "continuous_causal_summary": (str(causal_summary)
                                           if isinstance(causal_summary, str)
                                           else bool(causal_summary)),
            "prefix_augmentation_source": "train-only expected-prefix; zero initial state",
            "measurement_version": "v2-length-matched-stratified",
            "split_stratification": "yaklaşık 1/4 pozitif-olay içeren, kalanı tamamen negatif diziler",
            "length_matched_within_split": True,
            "positive_negative_and_majority_baseline_reported": True,
            "budgets_not_numerically_interchangeable": True,
        },
        "scale_and_ood": rows, "sample_efficiency_counter": sample_efficiency,
        "heldout_leaders_descriptive": winners,
        "measurement_complete": bool(complete),
        "not_claimed": ["general_intelligence", "phenomenal_consciousness",
                        "continuous_always_better", "fst_always_better",
                        "gradient_steps_equal_search_expansions"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--mission", action="store_true", help="kısıtlı-ağ otonom icat görevi")
    mode.add_argument("--scaling", action="store_true", help="durum ve zorluk ölçeklemesi")
    mode.add_argument("--generalization", action="store_true", help="mod7/8, OOD ve seçim-prior ölçümü")
    mode.add_argument("--active-generalization", action="store_true",
                      help="ayrı aktif-oracle OOD sorgu ve conformance ölçümü")
    mode.add_argument("--oracle-robustness", action="store_true",
                      help="maliyetli, gürültülü ve erişilemez oracle ölçümü")
    mode.add_argument("--correlated-oracle", action="store_true",
                      help="12 OOD görevinde korele aile sapması ve oy stratejileri")
    mode.add_argument("--adaptive-oracle", action="store_true",
                      help="uniform ve sıralı oylamanın bağımsız gürültüde maliyet karşılaştırması")
    mode.add_argument("--dual-oracle", action="store_true",
                      help="bağımsız ikinci oracle ile acceptance ve uyuşmazlık ölçümü")
    mode.add_argument("--continuous-comparison", action="store_true",
                      help="mod12/16/24 ve OOD: FST ↔ sürekli nedensel grafik")
    mode.add_argument("--transformer-comparison", action="store_true",
                      help="CCG ↔ eşit örnek/adım bütçeli Transformer ↔ isteğe bağlı pretrained GPT")
    mode.add_argument("--audit", action="store_true", help="terfi checkpoint'ine yeni mühürlü denetim")
    ap.add_argument("--offline", action="store_true", help="görevde ağı kapalı tut")
    ap.add_argument("--output", default="")
    ap.add_argument("--pretrained-checkpoint", default="",
                    help="varsa indirilen MiniGPT tiny_gpt.pt; strict yükleme ve aynı ince-ayar bütçesi")
    ap.add_argument("--checkpoint", default="/tmp/nora_autonomous_skill.json")
    args = ap.parse_args()
    if (args.mission or args.scaling or args.generalization or args.active_generalization
            or args.oracle_robustness or args.correlated_oracle
            or args.adaptive_oracle or args.dual_oracle or args.continuous_comparison
            or args.transformer_comparison or args.audit):
        started = time.perf_counter()
        if args.mission:
            report = autonomous_invention_mission(network=not args.offline,
                                                  checkpoint=args.checkpoint)
            output = args.output or "/tmp/nora_autonomous_mission.json"
            success = bool(report.get("success"))
        elif args.scaling:
            report = representation_scaling()
            output = args.output or "/tmp/nora_representation_scaling.json"
            success = bool(report["same_task_state_budget"][-1]["exact"]
                           and all(row["exact"] for row in report["task_family_difficulty"]))
        elif args.generalization:
            report = representation_generalization()
            output = args.output or "/tmp/nora_representation_generalization.json"
            success = bool(report["measurement_complete"])
        elif args.active_generalization:
            report = active_representation_generalization()
            output = args.output or "/tmp/nora_active_generalization.json"
            success = bool(report["measurement_complete"])
        elif args.oracle_robustness:
            report = active_oracle_robustness()
            output = args.output or "/tmp/nora_oracle_robustness.json"
            success = bool(report["measurement_complete"])
        elif args.correlated_oracle:
            report = correlated_oracle_robustness()
            output = args.output or "/tmp/nora_correlated_oracle.json"
            success = bool(report["measurement_complete"])
        elif args.adaptive_oracle:
            report = adaptive_oracle_cost_comparison()
            output = args.output or "/tmp/nora_adaptive_oracle.json"
            success = bool(report["measurement_complete"])
        elif args.dual_oracle:
            report = dual_oracle_acceptance()
            output = args.output or "/tmp/nora_dual_oracle.json"
            success = bool(report["measurement_complete"])
        elif args.continuous_comparison:
            report = continuous_representation_comparison()
            output = args.output or "/tmp/nora_continuous_comparison.json"
            success = bool(report["measurement_complete"])
        elif args.transformer_comparison:
            report = transformer_representation_comparison(
                pretrained_checkpoint=args.pretrained_checkpoint)
            output = args.output or "/tmp/nora_transformer_comparison.json"
            success = bool(report["measurement_complete"])
        else:
            report = sealed_checkpoint_audit(args.checkpoint)
            output = args.output or "/tmp/nora_final_sealed_audit.json"
            success = bool(report.get("exact") and report.get("sources_not_in_any_prior_split"))
        report["total_seconds"] = time.perf_counter() - started
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
        label = ("ÖLÇÜM TAMAM" if (args.generalization or args.active_generalization
                                    or args.oracle_robustness or args.correlated_oracle
                                    or args.adaptive_oracle or args.dual_oracle
                                    or args.continuous_comparison or args.transformer_comparison) and success
                 else ("GEÇTİ" if success else "KALDI"))
        print(f"KANIT {output} | {label}", flush=True)
        raise SystemExit(0 if success else 1)
    tests = [("Y1 koşullu ifade yeni örneklerde", conditional_expression),
             ("Y2 sembolik yasa → bağımsız plan", learned_law_plans),
             ("Y3 lineer ablasyon karşılaştırması", linear_ablation),
             ("Y4 audit reddi + atomik fit", audit_rejects_and_atomic_fit),
             ("Y5 Nora induce→plan→checkpoint", nora_induce_plan_checkpoint),
             ("Y6 gerektiğinde gizli-durum temsili açma", latent_representation_opens_only_when_needed),
             ("Y7 kalıcı temsil kişisel istekte çalışır", learned_representation_personal_application),
             ("Y8 kısıtlı araştırma ağ sınırı", restricted_research_boundary),
             ("Y9 teknik niyet ve bibliyografik kanıt", technical_epistemic_boundary),
             ("Y10 kanonik etiketleme ve seçim ablasyonu", selection_and_labeling_audit),
             ("Y11 aktif sorgu ve conformance protokolü", active_query_protocol_audit),
             ("Y12 oracle maliyeti, gürültü ve abstention", active_oracle_boundary_audit),
             ("Y13 sıralı oy ve korele oracle güveni", adaptive_oracle_boundary_audit),
             ("Y14 adaptive yanlış-ret aday kurtarma", adaptive_candidate_recovery_audit),
             ("Y15 bağımsız ikinci oracle acceptance", dual_oracle_boundary_audit)]
    failed = 0
    t0 = time.time()
    for _ in range(max(1, args.repeat)):
        for name, fn in tests:
            try:
                ok, detail = fn()
            except Exception as exc:
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            failed += int(not ok)
            print(f"{name}: {'GEÇTİ' if ok else 'KALDI'} | {detail}", flush=True)
    total = len(tests) * max(1, args.repeat)
    print(f"SONUÇ {total-failed}/{total} | {time.time()-t0:.1f} sn")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

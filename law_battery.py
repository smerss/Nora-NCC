#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""law_battery.py — yeni koşullu temsilin held-out ve planlama sınavı."""
from __future__ import annotations

import argparse
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


def _table_representation(table, alphabet) -> dict:
    states = 1 + max([state for state, _symbol in table]
                     + [target for target, _output in table.values()])
    return {
        "kind": "finite_state_transducer", "states": states,
        "alphabet": list(alphabet), "initial": 0,
        "transitions": [[state, symbol, target, output]
                        for (state, symbol), (target, output) in sorted(table.items())],
    }


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


def _conformance_probes(candidate, alphabet: str, lengths, max_states: int) -> list[str]:
    """Adaydan türetilen W-method ve ek-durum conformance sondaları.

    ``k=max_states-learned_states`` için durum kapsama yolları ile W ekleri
    arasına 0..k+1 genişliğinde orta parçalar konur. Uzunluk doldurma sondanın
    ayırt edici önekini değiştirmez ve girdiyi güvenli aktif alana taşır.
    """
    if isinstance(candidate, dict) and "transitions" in candidate:
        table = {(int(state), symbol): (int(target), output)
                 for state, symbol, target, output in candidate["transitions"]}
    else:
        table = candidate
    states = sorted({state for state, _symbol in table}
                    | {target for target, _output in table.values()})
    cover = {0: ""}
    queue = collections.deque([0])
    while queue:
        state = queue.popleft()
        for symbol in alphabet:
            target = table[(state, symbol)][0]
            if target not in cover:
                cover[target] = cover[state] + symbol
                queue.append(target)
    characterization = {""}
    for left, right in itertools.combinations(states, 2):
        queue = collections.deque([((left, right), "")])
        seen = {(left, right)}
        suffix = None
        while queue and suffix is None:
            (a_state, b_state), path = queue.popleft()
            for symbol in alphabet:
                next_a, output_a = table[(a_state, symbol)]
                next_b, output_b = table[(b_state, symbol)]
                if output_a != output_b:
                    suffix = path + symbol
                    break
                pair = (next_a, next_b)
                if pair not in seen:
                    seen.add(pair)
                    queue.append((pair, path + symbol))
        if suffix is None:
            raise ValueError("non-minimal-candidate")
        characterization.add(suffix)
    extra = max(0, int(max_states) - len(states))
    middles = ["".join(row) for width in range(extra + 2)
               for row in itertools.product(alphabet, repeat=width)]
    probes = {}
    allowed = sorted(set(map(int, lengths)))
    for state in states:
        for middle in middles:
            for suffix in characterization:
                base = cover[state] + middle + suffix
                target_lengths = [length for length in allowed if length >= len(base)]
                if not target_lengths:
                    continue
                target_length = target_lengths[0]
                source = base + alphabet[0] * (target_length - len(base))
                row = (len(middle), len(suffix), len(base), base, source)
                previous = probes.get(source)
                if previous is None or row[:4] > previous[:4]:
                    probes[source] = row
    ordered = sorted(probes.values(), key=lambda row: (-row[0], -row[1], -row[2], row[3]))
    return [row[4] for row in ordered]


def _active_representation(train, validation, pool, oracle, *, max_states: int,
                           alphabet: str, lengths, budget: int = ACTIVE_QUERY_BUDGET):
    """Aday ayrımı + conformance + bağımsız oracle ile aktif FST öğrenimi.

    Bu fonksiyona held-out hiç verilmez. Oracle yalnız seçilen girdi için
    çağrılır; disagreement cevapları mevcut exhaustive küme üzerinde sıfır
    expansion'lı filtre uygular. Minimal küme boşalırsa durum alt sınırı artar.
    """
    train, validation = list(train), list(validation)
    used = set()
    queries, syntheses, filters = [], [], []
    selected = candidates = None
    last_key = None
    verification_queue = []
    minimum_states = 1
    recovery_reference = None
    recovery_attempted = set()
    stop = "budget"
    while True:
        if candidates is None:
            learner = _prior_learner()
            result = learner.invent_representation(
                train, validation, min_states=minimum_states, max_states=max_states,
                return_candidates=True, allow_dfs_fallback=False,
            )
            tables = result.pop("candidate_tables", None)
            representations = result.pop("candidate_representations", None)
            syntheses.append(result)
            if result.get("status") == "invented":
                selected = learner.representations[result["id"]]
                candidates = [selected]
            elif result.get("status") == "ambiguous":
                selected = None
                candidates = tables if tables is not None else (representations or [])
            else:
                status = str(result.get("status"))
                recovery_key = None
                reference = None
                if recovery_reference is not None:
                    reference = (recovery_reference if "transitions" in recovery_reference
                                 else _table_representation(recovery_reference, alphabet))
                    recovery_key = (minimum_states, json.dumps(
                        reference["transitions"], sort_keys=True,
                    ))
                if (status in {"budget-exceeded", "unresolved"} and minimum_states > 1
                        and recovery_key is not None and recovery_key not in recovery_attempted
                        and len(queries) < int(budget)):
                    recovery_attempted.add(recovery_key)
                    probes = _conformance_probes(reference, alphabet, lengths, max_states)
                    extra = [source for source in probes if source not in used]
                    extra = extra[:ACTIVE_RECOVERY_QUERIES]
                    if extra:
                        for source in extra:
                            if len(queries) >= int(budget):
                                break
                            prediction = _run_finite_candidate(reference, source)
                            expected = oracle(source)
                            queries.append({
                                "source": source, "output": expected,
                                "reason": "recovery-conformance", "candidates_before": 0,
                                "prediction": prediction,
                                "prediction_matched": prediction == expected,
                            })
                            used.add(source)
                            validation.append((source, expected))
                        candidates = selected = None
                        continue
                stop = "synthesis-" + status
                break
            if not candidates:
                stop = "no-candidates"
                break
        if candidates:
            recovery_reference = candidates[0]
        if len(candidates) == 1:
            if "transitions" not in candidates[0]:
                selected = _table_representation(candidates[0], alphabet)
                candidates = [selected]
            else:
                selected = candidates[0]
            key = json.dumps(selected.get("transitions", []), sort_keys=True)
            if key != last_key:
                verification_queue = _conformance_probes(
                    selected, alphabet, lengths, max_states,
                )
                last_key = key
        else:
            selected = last_key = None
            verification_queue = []
        verification_mismatch = False
        while len(candidates) == 1 and len(queries) < int(budget):
            unused = [source for source in verification_queue if source not in used]
            if not unused:
                stop = "unique-conformance-complete"
                return selected, {
                    "stop": stop, "queries": queries, "syntheses": syntheses,
                    "filters": filters, "train_rows": len(train),
                    "validation_rows": len(validation),
                    "conformance_probe_count": len(verification_queue),
                }
            source = unused[0]
            prediction = _run_finite_candidate(selected, source)
            expected = oracle(source)
            queries.append({
                "source": source, "output": expected, "reason": "conformance",
                "candidates_before": 1, "prediction": prediction,
                "prediction_matched": prediction == expected,
            })
            used.add(source)
            validation.append((source, expected))
            if prediction != expected:
                verification_mismatch = True
                break
        if verification_mismatch:
            minimum_states = max(minimum_states, int(selected["states"]) + 1)
            candidates = selected = None
            if minimum_states > int(max_states):
                stop = "state-cap"
                break
            if len(queries) >= int(budget):
                stop = "budget"
                break
            continue
        if len(candidates) == 1:
            stop = "budget" if len(queries) >= int(budget) else "verification-reduced"
            break
        if len(queries) >= int(budget):
            stop = "budget"
            break
        unused = [source for source in pool if source not in used]
        if not unused:
            stop = "pool-exhausted"
            break
        count = min(ACTIVE_CANDIDATE_SAMPLE, len(candidates))
        indices = {0, len(candidates) - 1}
        if count > 2:
            indices.update((i * (len(candidates) - 1)) // (count - 1)
                           for i in range(count))
        sample = [candidates[index] for index in sorted(indices)]
        best = None
        # En kısa güvenli katmanda ayrım varken daha uzun katmana çıkma.
        for length in lengths:
            tier = [source for source in unused if len(source) == int(length)]
            for source in tier:
                outputs = [_run_finite_candidate(candidate, source) for candidate in sample]
                counts = collections.Counter(outputs)
                distinct = len(counts)
                if distinct <= 1:
                    continue
                entropy = -sum((number / len(outputs)) * math.log(number / len(outputs))
                               for number in counts.values())
                score = (distinct, entropy)
                if best is None or score > best[0]:
                    best = (score, source, distinct, entropy)
            if best is not None:
                break
        if best is None:
            stop = "no-pool-disagreement"
            break
        _score, source, distinct, entropy = best
        expected = oracle(source)
        before = len(candidates)
        candidates = [candidate for candidate in candidates
                      if _run_finite_candidate(candidate, source) == expected]
        filters.append({"before": before, "after": len(candidates), "expansions": 0})
        queries.append({
            "source": source, "output": expected, "reason": "disagreement",
            "candidates_before": before, "candidates_after": len(candidates),
            "sample_distinct_outputs": distinct, "sample_entropy": entropy,
            "prediction": None, "prediction_matched": None,
        })
        used.add(source)
        validation.append((source, expected))
        if not candidates:
            minimum_states = max(minimum_states, int(result.get("states", max_states)) + 1)
            candidates = None
            if minimum_states > int(max_states):
                stop = "state-cap"
                break
    return selected, {
        "stop": stop, "queries": queries, "syntheses": syntheses,
        "filters": filters, "train_rows": len(train),
        "validation_rows": len(validation),
        "conformance_probe_count": len(verification_queue),
    }


def _active_measure(family: str, modulus: int,
                    budget: int = ACTIVE_QUERY_BUDGET) -> dict:
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
        lambda source: _family_oracle(source, modulus, family),
        max_states=modulus, alphabet=alphabet, lengths=lengths, budget=budget,
    )
    # Bu noktadan önce held-out, sorgu seçicinin fonksiyon kapsamına hiç girmez.
    correct = (sum(_run_finite_candidate(selected, source) == expected
                   for source, expected in task["heldout"])
               if selected is not None else 0)
    queries = detail["queries"]
    if (any(query["source"] in original_sources for query in queries)
            or any(len(query["source"]) not in lengths for query in queries)
            or len({query["source"] for query in queries}) != len(queries)):
        raise AssertionError("active_query_protocol_violation")
    total = len(task["heldout"])
    reason_counts = collections.Counter(query["reason"] for query in queries)
    return {
        "family": family, "modulus": int(modulus), "status": detail["stop"],
        "resolved": selected is not None, "correct": correct, "total": total,
        "accuracy": correct / total, "exact": correct == total,
        "seconds": time.perf_counter() - started,
        "query_count": len(queries), "oracle_calls": len(queries),
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
            "baseline_comparison": "run representation_generalization separately; frozen rows are not rewritten",
        },
        "rows": rows,
        "findings": {
            "exact_rows": sum(row["exact"] for row in rows), "total_rows": len(rows),
            "nonexact": [f"{row['family']}-{row['modulus']}:{row['status']}"
                         for row in rows if not row["exact"]],
            "total_queries": sum(row["query_count"] for row in rows),
            "total_disagreement_queries": sum(row["disagreement_queries"] for row in rows),
            "total_conformance_queries": sum(row["conformance_queries"] for row in rows),
            "total_recovery_queries": sum(row["recovery_queries"] for row in rows),
            "all_oracle_calls_accounted": all(
                row["oracle_calls"] == row["query_count"] for row in rows
            ),
        },
        "measurement_complete": complete,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--mission", action="store_true", help="kısıtlı-ağ otonom icat görevi")
    mode.add_argument("--scaling", action="store_true", help="durum ve zorluk ölçeklemesi")
    mode.add_argument("--generalization", action="store_true", help="mod7/8, OOD ve seçim-prior ölçümü")
    mode.add_argument("--active-generalization", action="store_true",
                      help="ayrı aktif-oracle OOD sorgu ve conformance ölçümü")
    mode.add_argument("--audit", action="store_true", help="terfi checkpoint'ine yeni mühürlü denetim")
    ap.add_argument("--offline", action="store_true", help="görevde ağı kapalı tut")
    ap.add_argument("--output", default="")
    ap.add_argument("--checkpoint", default="/tmp/nora_autonomous_skill.json")
    args = ap.parse_args()
    if args.mission or args.scaling or args.generalization or args.active_generalization or args.audit:
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
        else:
            report = sealed_checkpoint_audit(args.checkpoint)
            output = args.output or "/tmp/nora_final_sealed_audit.json"
            success = bool(report.get("exact") and report.get("sources_not_in_any_prior_split"))
        report["total_seconds"] = time.perf_counter() - started
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
        label = ("ÖLÇÜM TAMAM" if (args.generalization or args.active_generalization) and success
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
             ("Y11 aktif sorgu ve conformance protokolü", active_query_protocol_audit)]
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

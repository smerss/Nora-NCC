#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""law_battery.py — yeni koşullu temsilin held-out ve planlama sınavı."""
from __future__ import annotations

import argparse
import base64
import hashlib
import itertools
import json
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
    ok = (learned.get("status") == "invented" and result.get("ok")
          and result.get("verified") and result.get("stdout") == expected
          and not unknown.get("ok") and unknown.get("reason") == "representation_input_gap"
          and anaphoric_ok)
    detail = (f"öğrenme={learned.get('status')} yol={result.get('path')} "
              f"stdout={result.get('stdout')!r} beklenen={expected!r} "
              f"doğrulama={result.get('verified')} alfabe-dışı={unknown.get('reason')} "
              f"AGI-anaför={anaphoric_ok}")
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


def _rows_hash(rows) -> str:
    raw = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _prior_learner():
    """Hedefe özel belge/tablo değil, yalnız önceki genel akış ilkelleri."""
    from nora_semantic_code import CompositionalCodeLearner
    learner = CompositionalCodeLearner()
    learner.operators.update({"bind", "iterate", "concatenate"})
    return learner


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
    for modulus in range(2, 7):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--mission", action="store_true", help="kısıtlı-ağ otonom icat görevi")
    mode.add_argument("--scaling", action="store_true", help="durum ve zorluk ölçeklemesi")
    mode.add_argument("--audit", action="store_true", help="terfi checkpoint'ine yeni mühürlü denetim")
    ap.add_argument("--offline", action="store_true", help="görevde ağı kapalı tut")
    ap.add_argument("--output", default="")
    ap.add_argument("--checkpoint", default="/tmp/nora_autonomous_skill.json")
    args = ap.parse_args()
    if args.mission or args.scaling or args.audit:
        started = time.perf_counter()
        if args.mission:
            report = autonomous_invention_mission(network=not args.offline,
                                                  checkpoint=args.checkpoint)
            output = args.output or "/tmp/nora_autonomous_mission.json"
            success = bool(report.get("success"))
        elif args.scaling:
            report = representation_scaling()
            output = args.output or "/tmp/nora_representation_scaling.json"
            success = bool(report["same_task_state_budget"][-1]["exact"])
        else:
            report = sealed_checkpoint_audit(args.checkpoint)
            output = args.output or "/tmp/nora_final_sealed_audit.json"
            success = bool(report.get("exact") and report.get("sources_not_in_any_prior_split"))
        report["total_seconds"] = time.perf_counter() - started
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
        print(f"KANIT {output} | {'GEÇTİ' if success else 'KALDI'}", flush=True)
        raise SystemExit(0 if success else 1)
    tests = [("Y1 koşullu ifade yeni örneklerde", conditional_expression),
             ("Y2 sembolik yasa → bağımsız plan", learned_law_plans),
             ("Y3 lineer ablasyon karşılaştırması", linear_ablation),
             ("Y4 audit reddi + atomik fit", audit_rejects_and_atomic_fit),
             ("Y5 Nora induce→plan→checkpoint", nora_induce_plan_checkpoint),
             ("Y6 gerektiğinde gizli-durum temsili açma", latent_representation_opens_only_when_needed),
             ("Y7 kalıcı temsil kişisel istekte çalışır", learned_representation_personal_application),
             ("Y8 kısıtlı araştırma ağ sınırı", restricted_research_boundary)]
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

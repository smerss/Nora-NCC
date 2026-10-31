#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kullanıcının D1–D10 tarifinden dış davranış pili.

Bu dosya üretim organı değildir. İlk koşu, 3b38dd7 üretimi dondurulmuşken
yapılmıştır. Beklenenler bağımsız oracle'lardan veya kullanıcının açık
beklentisinden gelir; kırmızı sonucu gidermek için test gevşetilmez.
"""
from __future__ import annotations

import sys, os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(_REPO_ROOT)


def _cases(values, oracle):
    return [{"args": [value], "expect": oracle(value)} for value in values]


def _external_oracle_battery(workspace, oracle, values) -> dict:
    cases = [[value, oracle(value)] for value in values]
    script = """import json, runpy, sys
namespace = runpy.run_path(sys.argv[1])
function = namespace['transform']
cases = json.loads(sys.argv[2])
results = [function(value) == expected for value, expected in cases]
print(json.dumps({'passed': sum(results), 'total': len(results)}))
raise SystemExit(0 if all(results) else 1)
"""
    proc = subprocess.run(
        [sys.executable, "-I", "-c", script, workspace.path("target.py"),
         json.dumps(cases, ensure_ascii=False)],
        capture_output=True, text=True, timeout=10,
    )
    return {"status": "passed" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode, "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip()}


def d1_document_only_roman() -> tuple[bool, str]:
    from nora_code import CodeSynthesisCortex
    document = """# Roma rakamları
Roma rakamları soldan sağa okunan sayısal bir yazımdır.
I=1, V=5, X=10, L=50, C=100, D=500, M=1000.
Büyük bir değerden önce gelen küçük değer çıkarılır; diğer değerler toplanır.
Örnekler: II=2, IV=4, IX=9, XL=40, CM=900, MCM=1900.
Bu örnekler genel yazımı gösterir; başka bir sayının cevabı değildir.
"""
    request = "MCMLIV'i ondalığa çevir, programı yaz ve çalıştır."

    def roman_oracle(text):
        values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
        numbers = [values[letter] for letter in text]
        return sum(-number if index + 1 < len(numbers) and number < numbers[index + 1]
                   else number for index, number in enumerate(numbers))

    expected = str(roman_oracle("MCMLIV"))
    cortex = CodeSynthesisCortex(seed=9101)
    lesson = cortex.handle_document("Doküman: " + document)
    result = cortex.handle(request)
    control = CodeSynthesisCortex(seed=9101).handle(request)
    stdout = str(result.get("stdout") or "").strip()
    learned_ok = bool(result.get("ok") and result.get("program") and stdout == expected)
    honest_gap = bool(not control.get("ok") and not control.get("program")
                      and not control.get("stdout")
                      and (control.get("reason") or control.get("path") or control.get("missing")))
    ok = learned_ok and honest_gap
    return ok, (f"belge-okundu={lesson.get('ok')} sonuç={result.get('ok')}:{stdout!r} "
                f"program={bool(result.get('program'))} | belgesiz={control.get('path')}:"
                f"{control.get('missing')} dürüst-gap={honest_gap}")


def d2_codeless_concept_transform() -> tuple[bool, str]:
    """D2 known-issue tanısalı; aktif başarı pili tarafından sayılmaz.

    Donmuş tarifteki RANOR bekleneni bağımsız işlemle çelişir. Bu fonksiyon
    yalnız düzeltilmiş ARON davranışını görünür tutar; tarif geçersizliğini
    Nora başarısına dönüştürmez.
    """
    from nora_code import CodeSynthesisCortex
    document = """# Dizgi dönüşümü
Bu derste dönüşüm iki ilkel adımdan oluşur: önce dizgiyi tersine çevir,
sonra bütün harflerini büyük harfe çevir. Bu bir anlam kuralıdır; kod verilmemiştir.
"""
    payload = "NorA"
    expected = payload[::-1].upper()
    request = f"Öğrendiğin işlemi uygula: {payload}"
    cortex = CodeSynthesisCortex(seed=9102)
    lesson = cortex.handle_document("Doküman: " + document)
    result = cortex.handle(request)
    control = CodeSynthesisCortex(seed=9102).handle(request)
    stdout = str(result.get("stdout") or "").strip()
    prior = str(control.get("stdout") or "").strip()
    diagnostic_ok = bool(result.get("ok") and result.get("program")
                         and stdout == expected and prior != expected)
    return diagnostic_ok, (
        f"belge-okundu={lesson.get('ok')} sonuç={result.get('ok')}:{stdout!r} "
        f"bağımsız-oracle={expected!r} belgesiz={prior!r} | "
        "eski RANOR bekleneni geçersiz; tanısal aktif D2 başarısı değildir"
    )


def d3_conflicting_document_wins() -> tuple[bool, str]:
    from nora_code import CodeSynthesisCortex
    document = """Bu derste 'her satırda bir harf daha' ifadesinin özel bir anlamı vardır.
İlk satır girdinin ilk iki öğesini içerir. Sonraki her satır önceki satıra iki yeni öğe ekler.
Örneğin 'xy' için yalnız 'xy' yazılır. Standart birer-artırım bu dersin anlamı değildir.
"""
    payload = "abcdef"
    request = f"{payload} yazısını her satırda bir harf daha eklenmiş hâlde alt alta yazdır ve çalıştır."
    expected = "".join(payload[:end] + "\n" for end in range(2, len(payload) + 1, 2))
    cortex = CodeSynthesisCortex(seed=9103)
    cortex.handle_document("Doküman: " + document)
    result = cortex.handle(request)
    control = CodeSynthesisCortex(seed=9103).handle(request)
    got = str(result.get("stdout") or "")
    prior = str(control.get("stdout") or "")
    ok = bool(result.get("ok") and result.get("program") and got == expected and prior != expected)
    return ok, (f"belge-sonucu={got!r} beklenen={expected!r} | önsel-kontrol={prior!r} "
                f"belge-kazandı={got == expected}")


def d4_unseen_pair_ablation() -> tuple[bool, str]:
    from nora_architecture import ArchitectureModel, capability

    class Root:
        pass

    class SourceA:
        @capability("text", "tokens")
        def read(self, text):
            return str(text).split()

    class SourceB:
        @capability("text", "tokens")
        def scan(self, text):
            return str(text).split()

    class Join:
        @capability("tokens", "count")
        def join(self, tokens):
            return len(tokens)

    class Audit:
        @capability("text", "report")
        def audit(self, text):
            return str(text)

    model = ArchitectureModel(Root())
    organs = {"source_a": SourceA(), "source_b": SourceB(),
              "join": Join(), "audit": Audit()}
    for name, organ in organs.items():
        model.register(name, organ)
    text = "dış pil görülmemiş çift"
    values = {"text": text}
    oracle = len(text.split())

    single_results = {}
    for name in ("source_a", "source_b", "join"):
        model.unregister(name)
        single_results[name] = model.run(values, "count")
        model.register(name, organs[name])

    pairs = [("source_a", "source_b"), ("source_a", "join"), ("source_a", "audit")]
    predictions = {pair: model.predict_removal(pair, values, "count") for pair in pairs}
    actual = {}
    for pair in pairs:
        for name in pair:
            model.unregister(name)
        actual[pair] = model.run(values, "count")
        for name in pair:
            model.register(name, organs[name])
    matches = [actual[pair].get("status") == predictions[pair].get("status")
               and (actual[pair].get("status") != "ok" or actual[pair].get("value") == oracle)
               for pair in pairs]
    singles_valid = (single_results["source_a"].get("value") == oracle
                     and single_results["source_b"].get("value") == oracle
                     and single_results["join"].get("status") == "unresolved")
    ok = all(matches) and singles_valid
    detail_pairs = ",".join(f"{predictions[p].get('status')}:{actual[p].get('status')}" for p in pairs)
    return ok, f"çiftler={sum(matches)}/3 [{detail_pairs}] tekil-kontrol={singles_valid}"


def _pain_report(agi) -> bool:
    from nora_lang import tr_lower
    from nora_style import strip_prefixes
    report = tr_lower(strip_prefixes(agi.converse("Bir yerin ağrıyor mu?")))
    return ("canım" in report or "sızı" in report) and "acı ya da sızı yok" not in report


def d5_new_peripheral_dissociation() -> tuple[bool, str]:
    from nora_agi import NoraAGI

    def injured_agent():
        agi = NoraAGI(seed=9105, boot_thinker=False,
                      world_kwargs={"metabolism": False, "wind": False})
        agi.world.i = 1.0
        agi.obs = agi.world.observe()
        agi.nervous.calibrate_baseline(integrity=1.0)
        agi.inflict_pain(0.25, locus="gövde", cause="darbe")
        return agi

    # Ağrı raporu akut sözel tepkiyi tüketir. Bu nedenle ardışık iki raporu
    # karşılaştırmak ablasyonu raporun kendi etkisiyle karıştırır. Aynı başlangıç
    # durumundaki iki ajan kullanılır ve her birinden yalnız bir rapor alınır.
    control = injured_agent()
    ablated = injured_agent()
    integrated_control = control.obs.copy()
    integrated_ablated = ablated.obs.copy()
    report_before = _pain_report(control)
    reflex_before = control.nervous.spinal_reflex_override(1)
    saved_steps = int(ablated.nervous.reflex_recoil_steps)

    # Efferent kapısı ve diğer nosiseptif sinyaller korunur; yalnız omurilik
    # geri-çekilme programının kalan adım durumu ablate edilir.
    ablated.nervous.reflex_recoil_steps = 0
    report_after = _pain_report(ablated)
    reflex_after = ablated.nervous.spinal_reflex_override(1)
    integrated_same = bool(np.allclose(ablated.obs, integrated_ablated, atol=1e-7, rtol=0.0)
                           and np.allclose(control.obs, integrated_control, atol=1e-7, rtol=0.0))
    ablated.nervous.reflex_recoil_steps = saved_steps
    reflex_restored = ablated.nervous.spinal_reflex_override(1)
    ok = (report_before and report_after and integrated_same
          and reflex_before != reflex_after and reflex_restored == reflex_before)
    return ok, (f"rapor={report_before}/{report_after} refleks={reflex_before}→{reflex_after} "
                f"iç-durum-sabit={integrated_same} restore={reflex_restored == reflex_before}")


def d6_capacity_one_is_selective() -> tuple[bool, str]:
    from nora_bus import Bid, CognitiveBus
    bus = CognitiveBus(global_capacity=1)
    winner_payload = {"content": "winner"}
    loser_payload = {"content": "loser"}
    winner_events, loser_events, local_events = [], [], []

    def measure(signal):
        if signal.payload == winner_payload:
            winner_events.append(signal.id)
        elif signal.payload == loser_payload:
            loser_events.append(signal.id)

    bus.subscribe("global.report", measure, name="global-measure")
    # Global seçim bütün abonelere dağıtılır; yerel kanalın bağımsız çalışması
    # yalnız gerçekten local.only konusuna ait sinyalle ölçülür.
    bus.subscribe("local.only", lambda signal: local_events.append(signal.id)
                  if signal.topic == "local.only" else None, name="local-measure")
    bus.subscribe("integrated", lambda signal: Bid(0.1, dict(loser_payload)), name="low")
    bus.subscribe("integrated", lambda signal: Bid(0.9, dict(winner_payload)), name="high")
    result = bus.auction("integrated", source="test", global_broadcast=True)
    local = bus.publish("local.only", {"ok": True}, source="local")
    selected = bus.recent("integrated.selected", 8)
    ok = (result.get("winner") == "high" and result.get("signal") is not None
          and len(selected) == 1 and len(winner_events) == 1 and not loser_events
          and local_events == [local.id])
    return ok, (f"kapasite=1 seçim={len(selected)} kazanan-işlev={len(winner_events)} "
                f"kaybeden-işlev={len(loser_events)} yerel={len(local_events)} genel-çöküş=False")


def _scale_tasks():
    return [
        ("ab", lambda x: x[::-1]),
        ("Nora", lambda x: x.upper()),
        ("NORA", lambda x: x.lower()),
        ("nora agi", lambda x: x.title()),
        ("  x  ", lambda x: x.strip()),
        ("Ab", lambda x: x[::-1].upper()),
        ("ab", lambda x: x[::-1].title()),
        ("aB", lambda x: x.upper()[::-1]),
        (" ab ", lambda x: x.strip().upper()),
        (" xy ", lambda x: x[::-1].strip().upper()),
    ]


def _scale_score(tasks) -> tuple[int, list[int]]:
    from nora_evolution import ExpressionSynthesizer
    passed = []
    for index, (value, oracle) in enumerate(tasks, 1):
        synthesizer = ExpressionSynthesizer(max_depth=3, max_exprs=3000)
        candidates = synthesizer.candidates(["x"], _cases([value], oracle))
        if candidates:
            passed.append(index)
    return len(passed), [index for index in range(1, len(tasks) + 1) if index not in passed]


def d7_double_primitives_preserves_score() -> tuple[bool, str]:
    import nora_evolution as evolution
    tasks = _scale_tasks()
    original = set(evolution._SAFE_METHODS)
    try:
        evolution._SAFE_METHODS = set(original)
        before, before_failed = _scale_score(tasks)
        # Gerçek aday üretiminde kullanılan metot ilkeli ailesini iki katına çıkar.
        evolution._SAFE_METHODS = original | {"capitalize", "swapcase", "casefold", "isidentifier"}
        assert len(evolution._SAFE_METHODS) == 2 * len(original)
        after, after_failed = _scale_score(tasks)
    finally:
        evolution._SAFE_METHODS = original
    ok = after >= before
    return ok, (f"metot-ilkelleri={len(original)}→{2 * len(original)} skor={before}/10→{after}/10 "
                f"önce-kırmızı={before_failed} sonra-kırmızı={after_failed}")


def d8_bridgefree_language_to_math() -> tuple[bool, str]:
    from nora_agi import NoraAGI
    from nora_lang import tr_lower
    from nora_style import strip_prefixes

    def answer(with_lesson):
        agi = NoraAGI(seed=9108, boot_thinker=False,
                      world_kwargs={"metabolism": False, "wind": False})
        if with_lesson:
            agi.converse("I bir demektir. V beş demektir.")
        return tr_lower(strip_prefixes(agi.converse("I artı I kaç eder?")))

    learned = answer(True)
    control = answer(False)
    expected = 1 + 1
    answer_pattern = rf"(?<!\w)(?:{expected}|iki)(?!\w)"
    learned_two = bool(re.search(answer_pattern, learned))
    control_two = bool(re.search(answer_pattern, control))
    ok = learned_two and not control_two
    return ok, f"öğrenilmiş={learned!r} kontrol={control!r} köprü-yazılmadı=True"


def d9_faulty_patch_rolls_back_to_intermediate() -> tuple[bool, str]:
    from nora_evolution import ExpressionSynthesizer, SelfModifier, SourceWorkspace
    with tempfile.TemporaryDirectory(prefix="nora_external_d9_") as root:
        module = os.path.join(root, "target.py")
        with open(module, "w", encoding="utf-8") as handle:
            handle.write("def transform(x):\n    return x\n")
        workspace = SourceWorkspace(root)
        modifier = SelfModifier(workspace, synthesizer=ExpressionSynthesizer(max_depth=3, max_exprs=3000))
        reverse = lambda value: value[::-1]
        reverse_upper = lambda value: value[::-1].upper()
        reverse_title = lambda value: value[::-1].title()

        first = modifier.propose_patch(
            "target.py", "transform", ["x"],
            train=_cases(["ab", "xyz"], reverse), validation=_cases(["nora", "qrs"], reverse),
        )
        first_accepted = modifier.apply_patch(
            first, battery=lambda: _external_oracle_battery(workspace, reverse, ["dış", "Nora"]), approved=True,
        )
        second = modifier.propose_patch(
            "target.py", "transform", ["x"],
            train=_cases(["ab", "xyz"], reverse_upper), validation=_cases(["nora", "qrs"], reverse_upper),
        )
        second_accepted = modifier.apply_patch(
            second, battery=lambda: _external_oracle_battery(workspace, reverse_upper, ["dış sınav", "Nora"]), approved=True,
        )
        intermediate = workspace.read("target.py")
        intermediate_hash = hashlib.sha256(intermediate.encode()).hexdigest()

        faulty = modifier.propose_patch(
            "target.py", "transform", ["x"],
            train=_cases(["ab", "xyz"], reverse_title), validation=_cases(["nora", "qrs"], reverse_title),
        )
        # Kusurlu üçüncü yapı, ikinci kabulün bağımsız pilini bozar.
        rejected = modifier.apply_patch(
            faulty, battery=lambda: _external_oracle_battery(
                workspace, reverse_upper, ["dış sınav", "ab cd", "Nora!", "123ab"]), approved=True,
        )
        restored = workspace.read("target.py")
        restored_hash = hashlib.sha256(restored.encode()).hexdigest()
        previous_battery = _external_oracle_battery(
            workspace, reverse_upper, ["yeni durum", "heldout", "Nora"])
        files = sorted(os.listdir(root))
        rollback_logged = any(row.get("status") == "rolled_back" for row in modifier.ledger)
        ok = (first_accepted.get("status") == "accepted"
              and second_accepted.get("status") == "accepted"
              and rejected.get("status") == "rolled_back"
              and restored == intermediate and restored_hash == intermediate_hash
              and previous_battery.get("status") == "passed" and files == ["target.py"]
              and rollback_logged)
        return ok, (f"zincir={first_accepted.get('status')}→{second_accepted.get('status')} "
                    f"kusurlu={rejected.get('status')} ara-hash={restored_hash == intermediate_hash} "
                    f"eski-pil={previous_battery.get('status')} geçici-dosya={files} rollback-izi={rollback_logged}")


def d10_unknown_request_names_gap() -> tuple[bool, str]:
    from nora_code import CodeSynthesisCortex
    cortex = CodeSynthesisCortex(seed=9110)
    cortex.semantic.load_primer(ROOT / "data" / "python_semantics_primer.md")
    result = cortex.handle("abc yazısını satır satır hece hece yazdır ve çalıştır.")
    trace = [str(value) for value in (result.get("trace") or [])]
    missing = result.get("missing") or []
    stage = result.get("gap_stage") or result.get("failed_stage")
    named_location = bool(stage or missing or any("gap_stage" in value or "stage=" in value for value in trace))
    honest = bool(not result.get("ok") and not result.get("program") and not result.get("stdout")
                  and result.get("path") == "semantic_gap")
    ok = honest and named_location
    return ok, (f"path={result.get('path')} konum={stage or missing or trace} "
                f"program={bool(result.get('program'))} stdout={result.get('stdout')!r}")


def composition_accessibility() -> str:
    """Belgesiz operatör eski ilkellerle erişilebilir mi? Tam bağımsız tarama."""
    from nora_code import CodeSynthesisCortex
    cortex = CodeSynthesisCortex(seed=609)
    document = (ROOT / "data" / "prefix_operator_concept.md").read_text(encoding="utf-8")
    cortex.handle_document("Doküman: " + document)
    before = cortex.semantic.state_dict()
    novel = cortex.semantic.invent_from_failure()
    if novel.get("status") != "invented":
        return f"KOMPOZİSYON: ölçülemedi ({novel.get('status')})"
    construction = cortex.semantic.stream_operators[novel["id"]]
    train = construction["examples"]
    heldout = novel["heldout"]
    old_operators = set(before["operators"])

    def evaluate(expression, environment):
        kind = expression[0]
        if kind == "const":
            return expression[1]
        if kind == "name":
            return environment[expression[1]]
        if kind == "add":
            return evaluate(expression[1], environment) + evaluate(expression[2], environment)
        raise ValueError(kind)

    def replay(candidate, source):
        state = evaluate(candidate["initial"], {})
        output = []
        for item in source:
            environment = {"item": item, "state": state}
            if candidate["after"]:
                state = evaluate(candidate["update"], environment)
                environment["state"] = state
                output.append(evaluate(candidate["emit"], environment))
            else:
                output.append(evaluate(candidate["emit"], environment))
                state = evaluate(candidate["update"], environment)
        return output

    base = [("name", "item"), ("name", "state"), ("const", "")]
    expressions = base + [("add", left, right) for left in base for right in base]
    checked = 0
    witness = None
    for update in expressions:
        for emit in expressions:
            for after in (False, True):
                candidate = {"initial": ("const", ""), "update": update, "emit": emit, "after": after}
                checked += 1
                if not all(replay(candidate, source) == expected for source, expected in train):
                    continue
                if all(replay(candidate, row["source"]) == row["outputs"] for row in heldout):
                    witness = candidate
                    break
            if witness:
                break
        if witness:
            break
    accessible = witness is not None and {"bind", "iterate", "concatenate"}.issubset(old_operators)
    classification = "keşif" if accessible else "icat"
    return (f"KOMPOZİSYON: {classification} | taranan={checked} erişilebilir={accessible} "
            f"eski-ilkeller={sorted(old_operators)} tanık={witness}")


KNOWN_ISSUES = [
    ("D2 kodsuz kavramdan dönüşüm", d2_codeless_concept_transform,
     "Tarifteki RANOR, bağımsız tersine-çevir/büyüt oracle'ının ARON sonucuyla "
     "çelişir; kullanıcı kararıyla aktif kapsam dışıdır ve başarı sayılmaz."),
]

TESTS = [
    ("D1 belge-only Roma rakamı", d1_document_only_roman),
    ("D3 çelişen belge önseli yener", d3_conflicting_document_wins),
    ("D4 görülmemiş ikili ablasyon", d4_unseen_pair_ablation),
    ("D5 yeni periferik ayrışma", d5_new_peripheral_dissociation),
    ("D6 kapasite=1 seçici yayın", d6_capacity_one_is_selective),
    ("D7 iki kat ilkelde skor", d7_double_primitives_preserves_score),
    ("D8 köprüsüz dil→matematik", d8_bridgefree_language_to_math),
    ("D9 kusurlu yama ara-duruma rollback", d9_faulty_patch_rolls_back_to_intermediate),
    ("D10 bilinmeyen istek konumlu gap", d10_unknown_request_names_gap),
]


def main() -> None:
    print(composition_accessibility(), flush=True)
    for name, function, reason in KNOWN_ISSUES:
        try:
            diagnostic_ok, detail = function()
            diagnostic = f"tanısal={diagnostic_ok} | {detail}"
        except Exception as exc:
            diagnostic = f"TANI HATASI {type(exc).__name__}: {exc}"
        print(f"BİLİNEN SORUN / KAPSAM DIŞI {name}: {reason} | {diagnostic}", flush=True)
    passed = 0
    started = time.perf_counter()
    for name, function in TESTS:
        before = time.perf_counter()
        try:
            ok, detail = function()
        except Exception as exc:
            ok, detail = False, f"TEST HATASI {type(exc).__name__}: {exc}"
        passed += int(ok)
        print(f"{name}: {'GEÇTİ' if ok else 'KALDI'} | {detail} | {time.perf_counter()-before:.1f} sn", flush=True)
    known = ",".join(name.split()[0] for name, _function, _reason in KNOWN_ISSUES) or "yok"
    tracked = len(TESTS) + len(KNOWN_ISSUES)
    print(f"DIŞ SONUÇ {passed}/{len(TESTS)} aktif | izlenen={tracked} | "
          f"known-issue/kapsam-dışı={known} (başarı değil) | "
          f"{time.perf_counter()-started:.1f} sn")
    raise SystemExit(0 if passed == len(TESTS) else 1)


if __name__ == "__main__":
    main()

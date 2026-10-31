#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Practical document-learning and compositional-code evidence.

The target request and expected output live only in this external battery.  The
production organ sees general documents and must infer a reusable plan.
"""
from __future__ import annotations

import sys, os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import nora  # noqa: F401  (eski nora_xxx import'ları için uyumluluk kancası)


import os
from pathlib import Path
import sys
import tempfile
import time

from nora_agi import NoraAGI
from nora_code import CodeSynthesisCortex
from nora_style import strip_prefixes


ROOT = Path(_REPO_ROOT)
PRIMER = ROOT / "data" / "python_semantics_primer.md"
ACCUMULATION = ROOT / "data" / "python_accumulation.md"
PREFIX_CONCEPT = ROOT / "data" / "prefix_operator_concept.md"
CUMULATIVE_TARGET = ("selamlar dünyalılar! yazısını her satırda bir harf daha "
                     "eklenmiş hâlde alt alta yazdır ve çalıştır.")
CUMULATIVE_PAYLOAD = "selamlar dünyalılar!"
TARGET = ("Merhaba Baran! yazısını alt satırlara doğru giderek harf harf "
          "nasıl yazdırılacağını iste. Kodu ver ve kodu çalıştır.")
PAYLOAD = "Merhaba Baran!"
CPYTHON_DOCS = (
    "https://api.github.com/repos/python/cpython/contents/Doc/tutorial/controlflow.rst?ref=3.14",
    "https://api.github.com/repos/python/cpython/contents/Doc/library/functions.rst?ref=3.12",
    "https://api.github.com/repos/python/cpython/contents/Doc/library/stdtypes.rst?ref=3.12",
)


def expected(text: str, unit: str) -> str:
    units = list(text) if unit == "character" else text.split()
    return "\n".join(units) + "\n"


def check_result(result: dict, text: str, unit: str) -> bool:
    return bool(result.get("ok") and result.get("verified")
                and result.get("stdout") == expected(text, unit)
                and result.get("program"))


def test_document_causes_learning() -> tuple[bool, str]:
    cortex = CodeSynthesisCortex(seed=601, pretrain=False)
    before = cortex.handle(TARGET)
    lesson = cortex.handle_document("Doküman: " + PRIMER.read_text(encoding="utf-8"))
    after = cortex.handle(TARGET)
    ok = (not before.get("ok") and not before.get("program")
          and "iterate" in (before.get("missing") or ())
          and lesson.get("ok") and check_result(after, PAYLOAD, "character"))
    return ok, (f"belge öncesi={before.get('path')}:{before.get('missing')} | "
                f"öğrenilen={lesson.get('new_operators')}+{lesson.get('new_facts')} | "
                f"belge sonrası={after.get('path')}")


def practical_trial() -> tuple[bool, str]:
    agi = NoraAGI(seed=602, boot_thinker=False)
    agi.step_life(150)
    started = time.perf_counter()
    reply = strip_prefixes(agi.converse(TARGET))
    elapsed = (time.perf_counter() - started) * 1000.0
    result = dict(agi.last_code or {})
    lines = result.get("stdout", "").splitlines()
    independent_lines = list(PAYLOAD)
    ok = (check_result(result, PAYLOAD, "character") and lines == independent_lines
          and "for character in" in reply and "bilmiyorum" not in reply.lower()
          and "operatörüm yok" not in reply.lower())
    detail = (f"{elapsed:.1f} ms | plan={result.get('operators')} | "
              f"satırlar={len(lines)}/{len(independent_lines)} | kod={result.get('program')!r}")
    return ok, detail


def cumulative_trial() -> tuple[bool, str]:
    cortex = CodeSynthesisCortex(seed=608)
    before = cortex.handle(CUMULATIVE_TARGET)
    lesson = cortex.handle_document("Doküman: " + ACCUMULATION.read_text(encoding="utf-8"))
    result = cortex.handle(CUMULATIVE_TARGET)
    expected_prefixes = "".join(CUMULATIVE_PAYLOAD[:index] + "\n"
                                for index in range(1, len(CUMULATIVE_PAYLOAD) + 1))
    heldout = cortex.handle("Nora! yazısını her satırda bir harf daha eklenmiş hâlde alt alta yazdır.")
    flat = cortex.handle("Nora! yazısını alt alta harf harf yazdır.")
    saved = cortex.semantic.state_dict()
    cortex.semantic.operators.discard("accumulate")
    ablated = cortex.handle(CUMULATIVE_TARGET)
    cortex.semantic.load_state_dict(saved)
    restored = cortex.handle("Ece! yazısını her satırda bir harf daha eklenmiş hâlde alt alta yazdır.")
    ok = (not before.get("ok") and "accumulate" in (before.get("missing") or ())
          and lesson.get("ok") and result.get("stdout") == expected_prefixes
          and result.get("verified") and heldout.get("stdout") == "N\nNo\nNor\nNora\nNora!\n"
          and flat.get("stdout") == "N\no\nr\na\n!\n"
          and not ablated.get("ok") and restored.get("stdout") == "E\nEc\nEce\nEce!\n")
    detail = (f"önce={before.get('path')}:{before.get('missing')} | öğrenilen={lesson.get('new_operators')} | "
              f"plan={result.get('operators')} | satır={len(result.get('stdout', '').splitlines())} | "
              f"yeni metin={heldout.get('stdout')!r} | düz kontrol={flat.get('stdout')!r} | "
              f"ablasyon={ablated.get('path')} restore={restored.get('stdout')!r}")
    return ok, detail


def invention_trial() -> tuple[bool, str]:
    """Belgeli ilk icat + belgesiz, kendi öngörü hatasından doğan ikinci icat."""
    cortex = CodeSynthesisCortex(seed=609)
    document = PREFIX_CONCEPT.read_text(encoding="utf-8")
    lesson = cortex.handle_document("Doküman: " + document)
    heldout = "Zemberek!"
    request = f"{heldout} yazısını her satırda bir harf daha eklenmiş hâlde alt alta yazdır ve çalıştır."
    result = cortex.handle(request)
    expected = "".join(heldout[:index] + "\n" for index in range(1, len(heldout) + 1))
    second_text = "Nora!"
    second = cortex.handle(f"{second_text} yazısını her satırda bir harf daha eklenmiş hâlde alt alta yazdır.")
    second_expected = "".join(second_text[:index] + "\n" for index in range(1, len(second_text) + 1))
    program = result.get("program", "")
    saved = cortex.semantic.state_dict()
    cortex.semantic.operators.discard("concatenate")
    ablated = cortex.handle(request)
    cortex.semantic.load_state_dict(saved)
    restored = cortex.handle(request)
    # Belgesiz ikinci kavram: dışarıdan input/output veya hedef program verilmez.
    # Bağımsız sınav, Nora'nın öz-deney düzeneğini ayrı bir yorumlayıcıyla tekrarlar.
    def eval_expr(expression, environment):
        kind = expression[0]
        if kind == "const":
            return expression[1]
        if kind == "name":
            return environment[expression[1]]
        if kind == "add":
            return eval_expr(expression[1], environment) + eval_expr(expression[2], environment)
        raise ValueError("bilinmeyen sınav ifadesi")

    def replay(construction, source):
        state = eval_expr(construction["initial"], {})
        outputs = []
        for item in source:
            environment = {"item": item, "state": state}
            if construction["after"]:
                state = eval_expr(construction["update"], environment)
                environment["state"] = state
                outputs.append(eval_expr(construction["emit"], environment))
            else:
                outputs.append(eval_expr(construction["emit"], environment))
                state = eval_expr(construction["update"], environment)
        return outputs

    documents_before = list(cortex.semantic.documents)
    novel = cortex.semantic.invent_from_failure()
    novel_id = novel.get("id", "")
    experiment = novel.get("experiment") or {}
    novel_rows = novel.get("heldout") or []
    independent_outputs = [replay(experiment, row["source"]) for row in novel_rows] if experiment else []
    novel_outputs = [cortex.semantic.apply_stream(novel_id, row["source"]) for row in novel_rows]
    novel_saved = cortex.semantic.state_dict()
    cortex.semantic.operators.discard("concatenate")
    novel_ablated = cortex.semantic.apply_stream(novel_id, novel_rows[0]["source"]) if novel_rows else None
    cortex.semantic.load_state_dict(novel_saved)
    novel_restored = (cortex.semantic.apply_stream(novel_id, novel_rows[0]["source"])
                      if novel_rows else None)

    document_clean = "itertools" not in document.lower() and "accumulate(" not in document.lower()
    invented = (result.get("ok") and "itertools" not in program.lower()
                and "accumulate(" not in program.lower() and "+" in program)
    documentless = (novel.get("status") == "invented" and novel.get("origin") == "self_failure"
                    and novel.get("failure", {}).get("mismatches", 0) > 0
                    and cortex.semantic.documents == documents_before and len(novel_rows) >= 2
                    and novel_outputs == independent_outputs
                    and [row.get("outputs") for row in novel_rows] == independent_outputs
                    and novel_ablated is None and novel_restored == independent_outputs[0])
    ok = (document_clean and lesson.get("ok") and invented and documentless
          and result.get("stdout") == expected and result.get("verified")
          and second.get("stdout") == second_expected and second.get("verified")
          and not ablated.get("ok") and restored.get("stdout") == expected)
    detail = (f"belge-temiz={document_clean} | öğrenme={lesson.get('new_operators')} | "
              f"program={program!r} | held-out={result.get('stdout')!r} | "
              f"ikinci={second.get('stdout')!r} | ilkel-ablasyon={ablated.get('path')} | "
              f"restore={restored.get('verified')} | belgesiz={novel_id}:"
              f"{novel.get('failure', {}).get('mismatches', 0)} öngörü hatası "
              f"bağımsız-held-out={novel_outputs == independent_outputs}")
    return ok, detail


def test_transfer_and_composite_reuse() -> tuple[bool, str]:
    cortex = CodeSynthesisCortex(seed=603)
    cases = (
        ("Nora! yazısını alt alta harf harf yazdır ve çalıştır.", "Nora!", "character"),
        ("Ankara yazısını satır satır karakter karakter yazdıran Python kodunu çalıştır.", "Ankara", "character"),
        ("'A B!' yazısını dikey harf harf yazdır.", "A B!", "character"),
        ("Merhaba Baran! yazısını alt alta sözcük sözcük yazdır ve çalıştır.", PAYLOAD, "word"),
        ("iki kelime yazısını satır satır kelime kelime yazdır.", "iki kelime", "word"),
    )
    results = [cortex.handle(request) for request, _text, _unit in cases]
    checks = [check_result(result, text, unit)
              for result, (_request, text, unit) in zip(results, cases)]
    reused = any("composite=reused" in (result.get("trace") or []) for result in results[1:3])
    ok = all(checks) and reused and len(cortex.semantic.composites) >= 2
    return ok, (f"yeni istekler={sum(checks)}/{len(checks)} | "
                f"bileşik operatörler={sorted(cortex.semantic.composites)} | yeniden kullanım={reused}")


def test_operator_ablation_and_restore() -> tuple[bool, str]:
    cortex = CodeSynthesisCortex(seed=604)
    baseline = cortex.handle(TARGET)
    saved = cortex.semantic.state_dict()
    cortex.semantic.operators.discard("iterate")
    ablated = cortex.handle(TARGET)
    cortex.semantic.load_state_dict(saved)
    restored = cortex.handle("Zeynep! yazısını alt alta harf harf yazdır.")
    cortex.semantic.facts.pop("text_sequence", None)
    meaning_ablated = cortex.handle("Deniz! yazısını alt alta harf harf yazdır.")
    word_control = cortex.handle("iki sözcük yazısını alt alta sözcük sözcük yazdır.")
    ok = (check_result(baseline, PAYLOAD, "character")
          and not ablated.get("ok") and not ablated.get("program")
          and check_result(restored, "Zeynep!", "character")
          and not meaning_ablated.get("ok")
          and check_result(word_control, "iki sözcük", "word"))
    return ok, (f"aktif={baseline.get('path')} | iterate ablasyon={ablated.get('path')} | "
                f"restore={restored.get('stdout')!r} | anlam ablasyon={meaning_ablated.get('path')} | "
                f"bağımsız sözcük kontrolü={word_control.get('ok')}")


def test_semantic_persistence() -> tuple[bool, str]:
    agi = NoraAGI(seed=605, boot_thinker=False)
    agi.converse(TARGET)
    fd, path = tempfile.mkstemp(prefix="nora_semantic_code_", suffix=".pkl")
    os.close(fd)
    try:
        agi.save(path)
        restored = NoraAGI.load(path)
        result = restored.code_cortex.handle("Ece! yazısını alt alta harf harf yazdır.")
        ok = check_result(result, "Ece!", "character") and bool(restored.code_cortex.semantic.composites)
        return ok, f"kayıt/yükleme sonrası={result.get('path')} | çıktı={result.get('stdout')!r}"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def web_trial() -> tuple[bool, str]:
    agi = NoraAGI(seed=606, boot_thinker=False, code_pretrain=False)
    reports = []
    for url in CPYTHON_DOCS:
        agi.converse("Bu Python belgesini oku ve işlem anlamlarını öğren: " + url)
        reports.append(dict(agi.last_code or {}))
    char = agi.code_cortex.handle(TARGET)
    word = agi.code_cortex.handle("Merhaba Baran! yazısını alt alta sözcük sözcük yazdır.")
    fetched = all(report.get("fetched") and report.get("ok") for report in reports)
    parsed = sum(int(report.get("parsed", 0) or 0) for report in reports)
    ok = (fetched and parsed > 100 and check_result(char, PAYLOAD, "character")
          and check_result(word, PAYLOAD, "word"))
    return ok, (f"gerçek CPython belgeleri={len(reports)} | AST örneği={parsed} | "
                f"operatörler={sorted(agi.code_cortex.semantic.operators)} | "
                f"karakter={char.get('ok')} sözcük={word.get('ok')}")


def main() -> None:
    if "--web" in sys.argv:
        tests = (("W1 gerçek internet belgeleriyle öğrenme ve uygulama", web_trial),)
    else:
        tests = (
            ("P1 belge öğrenmeyi neden olarak değiştirir", test_document_causes_learning),
            ("P2 gerçek converse hedefi ve bağımsız satır oracle'ı", practical_trial),
            ("P3 parafraz/yük aktarımı ve bileşik operatör", test_transfer_and_composite_reuse),
            ("P4 operatör/anlam ablasyonu ve geri yükleme", test_operator_ablation_and_restore),
            ("P5 öğrenilen anlam ve planın kalıcılığı", test_semantic_persistence),
            ("P6 belgeden birikimli önek operatörü", cumulative_trial),
            ("P7 belgeli ve belgesiz öz-hata kaynaklı operatör icadı", invention_trial),
        )
    passed = 0
    started = time.perf_counter()
    for name, test in tests:
        ok, detail = test()
        passed += int(ok)
        print(f"{'GEÇTİ' if ok else 'KALDI'} {name}: {detail}", flush=True)
    print(f"SONUÇ: {passed}/{len(tests)} ({time.perf_counter() - started:.2f} sn)", flush=True)
    raise SystemExit(0 if passed == len(tests) else 1)


if __name__ == "__main__":
    main()

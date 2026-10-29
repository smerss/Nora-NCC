#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""evolution_battery.py — güvenli kaynak-kod değişiminin dış sınavı.

Örnek çıktılar testin bağımsız oracle'ından üretilir; aday ifade teste yazılmaz.
Train, validation ve son audit kümeleri ayrıdır. Her ret yolunda kaynak dosyanın
orijinali korunur.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile
import time

from nora_evolution import ExpressionSynthesizer, SelfModifier, SourceWorkspace


def _cases(inputs, fn):
    return [{"args": [value], "expect": fn(value)} for value in inputs]


def _workspace(source="def transform(x):\n    return x\n"):
    d = tempfile.TemporaryDirectory(prefix="nora_evolution_")
    with open(os.path.join(d.name, "target.py"), "w", encoding="utf-8") as fh:
        fh.write(source)
    return d, SourceWorkspace(d.name)


def accepted_source_change() -> tuple[bool, str]:
    d, ws = _workspace()
    try:
        modifier = SelfModifier(ws, synthesizer=ExpressionSynthesizer(max_depth=3, max_exprs=3000))
        train = _cases(["ab", "xyz", "kalem"], lambda x: x[::-1])
        validation = _cases(["nora", "qrs", "bulut"], lambda x: x[::-1])
        audit = _cases(["yeni", "portakal"], lambda x: x[::-1])
        result = modifier.improve("target.py", "transform", ["x"],
                                  train=train, validation=validation, audit=audit)
        measured = ws.evaluate("target.py", "transform", audit)
        ok = (result["status"] == "accepted" and measured["score"] == 1.0 and
              result.get("prior") == "safe-ast/enumerative" and
              ws.read("target.py") != "def transform(x):\n    return x\n")
        return ok, f"status={result['status']} expr={result.get('expression')} audit={measured['score']}"
    finally:
        d.cleanup()


def arithmetic_source_change() -> tuple[bool, str]:
    d, ws = _workspace()
    try:
        modifier = SelfModifier(ws, synthesizer=ExpressionSynthesizer(max_depth=3, max_exprs=3000))
        oracle = lambda x: 2 * x + 1
        result = modifier.improve("target.py", "transform", ["x"],
                                  train=_cases([0, 1, 2, 4], oracle),
                                  validation=_cases([3, 5, 7], oracle),
                                  audit=_cases([8, 11], oracle))
        ok = result["status"] == "accepted" and result.get("audit") == 1.0
        return ok, f"status={result['status']} expr={result.get('expression')} audit={result.get('audit')}"
    finally:
        d.cleanup()


def validation_rolls_back() -> tuple[bool, str]:
    original = "def transform(x):\n    return x\n"
    d, ws = _workspace(original)
    try:
        modifier = SelfModifier(ws, synthesizer=ExpressionSynthesizer(max_depth=3, max_exprs=3000))
        result = modifier.improve("target.py", "transform", ["x"],
                                  train=_cases(["ab", "xyz"], lambda x: x[::-1]),
                                  validation=_cases(["nora", "qrs"], lambda x: x.upper()))
        ok = result["status"] == "unresolved" and ws.read("target.py") == original and result.get("rolled_back")
        return ok, f"status={result['status']} rollback={ws.read('target.py') == original}"
    finally:
        d.cleanup()


def audit_rolls_back() -> tuple[bool, str]:
    original = "def transform(x):\n    return x\n"
    d, ws = _workspace(original)
    try:
        modifier = SelfModifier(ws, synthesizer=ExpressionSynthesizer(max_depth=3, max_exprs=3000))
        result = modifier.improve("target.py", "transform", ["x"],
                                  train=_cases(["ab", "xyz"], lambda x: x[::-1]),
                                  validation=_cases(["nora", "qrs"], lambda x: x[::-1]),
                                  audit=_cases(["yeni"], lambda x: "farklı-" + x))
        ok = result["status"] == "audit-rollback" and ws.read("target.py") == original
        return ok, f"status={result['status']} original={ws.read('target.py') == original}"
    finally:
        d.cleanup()


def no_need_and_unresolved() -> tuple[bool, str]:
    source = "def transform(x):\n    return x[::-1]\n"
    d, ws = _workspace(source)
    try:
        modifier = SelfModifier(ws, synthesizer=ExpressionSynthesizer(max_depth=2, max_exprs=600))
        no_need = modifier.improve("target.py", "transform", ["x"],
                                   train=_cases(["ab"], lambda x: x[::-1]),
                                   validation=_cases(["nora"], lambda x: x[::-1]))
        # Bu rastgele eşleme güvenli ifade uzayında açıklanamıyor; uydurma kabul yok.
        impossible = modifier.improve("target.py", "transform", ["x"],
                                      train=[{"args": ["a"], "expect": "q?"},
                                             {"args": ["b"], "expect": "z!"}],
                                      validation=[{"args": ["c"], "expect": "v#"}])
        ok = (no_need["status"] == "no-need" and impossible["status"] == "unresolved"
              and ws.read("target.py") == source)
        return ok, f"no-need={no_need['status']} impossible={impossible['status']}"
    finally:
        d.cleanup()


def timeout_rolls_back() -> tuple[bool, str]:
    original = "def transform(x):\n    while True:\n        pass\n"
    d, ws = _workspace(original)
    try:
        modifier = SelfModifier(ws, synthesizer=ExpressionSynthesizer(max_depth=2, max_exprs=300))
        result = modifier.improve("target.py", "transform", ["x"],
                                  train=[{"args": ["a"], "expect": "q?"}],
                                  validation=[{"args": ["b"], "expect": "z!"}],
                                  timeout=0.1)
        ok = result["status"] == "unresolved" and ws.read("target.py") == original
        return ok, f"status={result['status']} timeout-baseline={result.get('baseline')} original={ws.read('target.py') == original}"
    finally:
        d.cleanup()


def safety_boundary() -> tuple[bool, str]:
    d, ws = _workspace()
    try:
        modifier = SelfModifier(ws)
        try:
            modifier._replace_function("def f(x):\n    return x\n", "f", ["x"],
                                       "__import__('os').system('true')")
        except ValueError:
            unsafe = True
        else:
            unsafe = False
        try:
            ws.path("../outside.py")
        except ValueError:
            traversal = True
        else:
            traversal = False
        return unsafe and traversal, f"unsafe-AST-ret={unsafe} path-ret={traversal}"
    finally:
        d.cleanup()


def _external_oracle_battery(workspace: SourceWorkspace, oracle, values) -> dict:
    """Beklentileri bağımsız oracle hesaplar; aday ayrı Python sürecinde sınanır."""
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


def _external_reverse_battery(workspace: SourceWorkspace) -> dict:
    """Aday üretiminde kullanılmayan, ayrı Python sürecinde çalışan kabul pili."""
    return _external_oracle_battery(
        workspace, lambda value: value[::-1], ["dış sınav", "ab cd", "12345", "Nora!"],
    )


def approval_battery_and_rollback() -> tuple[bool, str]:
    original = "def transform(x):\n    return x\n"
    d, ws = _workspace(original)
    try:
        modifier = SelfModifier(ws, synthesizer=ExpressionSynthesizer(max_depth=3, max_exprs=3000))
        train = _cases(["ab", "xyz", "kalem"], lambda value: value[::-1])
        validation = _cases(["nora", "qrs"], lambda value: value[::-1])
        proposal = modifier.propose_patch("target.py", "transform", ["x"],
                                          train=train, validation=validation)
        calls = []
        battery = lambda: calls.append(1) or _external_reverse_battery(ws)
        unapproved = modifier.apply_patch(proposal, battery=battery, approved=False)
        unchanged_without_approval = ws.read("target.py") == original and not calls
        accepted = modifier.apply_patch(proposal, battery=battery, approved=True)
        kept_source = ws.read("target.py")
        kept_ok = (accepted.get("status") == "accepted" and len(calls) == 1
                   and kept_source != original
                   and ws.evaluate("target.py", "transform",
                                   _cases(["taze", "heldout"], lambda value: value[::-1]))["score"] == 1.0)

        # İkinci gereksinim, kabul edilmiş ilk yapının üstüne kurulur. Dış pilin
        # beklentisi yeni oracle'dan hesaplanır; ilk yama artık tek başına yetmez.
        second_oracle = lambda value: value[::-1].upper()
        second_proposal = modifier.propose_patch(
            "target.py", "transform", ["x"],
            train=_cases(["ab", "xyz", "kalem"], second_oracle),
            validation=_cases(["nora", "qrs"], second_oracle),
        )
        second_rejected = modifier.apply_patch(
            second_proposal, battery=lambda: {"status": "failed"}, approved=True,
        )
        rollback_to_first = ws.read("target.py") == kept_source
        second_calls = []
        second_battery = lambda: (second_calls.append(1) or _external_oracle_battery(
            ws, second_oracle, ["dış sınav", "ab cd", "Nora!", "123ab"],
        ))
        second_accepted = modifier.apply_patch(second_proposal, battery=second_battery, approved=True)
        second_source = ws.read("target.py")

        def returned_expression(source):
            tree = ast.parse(source)
            function = next(node for node in tree.body
                            if isinstance(node, ast.FunctionDef) and node.name == "transform")
            return function.body[0].value

        first_expression = returned_expression(kept_source)
        second_expression = returned_expression(second_source)
        first_dump = ast.dump(first_expression, include_attributes=False)
        incumbent_preserved = any(ast.dump(node, include_attributes=False) == first_dump
                                  for node in ast.walk(second_expression))
        structural = (any(isinstance(node, ast.Subscript) for node in ast.walk(first_expression))
                      and any(isinstance(node, ast.Attribute) for node in ast.walk(second_expression))
                      and incumbent_preserved
                      and second_proposal.get("incumbent") == proposal.get("expression"))
        second_ok = (second_accepted.get("status") == "accepted" and len(second_calls) == 1
                     and second_rejected.get("status") == "rolled_back" and rollback_to_first
                     and structural and sum(row.get("status") == "accepted" for row in modifier.ledger) == 2
                     and ws.evaluate("target.py", "transform",
                                     _cases(["taze", "heldout"], second_oracle))["score"] == 1.0)
    finally:
        d.cleanup()

    bad_original = "def transform(x):\n    return ''\n"
    d2, ws2 = _workspace(bad_original)
    try:
        modifier2 = SelfModifier(ws2, synthesizer=ExpressionSynthesizer(max_depth=3, max_exprs=3000))
        # Palindromlar kimlik ve ters çevirmeyi ayırt etmez; dış pil ayırt eder.
        train2 = _cases(["aa", "bb"], lambda value: value[::-1])
        validation2 = _cases(["cc"], lambda value: value[::-1])
        proposal2 = modifier2.propose_patch("target.py", "transform", ["x"],
                                            train=train2, validation=validation2)
        rejected = modifier2.apply_patch(proposal2, battery=lambda: _external_reverse_battery(ws2),
                                         approved=True)
        rollback_ok = (rejected.get("status") == "rolled_back"
                       and ws2.read("target.py") == bad_original)
    finally:
        d2.cleanup()
    ok = (proposal.get("status") == "proposed" and unapproved.get("status") == "approval-required"
          and unchanged_without_approval and kept_ok and second_ok
          and proposal2.get("status") == "proposed" and rollback_ok)
    detail = (f"öneri={proposal.get('status')}:{proposal.get('expression')} | "
              f"onaysız={unapproved.get('status')} değişim={not unchanged_without_approval} | "
              f"kabul={accepted.get('status')} | yapısal-zincir={proposal.get('expression')} → "
              f"{second_proposal.get('expression')} AST-korundu={structural} ikinci-pil={len(second_calls)} "
              f"ikinci-rollback={rollback_to_first} | "
              f"dış-pil-ret={rejected.get('status')} | kayıp-yama={proposal2.get('expression')} "
              f"rollback={rollback_ok}")
    return ok, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()
    tests = [("E1 kaynak değişimi + bağımsız audit", accepted_source_change),
             ("E2 aritmetik ifade sentezi", arithmetic_source_change),
             ("E3 validation başarısız → rollback", validation_rolls_back),
             ("E4 audit başarısız → rollback", audit_rolls_back),
             ("E5 no-need ve çözülemeyende dürüstlük", no_need_and_unresolved),
             ("E6 timeout sonrası rollback", timeout_rolls_back),
             ("E7 güvenli AST ve çalışma alanı sınırı", safety_boundary),
             ("E8 onaylı yapısal çok-adımlı yama + bağımsız pil + rollback", approval_battery_and_rollback)]
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

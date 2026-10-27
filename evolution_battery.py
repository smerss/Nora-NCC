#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""evolution_battery.py — güvenli kaynak-kod değişiminin dış sınavı.

Örnek çıktılar testin bağımsız oracle'ından üretilir; aday ifade teste yazılmaz.
Train, validation ve son audit kümeleri ayrıdır. Her ret yolunda kaynak dosyanın
orijinali korunur.
"""
from __future__ import annotations

import argparse
import os
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
             ("E7 güvenli AST ve çalışma alanı sınırı", safety_boundary)]
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

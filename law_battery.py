#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""law_battery.py — yeni koşullu temsilin held-out ve planlama sınavı."""
from __future__ import annotations

import argparse
import time
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()
    tests = [("Y1 koşullu ifade yeni örneklerde", conditional_expression),
             ("Y2 sembolik yasa → bağımsız plan", learned_law_plans),
             ("Y3 lineer ablasyon karşılaştırması", linear_ablation),
             ("Y4 audit reddi + atomik fit", audit_rejects_and_atomic_fit),
             ("Y5 Nora induce→plan→checkpoint", nora_induce_plan_checkpoint)]
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

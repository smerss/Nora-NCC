#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""planning_battery.py — model-tabanlı çok adımlı plan ve belirsizlik sınavı."""
from __future__ import annotations

import sys, os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import nora  # noqa: F401  (eski nora_xxx import'ları için uyumluluk kancası)


import argparse
import time
import numpy as np

from nora_causal import CausalTransitionModel
from nora_planning import AdaptivePlanner


def transition_model() -> CausalTransitionModel:
    model = CausalTransitionModel(["f0", "f1"])
    for x in range(-2, 5):
        for y in range(-1, 3):
            state = np.array([x, y], dtype=float)
            for action, delta in (("left", (-1, 0)), ("right", (1, 0)),
                                  ("up", (0, 1)), ("both", (1, 1))):
                model.observe(state, action, state + np.asarray(delta), intervened=True)
    return model


def learned_multistep() -> tuple[bool, str]:
    model = transition_model()
    planner = AdaptivePlanner(model, beam=10, max_horizon=6)
    goal = {"f0": {"target": 2.0, "tol": 1e-8}, "f1": {"target": 1.0, "tol": 1e-8}}
    result = planner.plan([0, 0], ["left", "right", "up", "both"], goal)
    reached = bool(result.get("states")) and np.allclose(result["states"][-1], [2, 1])
    ok = result["status"] == "ok" and reached and len(result["actions"]) >= 1
    return ok, f"status={result['status']} path={result.get('actions')} state={result.get('states',[None])[-1]}"


def unseen_action_features() -> tuple[bool, str]:
    model = CausalTransitionModel(["f0"])
    for x in (-2, -1, 0, 1, 2, 3):
        for step in (0.5, 1.0, 2.0):
            action = {"name": "move", "features": {"step": float(step)}}
            model.observe([x], action, [x + step], intervened=True)
    unseen = {"name": "move", "features": {"step": 1.5}}
    other = {"name": "move", "features": {"step": 0.5}}
    planner = AdaptivePlanner(model, beam=8, max_horizon=4)
    result = planner.plan([0], [unseen, other], {"f0": {"target": 3.0, "tol": 1e-8}})
    ok = result["status"] == "ok" and any(a is unseen for a in result["actions"])
    return ok, f"status={result['status']} path={result.get('actions')} confidence={result.get('confidence'):.3f}"


def no_model_no_plan() -> tuple[bool, str]:
    planner = AdaptivePlanner(CausalTransitionModel(["f0"]), max_horizon=4)
    result = planner.plan([0], ["right"], {"f0": 1})
    return result["status"] == "unresolved" and not result["actions"], f"status={result['status']} actions={result.get('actions')}"


def uncertainty_stops() -> tuple[bool, str]:
    planner = AdaptivePlanner(transition_model(), beam=8, max_horizon=5, min_confidence=0.99)
    result = planner.plan([0, 0], ["right", "up"], {"f0": 5, "f1": 5})
    ok = result["status"] != "ok" and result["expanded"] > 0
    return ok, f"status={result['status']} stop={result.get('stop_reason')} horizon={result.get('horizon')}"


def affordance_is_goal_relative() -> tuple[bool, str]:
    planner = AdaptivePlanner(transition_model())
    actions = ["left", "right"]
    toward = planner.affordances([0, 0], actions, {"f0": 2})
    away = planner.affordances([0, 0], actions, {"f0": -2})
    ok = (toward and away and toward[0]["action"] == "right" and away[0]["action"] == "left"
          and toward[0]["progress"] > 0 and away[0]["progress"] > 0)
    return ok, f"toward={[(x['action'],round(x['progress'],2)) for x in toward]} away={[(x['action'],round(x['progress'],2)) for x in away]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()
    tests = [("L1 öğrenilen çok adımlı plan", learned_multistep),
             ("L2 görülmemiş eylem özniteliği transferi", unseen_action_features),
             ("L3 model yoksa plan uydurma", no_model_no_plan),
             ("L4 belirsizlik ufku durdurur", uncertainty_stops),
             ("L5 affordance hedefe göre öğrenilir", affordance_is_goal_relative)]
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

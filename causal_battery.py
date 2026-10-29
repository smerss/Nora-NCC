#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""causal_battery.py — öğrenilen geçiş modelinin dış, held-out sınavı."""
from __future__ import annotations

import argparse
import time

import numpy as np

from nora_causal import CausalTransitionModel


def categorical_interventions() -> tuple[bool, str]:
    rng = np.random.default_rng(8101)
    model = CausalTransitionModel(feature_names=["x"], min_support=3)
    law = {"artır": 1.4, "azalt": -0.7, "bekle": 0.05}
    actions = tuple(law)
    for i in range(90):
        action = actions[i % len(actions)]
        x = float(rng.normal())
        # Bu yasa yalnız test ortamında; model yalnız önce/sonra çiftini görür.
        model.observe([x], action, [x + law[action]], intervened=True)
    errors, baseline = [], []
    for action in actions:
        pred = model.predict([0.0], action)
        if not pred:
            return False, f"öğrenilen eylem için tahmin yok: {action}"
        p = pred[0]
        errors.append(abs(p["delta"] - law[action]))
        baseline.append(abs(float(np.mean(list(law.values()))) - law[action]))
        if p["evidence"] != "intervention":
            return False, "müdahale kanıtı yanlış sınıflandı"
    unknown = model.require_prediction([0.0], "yeni-eylem")
    if unknown["status"] != "unresolved":
        return False, "görülmemiş eylem uydurma tahmin verdi"
    if not np.mean(errors) < np.mean(baseline):
        return False, f"model baseline'ı geçemedi: {errors} vs {baseline}"
    model.by_action.clear()
    if model.require_prediction([0.0], "artır")["status"] != "unresolved":
        return False, "tablo ablasyonunda sahte tahmin"
    return True, f"mutlak-etki MAE={np.mean(errors):.3f} < taban={np.mean(baseline):.3f}; unknown/ablasyon=unresolved"


def linear_family_transfer() -> tuple[bool, str]:
    rng = np.random.default_rng(8102)
    model = CausalTransitionModel(feature_names=["x"], min_support=3)

    def world(x, features):
        # İki öznitelikli bağımsız test yasası; katsayılar modele yazılmaz.
        return x + 0.35 + 0.8 * features[0] - 0.4 * features[1]

    train = [(a, b) for a in (-1.0, 0.0, 1.0) for b in (0.0, 1.0)]
    for a, b in train:
        for _ in range(2):
            x = float(rng.normal())
            model.observe([x], {"features": [a, b]}, [world(x, (a, b))], intervened=True)
    heldout = [(2.5, -1.0), (-2.0, 0.5), (0.5, -2.0)]
    errors, baseline = [], []
    for a, b in heldout:
        pred = model.predict([0.0], {"features": [a, b]})
        if not pred or pred[0]["source"] != "transition-linear":
            return False, f"held-out lineer tahmin yok: {(a, b)} {pred}"
        actual = world(0.0, (a, b))
        errors.append(abs(pred[0]["delta"] - actual))
        baseline.append(abs(0.35 - actual))
    if not np.mean(errors) < np.mean(baseline):
        return False, f"held-out transfer baseline altında: {errors} vs {baseline}"
    before = model.predict([0.0], {"features": [2.5, -1.0]})
    state = model.state_dict()
    restored = CausalTransitionModel()
    restored.load_state_dict(state)
    after = restored.predict([0.0], {"features": [2.5, -1.0]})
    if before != after:
        return False, "model checkpoint tahmini değiştirdi"
    restored.linear.clear()
    if restored.require_prediction([0.0], {"features": [3.7, 1.2]})["status"] != "unresolved":
        return False, "lineer ablasyon çalışmadı"
    return True, f"held-out MAE={np.mean(errors):.3f} < taban={np.mean(baseline):.3f}; checkpoint+ablasyon ✓"


def association_is_not_intervention() -> tuple[bool, str]:
    model = CausalTransitionModel(feature_names=["x"], min_support=2)
    for i in range(12):
        action = "gözlem-A" if i % 2 else "gözlem-B"
        delta = 1.0 if action.endswith("A") else 0.0
        model.observe([0.0], action, [delta], intervened=False)
    pred = model.predict([0.0], "gözlem-A")
    ok = bool(pred) and pred[0]["evidence"] == "association"
    return ok, f"evidence={pred[0]['evidence'] if pred else 'none'}"


def purity_and_rollout() -> tuple[bool, str]:
    model = CausalTransitionModel(feature_names=["x"], min_support=2)
    for i in range(12):
        a = 0.0 if i % 2 else 1.0
        model.observe([0.0], {"features": [a]}, [0.2 + a], intervened=True)
    state = np.array([0.4])
    before = state.copy()
    n = model.transitions
    first = model.predict(state, {"features": [1.0]})
    second = model.rollout(state, [{"features": [1.0]}, {"features": [0.0]}])
    ok = (np.array_equal(state, before) and model.transitions == n and
          bool(first) and second["status"] == "ok" and len(second["steps"]) == 2)
    return ok, f"predict+rollout salt-okunur; adım={len(second['steps'])}"


def live_nora_transition_channel() -> tuple[bool, str]:
    from nora_agi import NoraAGI
    agi = NoraAGI(seed=8103, boot_thinker=False, single_life=False,
                  world_kwargs={"n_food": 3, "wind": False, "metabolism": False})
    agi.step_life(6)
    if agi.causal.transitions != 6:
        return False, f"canlı kayıt={agi.causal.transitions}"
    obs = agi.obs.copy()
    n = agi.causal.transitions
    agi.tahmin_et(obs, 1)
    if not np.array_equal(obs, agi.obs) or agi.causal.transitions != n:
        return False, "tahmin yaşam durumunu değiştirdi"
    return True, "6/6 gerçek geçiş kayda girdi; tahmin salt-okunur"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()
    tests = [("K1 müdahale → mutlak nedensel etki", categorical_interventions),
             ("K2 eylem ailesi → held-out transfer + persistence", linear_family_transfer),
             ("K3 gözlem ≠ müdahale", association_is_not_intervention),
             ("K4 saf tahmin ve çok adımlı rollout", purity_and_rollout),
             ("K5 canlı Nora geçiş kanalı", live_nora_transition_channel)]
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

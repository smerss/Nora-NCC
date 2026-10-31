#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Yaşamsal frenlerin ve uzun-yaşam davranışının dış sınanması.

Bu pil bilinç veya ölümsüzlük ölçmez.  Öğrenilmiş eylem rehberliğinin kritik
homeostatik refleksi ezmemesini ve sonlu, aynı dünya kurallarında belirlenmiş
ufka kadar yaşamı sınırlı tohum kohortunda ölçer.  Beklenen eylem/yaşam süresi
üretim koduna yazılmaz; burada yalnız bağımsız ölçüt bulunur.
"""
from __future__ import annotations

import sys, os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import nora  # noqa: F401  (eski nora_xxx import'ları için uyumluluk kancası)


import argparse
import time

import numpy as np

from nora_agi import NoraAGI
from nora_mind import I_E, I_I, I_W


SEEDS = (2, 3, 4, 29, 44, 45)
HORIZON = 1500


def emergency_reflex_priority() -> tuple[bool, str]:
    """Kritik bedende öğrenilmiş rehber, suya ilerleme frenini ele geçirmemeli."""
    agi = NoraAGI(seed=2, boot_thinker=False,
                  world_kwargs={"metabolism": True, "wind": False, "n_food": 1})
    w = agi.world
    w.pos = np.array([5.0, 5.0])
    w.vel = np.zeros(2)
    w.th = 0.0
    w.water_c = np.array([[8.0, 5.0], [20.0, 20.0]])
    w.food = np.array([[20.0, 5.0]])
    w.haz_c = np.array([[20.0, 10.0], [20.0, 15.0]])
    w.rock = np.array([15.0, 15.0])
    w.rock_v = np.zeros(2)
    w.e, w.w, w.i = 0.20, 0.20, 0.80
    agi.obs = w.observe()
    agi.died = False

    calls: list[int] = []

    def adversarial_guide(proposed: int) -> int:
        calls.append(int(proposed))
        return 2  # yaşamsal hedefe yönelmek yerine sürekli sola dön

    agi.predictor.guided_action = adversarial_guide
    x_before = float(w.pos[0])
    agi.step_life(1)
    progress = float(w.pos[0]) - x_before
    ok = bool(not calls and progress > 0.0 and agi.mortality.is_alive)
    return ok, (f"rehber-çağrısı={len(calls)} | ilerleme={progress:+.4f} | "
                f"E/W/I={agi.obs[I_E]:.3f}/{agi.obs[I_W]:.3f}/{agi.obs[I_I]:.3f}")


def cohort_survival() -> tuple[bool, str]:
    """Önceden ölen altı tohumun tamamını aynı sonlu ufukta yeniden ölç."""
    rows = []
    survivors = 0
    for seed in SEEDS:
        agi = NoraAGI(seed=seed, boot_thinker=False)
        life = agi.step_life(HORIZON)
        alive = bool(agi.mortality.is_alive)
        survivors += int(alive)
        rows.append((seed, int(agi.mind.age), alive,
                     float(agi.obs[I_E]), float(agi.obs[I_W]), float(agi.obs[I_I]),
                     int(life["ate"]), int(life["drank"])))
        print(f"  S2 seed={seed:2d} yaş={agi.mind.age:4d} canlı={alive} "
              f"E={agi.obs[I_E]:.3f} W={agi.obs[I_W]:.3f} "
              f"ye/iç={rows[-1][-2]}/{rows[-1][-1]}", flush=True)
    ok = survivors == len(SEEDS)
    detail = "; ".join(
        f"{seed}: yaş {age}, {'canlı' if alive else 'ÖLÜ'}, "
        f"E/W/I={e:.2f}/{w:.2f}/{i:.2f}, ye/iç={eat}/{drink}"
        for seed, age, alive, e, w, i, eat, drink in rows
    )
    return ok, f"kohort={survivors}/{len(SEEDS)} | {detail}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tests", nargs="*")
    args = parser.parse_args()
    selected = args.tests or ["S1", "S2"]
    unknown = set(selected) - {"S1", "S2"}
    if unknown:
        parser.error(f"bilinmeyen sınav: {sorted(unknown)}")
    tests = {"S1": emergency_reflex_priority, "S2": cohort_survival}
    started = time.perf_counter()
    passed = 0
    for name in selected:
        ok, detail = tests[name]()
        passed += int(ok)
        print(f"{name} {'✓' if ok else '✗'} {detail}", flush=True)
    print(f"\nSURVIVAL: {passed}/{len(selected)} | {time.perf_counter() - started:.1f} sn")
    return 0 if passed == len(selected) else 1


if __name__ == "__main__":
    raise SystemExit(main())

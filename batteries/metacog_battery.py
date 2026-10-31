#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""metacog_battery.py — işlevsel öz-izleme; bilinç/qualia ölçümü değildir."""
from __future__ import annotations

import sys, os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


import argparse
import os
import tempfile
import time

from nora_metacog import PredictionMonitor


def calibration_learns_error() -> tuple[bool, str]:
    monitor = PredictionMonitor()
    for i in range(12):
        token = monitor.begin([{"variable": "f0", "delta": 1.0,
                                "confidence": 0.9, "source": "controlled"}], [0.0])
        monitor.resolve(token, [1.0 if i < 3 else 0.0])
    report = monitor.report()
    future = monitor.calibrate([{"variable": "f0", "delta": 1.0,
                                 "confidence": 0.9, "source": "controlled"}])[0]
    ok = (report["n"] == 12 and report["calibrated_brier"] < report["raw_brier"]
          and future["confidence"] < future["raw_confidence"])
    return ok, f"n={report['n']} Brier {report['raw_brier']:.3f}→{report['calibrated_brier']:.3f} güven {future['raw_confidence']:.2f}→{future['confidence']:.2f}"


def prequential_and_empty() -> tuple[bool, str]:
    monitor = PredictionMonitor()
    empty = monitor.report()
    token = monitor.begin([{"variable": "f0", "delta": 1.0,
                            "confidence": 0.8, "source": "s"}], [0.0])
    first = monitor.resolve(token, [1.0])[0]
    ok = empty["status"] == "insufficient" and first["calibrated_confidence"] == first["raw_confidence"]
    return ok, f"empty={empty['status']} ilk-kalibrasyon={first['calibrated_confidence']:.2f} ham={first['raw_confidence']:.2f}"


def live_nora_monitor() -> tuple[bool, str]:
    from nora_agi import NoraAGI
    agi = NoraAGI(seed=8821, boot_thinker=False,
                  world_kwargs={"metabolism": False, "wind": False, "n_food": 3})
    agi.step_life(12)
    report = agi.meta_report()
    before = agi.causal.transitions
    calibrated = agi.monitored_prediction(agi.obs, 0)
    pure = agi.causal.transitions == before
    ok = report["n"] > 0 and pure and all("raw_confidence" in p for p in calibrated)
    return ok, f"canlı ölçüm={report['n']} doğruluk={report.get('accuracy')} salt-okunur={pure}"


def persistence() -> tuple[bool, str]:
    from nora_agi import NoraAGI
    prefix = tempfile.mktemp(prefix="nora_meta_", dir="/tmp")
    try:
        agi = NoraAGI(seed=8822, boot_thinker=False,
                      world_kwargs={"metabolism": False, "wind": False, "n_food": 3})
        agi.step_life(10)
        agi.save(prefix)
        restored = NoraAGI.load(prefix)
        a, b = agi.meta_report(), restored.meta_report()
        # Raporun sıralı outcomes listesi de dahil gelecek kalibrasyon durumu korunur.
        same = a == b and agi.meta_monitor.state_dict()["stats"] == restored.meta_monitor.state_dict()["stats"]
        return same, f"rapor-eşit={a == b} kalibrasyon-eşit={same}"
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
    tests = [("M1 hata geçmişi güveni kalibre eder", calibration_learns_error),
             ("M2 prequential ölçüm + boşta dürüstlük", prequential_and_empty),
             ("M3 canlı yaşamda öz-izleme", live_nora_monitor),
             ("M4 kalibrasyon checkpoint'i", persistence)]
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

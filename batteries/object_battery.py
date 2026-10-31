#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""object_battery.py — nesne inancı, occlusion ve persistence dış sınavı."""
from __future__ import annotations

import sys, os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import nora  # noqa: F401  (eski nora_xxx import'ları için uyumluluk kancası)


import argparse
import time
from nora_objects import ObjectBelief


def obs(x, color="red", *, external=None):
    # external id kasıtlı olarak değişebilir; organ bunu kimlik cevabı olarak kullanmaz.
    row = {"position": [float(x), 0.0], "color": color}
    if external is not None:
        row["id"] = external
    return row


def run_with_occlusion(tracker):
    first = tracker.observe([obs(0, external="a"), obs(10, "blue", external="b")])
    aid = tracker.last_assignments[0]
    tracker.observe([obs(1, external="degisti"), obs(10, "blue", external="b2")])
    tracker.observe([obs(10, "blue")])
    hidden = tracker.get(aid)
    tracker.observe([obs(10, "blue")])
    final = tracker.observe([obs(3, external="baska"), obs(10, "blue")])
    return aid, hidden, final, tracker


def reidentification() -> tuple[bool, str]:
    aid, hidden, final, tracker = run_with_occlusion(ObjectBelief(max_missed=3))
    after = tracker.get(aid)
    # Bağımsız hareket oracle'ı: kırmızı nesne her karede +1 ilerler.
    predicted = hidden["features"].get("position.0") if hidden else None
    ok = (hidden is not None and hidden["occluded"] and not hidden["visible"] and
          after is not None and after["visible"] and after["missed"] == 0 and
          predicted == 2.0 and tracker.last_assignments.get(0) == aid)
    return ok, f"track={aid} reidentified={tracker.last_assignments.get(0)} hidden_x={predicted} final={after and after['features']}"


def no_permanence_ablation() -> tuple[bool, str]:
    aid, hidden, final, tracker = run_with_occlusion(ObjectBelief(max_missed=0))
    ok = hidden is None and tracker.last_assignments.get(0) != aid
    return ok, f"ablated_old={aid} new={tracker.last_assignments.get(0)} hidden={hidden}"


def persistence_mid_occlusion() -> tuple[bool, str]:
    a = ObjectBelief(max_missed=3)
    a.observe([obs(0), obs(10, "blue")])
    a.observe([obs(1), obs(10, "blue")])
    a.observe([obs(10, "blue")])
    b = ObjectBelief()
    b.load_state_dict(a.state_dict())
    ra = a.observe([obs(10, "blue")])
    rb = b.observe([obs(10, "blue")])
    ra2 = a.observe([obs(3), obs(10, "blue")])
    rb2 = b.observe([obs(3), obs(10, "blue")])
    ok = ra == rb and ra2 == rb2 and a.last_assignments == b.last_assignments
    return ok, f"same_snapshots={ra2 == rb2} assignments={a.last_assignments == b.last_assignments}"


def capacity_and_uncertainty() -> tuple[bool, str]:
    tracker = ObjectBelief(max_tracks=1, max_missed=1)
    tracker.observe([obs(0)])
    rows = tracker.observe([obs(100, "blue")])
    row = rows[0]
    ok = tracker.dropped == 1 and row["occluded"] and row["confidence"] < 0.65
    return ok, f"dropped={tracker.dropped} confidence={row['confidence']:.3f} occluded={row['occluded']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()
    tests = [("O1 occlusion sonrası kimlik ve görünürlük", reidentification),
             ("O2 permanence ablasyonu fark üretir", no_permanence_ablation),
             ("O3 occlusion ortasında checkpoint devamı", persistence_mid_occlusion),
             ("O4 kapasite ve güven sınırı", capacity_and_uncertainty)]
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

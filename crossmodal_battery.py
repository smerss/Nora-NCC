#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""crossmodal_battery.py — iç izler arasında çapraz-modal bağ ve ablasyon."""
from __future__ import annotations

import argparse
import os
import tempfile
import time

from nora_crossmodal import CrossModalBinder


def regular_binding() -> tuple[bool, str]:
    binder = CrossModalBinder(window=1.0)
    for i in range(4):
        binder.observe("sound", "v0", {"pitch": .2}, timestamp=float(i))
        binder.observe("vision", "t0", {"pitch": .2}, timestamp=i + .1)
        if i == 0:
            binder.observe("vision", "t1", {"pitch": .9}, timestamp=i + .1)
    result = binder.query("sound", "v0")
    ok = (result["status"] == "ok" and result["match"]["token"] == "t0"
          and result["count"] == 4 and result["similarity"] == 1.0)
    return ok, f"match={result.get('match')} count={result.get('count')} confidence={result.get('confidence'):.3f}"


def shuffled_and_insufficient() -> tuple[bool, str]:
    binder = CrossModalBinder(window=1.0)
    for i in range(4):
        binder.observe("sound", "v0", {"pitch": .2}, timestamp=float(i))
        binder.observe("vision", "t0" if i % 2 else "t1", {"pitch": .2}, timestamp=i + .1)
    shuffled = binder.query("sound", "v0")
    small = CrossModalBinder(window=1.0)
    for i in range(2):
        small.observe("sound", "v0", {"pitch": .2}, timestamp=float(i))
        small.observe("vision", "t0", {"pitch": .2}, timestamp=i + .1)
    insufficient = small.query("sound", "v0")
    ok = shuffled["status"] == "unresolved" and insufficient["status"] == "unresolved"
    return ok, f"shuffle={shuffled['status']} ({shuffled.get('count')}) insufficient={insufficient['status']}"


def checkpoint_and_continuation() -> tuple[bool, str]:
    a = CrossModalBinder(window=1.0)
    for i in range(3):
        a.observe("sound", "v0", {"pitch": .2}, timestamp=float(i))
        a.observe("vision", "t0", {"pitch": .2}, timestamp=i + .1)
    b = CrossModalBinder()
    b.load_state_dict(a.state_dict())
    for i in range(3, 5):
        for binder in (a, b):
            binder.observe("sound", "v0", {"pitch": .2}, timestamp=float(i))
            binder.observe("vision", "t0", {"pitch": .2}, timestamp=i + .1)
    qa, qb = a.query("sound", "v0"), b.query("sound", "v0")
    return qa == qb and qa["status"] == "ok", f"continuation-equal={qa == qb} count={qa.get('count')}"


def live_nora_binding() -> tuple[bool, str]:
    from nora_agi import NoraAGI
    agi = NoraAGI(seed=9491, boot_thinker=False)
    vid = ""
    for _ in range(4):
        agi.converse("Merhaba.", speaker="fiziksel-kaynak")
        auditory = agi.last_auditory
        vid = auditory.voice_id
        features = dict(zip(("pitch", "jitter", "tempo", "brightness", "roughness"),
                            auditory.print.vec()))
        features["position"] = [0.0, 0.0]
        agi.perceive_objects([features])
    result = agi.crossmodal.query("sound", vid)
    prefix = tempfile.mktemp(prefix="nora_crossmodal_", dir="/tmp")
    try:
        agi.save(prefix)
        restored = NoraAGI.load(prefix)
        loaded = restored.crossmodal.query("sound", vid)
        ok = (result["status"] == "ok" and loaded == result and
              result["match"]["modality"] == "vision" and result["match"]["token"] == "t0")
        return ok, f"live={result.get('status')} match={result.get('match')} loaded-equal={loaded == result}"
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
    tests = [("X1 düzenli birlikte-oluş → doğru bağ", regular_binding),
             ("X2 shuffle/insufficient → unresolved", shuffled_and_insufficient),
             ("X3 checkpoint ve devam", checkpoint_and_continuation),
             ("X4 canlı kulak + nesne izi + yükleme", live_nora_binding)]
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""workspace_battery.py — yayınlanan bilginin başka uzmanın kararına etkisi."""
from __future__ import annotations

import sys, os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import nora  # noqa: F401  (eski nora_xxx import'ları için uyumluluk kancası)


import argparse
import os
import tempfile
import time

from nora_apply import WorkingMemory
from nora_bus import CognitiveBus
from nora_workspace import IntegrationWorkspace


def producer_changes_consumer() -> tuple[bool, str]:
    bus = CognitiveBus()
    workspace = IntegrationWorkspace(bus)
    workspace.attach(["fact"])
    seen = []
    bus.subscribe("fact", lambda signal: seen.append(workspace.read("fact", "yok")), name="consumer")
    bus.publish("fact", {"value": 3}, source="producer")
    workspace.detach()
    bus.publish("fact", {"value": 9}, source="producer")
    ok = seen == [{"value": 3}, {"value": 3}]
    return ok, f"consumer={seen}; detach sonrası yeni değer yok"


def focus_is_explicit_and_safe() -> tuple[bool, str]:
    bus = CognitiveBus()
    wm = WorkingMemory(capacity=7)
    for i in range(7):
        wm.write(f"user{i}", i, "test")
    before = dict(wm.slots)
    workspace = IntegrationWorkspace(bus, wm)
    workspace.attach(["fact"])
    bus.publish("fact", {"value": 4}, source="producer")
    untouched = list(wm.slots) == list(before) and wm.read("focus") is None
    focused = workspace.focus("fact")
    ok = (untouched and focused and wm.read("focus") == {"value": 4}
          and len(wm.evicted) == 1 and wm.evicted[0].key == "user0")
    return ok, f"auto-WM={not untouched} focus={focused} explicit-eviction={len(wm.evicted)}"


def live_events_and_policy_ablation() -> tuple[bool, str]:
    import numpy as np
    from nora_agi import NoraAGI
    a = NoraAGI(seed=9391, boot_thinker=False,
                world_kwargs={"metabolism": False, "wind": False, "n_food": 3})
    b = NoraAGI(seed=9391, boot_thinker=False,
                world_kwargs={"metabolism": False, "wind": False, "n_food": 3})
    b.workspace.detach()
    writes_before = a.apply_eng.wm.writes
    transitions_a, transitions_b = [], []
    a.bus.subscribe("body.transition", lambda s: transitions_a.append(s.payload), name="measure")
    b.bus.subscribe("body.transition", lambda s: transitions_b.append(s.payload), name="measure")
    a.step_life(6)
    b.step_life(6)
    same = len(transitions_a) == len(transitions_b) == 6
    for left, right in zip(transitions_a, transitions_b):
        same = same and left["action"] == right["action"] and np.array_equal(left["after"], right["after"])
    ok = (same and a.workspace.read("body.transition") is not None
          and a.apply_eng.wm.writes == writes_before)
    return ok, f"aynı-politika={same} workspace-kayıt={a.workspace.writes} otomatik-WM-artışı={a.apply_eng.wm.writes-writes_before}"


def checkpoint_workspace() -> tuple[bool, str]:
    from nora_agi import NoraAGI
    prefix = tempfile.mktemp(prefix="nora_workspace_")
    try:
        agi = NoraAGI(seed=9392, boot_thinker=False,
                      world_kwargs={"metabolism": False, "wind": False, "n_food": 3})
        agi.step_life(3)
        agi.save(prefix)
        restored = NoraAGI.load(prefix)
        same = agi.workspace.snapshot().keys() == restored.workspace.snapshot().keys()
        same = same and agi.workspace.provenance("body.transition") == restored.workspace.provenance("body.transition")
        return same, f"alanlar/kaynak-eşit={same}"
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
    tests = [("W1 yayın → başka uzmanın girdisi + ablasyon", producer_changes_consumer),
             ("W2 focus açık; otomatik WM eviction yok", focus_is_explicit_and_safe),
             ("W3 canlı olaylar + politika ablasyonu", live_events_and_policy_ablation),
             ("W4 workspace checkpoint", checkpoint_workspace)]
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

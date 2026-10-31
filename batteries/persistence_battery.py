#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""persistence_battery.py — yeniden başlatmada davranış sürekliliği sınavı.

Başarı ölçütü "dosya var" değildir: özgün ve yüklenmiş süreç aynı kayıt
noktasından sonra aynı dış geçişleri üretmelidir. Ayrıca ölüm, hedef ve çalışma
belleği gibi gelecek davranışı değiştiren durumlar ayrı ayrı sınanır.
"""
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

import numpy as np

from nora_agi import NoraAGI
from nora_state import StateError, encode


def _cleanup(prefix):
    for ext in (".agi.json", ".agi_mb.npz", ".mind.npz", ".voice.pkl.gz", ".broca.pt"):
        try:
            os.remove(prefix + ext)
        except FileNotFoundError:
            pass


def exact_continuation() -> tuple[bool, str]:
    prefix = tempfile.mktemp(prefix="nora_p_exact_")
    try:
        original = NoraAGI(seed=7301, boot_thinker=False, single_life=False,
                           world_kwargs={"n_food": 5, "wind": True, "metabolism": True})
        original.step_life(5)
        original.save(prefix)
        restored = NoraAGI.load(prefix)
        left, right = [], []
        original.bus.subscribe("body.transition", lambda s: left.append(s.payload), name="measure")
        restored.bus.subscribe("body.transition", lambda s: right.append(s.payload), name="measure")
        for _ in range(8):
            original.step_life(1)
            restored.step_life(1)
        if len(left) != 8 or len(right) != 8:
            return False, f"adım sayısı {len(left)}/{len(right)}"
        for i, (a, b) in enumerate(zip(left, right)):
            if a["action"] != b["action"] or a["info"] != b["info"] or not np.array_equal(a["after"], b["after"]):
                return False, f"devam farkı adım={i}"
        if original.mind.age != restored.mind.age:
            return False, "mind yaşı farklı"
        return True, "8/8 eylem+gözlem+olay birebir; RNG devam etti"
    finally:
        _cleanup(prefix)


def world_rules_and_death() -> tuple[bool, str]:
    prefix = tempfile.mktemp(prefix="nora_p_world_")
    try:
        agi = NoraAGI(seed=7302, boot_thinker=False,
                      world_kwargs={"n_food": 2, "wind": False, "metabolism": False})
        agi.world.poison = True
        agi.world.swap = True
        agi.world.invert = True
        agi.mortality.update_from_vitals(0.0, 0.8, 0.8, age=agi.mind.age, died_signal="sınav")
        agi.save(prefix)
        restored = NoraAGI.load(prefix)
        checks = [restored.world.metab == 0.0, not restored.world.wind_on,
                  restored.world.poison, restored.world.swap, restored.world.invert,
                  not restored.mortality.is_alive]
        age = restored.mind.age
        restored.step_life(4)
        checks.append(restored.mind.age == age)
        if not all(checks):
            return False, f"world/mortality kontrolleri={checks}"
        return True, "metabolizma+rüzgâr+kural bayrakları+ölüm kalıcı; ölü yaşam ilerlemedi"
    finally:
        _cleanup(prefix)


def goals_and_typed_memory() -> tuple[bool, str]:
    prefix = tempfile.mktemp(prefix="nora_p_goal_")
    try:
        agi = NoraAGI(seed=7303, boot_thinker=False,
                      world_kwargs={"n_food": 2, "wind": False, "metabolism": False})
        agi.converse("Hedefin: önümüzdeki üç tur boyunca bana her turda bir harfin okunuşunu öğret.")
        marker = {"items": (1, 2, 3), "done": False, "ratio": 0.25}
        agi.apply_eng.wm.write("plan_state", marker, "measure")
        agi.save(prefix)
        restored = NoraAGI.load(prefix)
        if restored._goals != agi._goals:
            return False, "hedef yığını farklı"
        if restored.apply_eng.wm.read("plan_state") != marker:
            return False, f"tipli bellek bozuldu: {restored.apply_eng.wm.read('plan_state')!r}"
        replies = []
        for item in (agi, restored):
            item.last_real_time = time.time()  # dış saat; beyin durumu değil
            replies.append(item.converse("Bugün hava nasıl bilmiyorum, sen ne dersin?"))
        if not all("1/3" in r for r in replies):
            return False, f"hedef davranışı devam etmedi: {replies}"
        if agi._goals != restored._goals:
            return False, "aynı hedef adımı farklı ilerledi"
        return True, "hedef yığını+tuple/dict çalışma belleği+sonraki 1/3 davranışı korundu"
    finally:
        _cleanup(prefix)


def unsupported_is_explicit() -> tuple[bool, str]:
    try:
        encode(object())
    except StateError as exc:
        return True, f"açık ret: {type(exc).__name__}"
    return False, "desteklenmeyen nesne sessizce kabul edildi"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()
    tests = [("P1 birebir yaşam devamı", exact_continuation),
             ("P2 dünya kuralları ve geri-dönüşsüz ölüm", world_rules_and_death),
             ("P3 hedef ve tipli çalışma belleği", goals_and_typed_memory),
             ("P4 desteklenmeyen durum açık ret", unsupported_is_explicit)]
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

"""
NİHAİ OTONOM DENEY: yoğun girdi -> sessizlik -> kararlılık denetimi.

  1) Girdi yoğunluğu: kısa sürede yüksek frekanslı, duygusal ve sosyal girdiler. Küresel Çalışma Alanı ihale
     dağılımı ve A24 iç-duyum / nörokimya parametrelerinin değişimi izlenir.
  2) Sessizlik: kullanıcı hiç yazmaz. Arka plan döngüsünün (step_life -> life.tick) CA3 taraması, boşluk/merak
     kaydı, zihinsel gerilim ve kendiliğinden mesaj (spontaneous_action) üretimi gözlenir.
  3) Kararlılık: tick süresi sabit mi (O(1)), yapılar sınırlı mı, mesaj yağmuru / sonsuz döngü var mı.

Uyarı: bu betik yalnızca uyaran verir; Nora'ya söz öğretmez. Dili `--corpus` ile verilen metinden öğrenir.
    python nora_life_experiment.py --corpus NORA_EVREN.md --silence 24000
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np

from nora_agi import NoraAGI

STIMULI = [
    "Merhaba Nora! Nasılsın?", "Bugün çok yorgunum, canım sıkkın.", "Sınavı kazandım, çok mutluyum!",
    "Nasılsın?", "Teşekkür ederim, çok iyisin.", "Naber, ne yapıyorsun?", "Çok yalnız hissediyorum.",
    "Harika bir gün!", "Nasılsın?", "Görüşürüz, kendine iyi bak.",
]


def run(corpus: list[str], burst: int = 60, silence: int = 24000, chunk: int = 600, seed: int = 7,
        sources=None, verbose: bool = True) -> dict:
    agi = NoraAGI(seed=seed, corpus=corpus)
    agi.step_life(1500)
    if sources:
        agi.life.sources = list(sources)
    rep: dict = {}

    # ---------------- 1) yoğun girdi
    chem0, vis0 = dict(agi.chem.snapshot()), dict(agi.nervous.snapshot())
    winners: Counter = Counter()
    for i in range(burst):
        agi.converse(STIMULI[i % len(STIMULI)], speaker="insan")
        winners[agi.last_gw_trace.get("winner", "?")] += 1
        agi.step_life(2)
    rep["burst"] = {"gw_winners": dict(winners), "chem_before": chem0, "chem_after": dict(agi.chem.snapshot()),
                    "visceral_before": vis0, "visceral_after": dict(agi.nervous.snapshot()),
                    "gaps_after_burst": agi.life.snapshot()["gaps"]}

    # ---------------- 2) sessizlik
    sizes0 = _sizes(agi)
    per_tick = []
    msgs: list[dict] = []
    t_start = time.perf_counter()
    for c in range(silence // chunk):
        t_a, s_a = agi.life.t, agi.life.tick_seconds
        agi.idle(chunk)
        per_tick.append((agi.life.tick_seconds - s_a) / max(1, agi.life.t - t_a))   # yalnızca arka plan tick maliyeti
        msgs += agi.drain_outbox()
        if not agi.mortality.is_alive:
            break
    rep["silence"] = {"steps": int(agi.mind.age), "alive": bool(agi.mortality.is_alive),
                      "actions": dict(agi.life.stats), "spontaneous": msgs,
                      "ca3_summaries": list(agi.life.summaries)[-3:], "gaps": agi.life.snapshot()["gaps"],
                      "energy": round(float(agi.world.e), 3), "wall_s": round(time.perf_counter() - t_start, 1)}

    # ---------------- 3) kararlılık
    sizes1 = _sizes(agi)
    q = len(per_tick) // 4 or 1
    rep["stability"] = {"tick_ms_first_quarter": round(1e3 * float(np.mean(per_tick[:q])), 3),
                        "tick_ms_last_quarter": round(1e3 * float(np.mean(per_tick[-q:])), 3),
                        "sizes_before": sizes0, "sizes_after": sizes1,
                        "max_unanswered_burst": int(agi.life.unanswered), "spontaneous_total": len(msgs)}
    if verbose:
        _print(rep)
    return rep


def _sizes(agi) -> dict:
    L = agi.life
    return {"ledger": len(L.ledger.gaps), "outbox": len(L.outbox), "summaries": len(L.summaries),
            "news": len(L.news), "pending_hyp": len(L.pool.pending), "rates_log": len(L.rates_log),
            "voice_vocab": len(L.voice.count)}


def _print(rep: dict):
    b, s, st = rep["burst"], rep["silence"], rep["stability"]
    print("== 1) YOĞUN GİRDİ ==")
    print(" GW kazananları:", b["gw_winners"])
    print(" kimya önce :", b["chem_before"])
    print(" kimya sonra:", b["chem_after"])
    print(" iç-duyum   :", b["visceral_before"], "->", b["visceral_after"])
    print("== 2) SESSİZLİK ==")
    print(f" adım={s['steps']} yaşıyor={s['alive']} enerji={s['energy']} süre={s['wall_s']}s")
    print(" arka plan eylemleri:", s["actions"])
    print(" CA3 özetleri (son):", s["ca3_summaries"])
    print(" gerilim defteri:", s["gaps"])
    print(f" kendiliğinden mesaj: {len(s['spontaneous'])}")
    for m in s["spontaneous"][:8]:
        print("   ", m)
    print("== 3) KARARLILIK ==")
    print(st)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="*", default=[])
    ap.add_argument("--silence", type=int, default=24000)
    ap.add_argument("--burst", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    run(a.corpus, burst=a.burst, silence=a.silence, seed=a.seed)

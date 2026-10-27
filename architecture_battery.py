#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""architecture_battery.py — genel yayın ve çalıştırılabilir mimari için dış sınav.

Testler Nora'nın iç raporunu kanıt saymaz. Bağımsız, küçük uzmanlar çalışma
zamanında eklenir; sonuçlar bu uzmanların gerçek çağrıları ve bağımsız bir
oracle ile karşılaştırılır. Ablasyon ve hata yolları da başarı ölçütüdür.
"""
from __future__ import annotations

import argparse
import time

from nora_architecture import ArchitectureModel, capability
from nora_bus import Bid, CognitiveBus


class _Root:
    pass


class _Tokenizer:
    @capability("text", "tokens")
    def tokenize(self, text):
        return str(text).split()


class _Counter:
    @capability("tokens", "count")
    def count(self, tokens):
        return len(tokens)


class _Bad:
    @capability("text", "count", validator=lambda value: isinstance(value, int))
    def count(self, text):
        return str(text)


def dynamic_composition() -> tuple[bool, str]:
    root = _Root()
    root.tokenizer = _Tokenizer()
    root.counter = _Counter()
    model = ArchitectureModel(root)
    samples = ["sıfır", "iki ayrı sözcük", "taze veri ile yeni bir örnek"]
    outputs = []
    for text in samples:
        result = model.run({"text": text}, "count")
        if result["status"] != "ok":
            return False, f"zincir çalışmadı: {result}"
        outputs.append(result["value"])
        if result["value"] != len(text.split()):
            return False, "bağımsız oracle ile uyuşmazlık"
    if len(model.plan({"text"}, "count") or []) != 2:
        return False, "beklenen iki organlı tip zinciri kurulamadı"
    model.unregister("counter")
    if model.run({"text": samples[-1]}, "count")["status"] != "unresolved":
        return False, "ablasyonda sahte başarı"
    return True, f"dinamik zincir={outputs}; ablasyon=unresolved"


def bus_auction_and_ablation() -> tuple[bool, str]:
    bus = CognitiveBus()
    calls = []

    def expert(signal):
        calls.append(signal.id)
        return Bid(2.0, {"action": 4}, "bağımsız ölçüm")

    bus.subscribe("action.propose", expert, name="expert")
    won = bus.auction("action.propose", {"x": 1}, source="mind",
                      baseline=Bid(0.0, {"action": 0}))
    if won is None or won["winner"] != "expert" or len(calls) != 1:
        return False, f"tek çağrılı yarışma bozuk: calls={len(calls)}"
    lineage = bus.trace(won["signal"])
    if len(lineage) < 2 or lineage[0]["id"] != won["request"]:
        return False, "kaynak zinciri yok"
    bus.set_enabled("expert", False)
    before = len(calls)
    fallback = bus.auction("action.propose", {"x": 2}, source="mind",
                           baseline=Bid(0.0, {"action": 0}))
    if fallback is None or fallback["winner"] != "mind" or len(calls) != before:
        return False, "ablasyon eski davranışa dönmedi"
    return True, "yarışma=1 çağrı; ablasyon=mind; lineage=ok"


def invalid_output_is_not_success() -> tuple[bool, str]:
    root = _Root()
    root.bad = _Bad()
    model = ArchitectureModel(root)
    result = model.run({"text": "doğrulama"}, "count")
    ok = result["status"] == "error" and not model.executions[-1]["status"] == "ok"
    return ok, f"durum={result['status']}"


def live_nora_integration() -> tuple[bool, str]:
    from nora_agi import NoraAGI
    agi = NoraAGI(seed=4401, boot_thinker=False,
                  world_kwargs={"metabolism": False, "wind": False, "n_food": 3})
    if agi.bus is None or agi.architecture is None:
        return False, "çalışma-zamanı çekirdeği kurulamadı"
    calls = []

    def external(signal):
        calls.append(signal.payload["proposed"])
        return Bid(3.0, {"action": 0}, "güvenli durma teklifi")

    agi.bus.subscribe("action.propose", external, name="external")
    agi.bus.subscribe("body.transition", lambda signal: None, name="observer")
    agi.step_life(3)
    selected = agi.bus.recent("action.propose.selected", 3)
    transitions = agi.bus.recent("body.transition", 3)
    if len(calls) != 3 or len(selected) != 3 or len(transitions) != 3:
        return False, f"eksik sinyal: calls={len(calls)} selected={len(selected)} transitions={len(transitions)}"
    if any(s.source != "external" for s in selected):
        return False, "dış uzman seçilemedi"
    if any(int(s.payload["action"]) != 0 for s in transitions):
        return False, "seçim gerçek beden eylemine ulaşmadı"
    agi.bus.set_enabled("external", False)
    selected_before = len(agi.bus.recent("action.propose.selected", 16))
    agi.step_life(1)
    selected_after = len(agi.bus.recent("action.propose.selected", 16))
    if len(calls) != 3 or selected_after != selected_before:
        return False, "canlı ablasyonda yeni dış seçim/çağrı oluştu"
    return True, "3/3 dış seçim bedene ulaştı; kapatınca yeni dış seçim yok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()
    tests = [("C1 dinamik tip-zinciri + ablasyon", dynamic_composition),
             ("C2 genel yayın/yarışma + kaynak + ablasyon", bus_auction_and_ablation),
             ("C3 hatalı çıktı başarı sayılmaz", invalid_output_is_not_success),
             ("C4 Nora canlı yayın → beden eylemi", live_nora_integration)]
    t0 = time.time()
    failed = 0
    for rep in range(max(1, args.repeat)):
        for name, fn in tests:
            try:
                ok, detail = fn()
            except Exception as exc:
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            failed += int(not ok)
            print(f"{name}: {'GEÇTİ' if ok else 'KALDI'} | {detail}", flush=True)
    print(f"SONUÇ {len(tests) * max(1, args.repeat) - failed}/{len(tests) * max(1, args.repeat)} | {time.time()-t0:.1f} sn")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

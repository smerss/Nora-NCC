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


def live_counterfactual_removal() -> tuple[bool, str]:
    """Modelin çıkarma yanıtı, sonraki gerçek ablasyonla aynı olmalı."""
    from nora_agi import NoraAGI
    agi = NoraAGI(seed=4402, boot_thinker=False,
                  world_kwargs={"metabolism": False, "wind": False, "n_food": 3})
    model = agi.architecture
    payload = "Mimari!"
    request = f"{payload} yazısını alt alta harf harf yazdır."
    expected = "".join(character + "\n" for character in payload)
    values = {"request": request}
    baseline = model.run(values, "code_result")
    if baseline.get("status") != "ok" or baseline["value"].get("stdout") != expected:
        return False, f"canlı yürütülebilir model tabanı yok: {baseline}"

    critical = model.predict_removal("code_cortex", values, "code_result")
    still_present = agi.code_cortex is model._objects["code_cortex"]
    cortex = agi.code_cortex
    model.unregister("code_cortex")
    actual_critical = model.run(values, "code_result")
    model.register("code_cortex", cortex)
    restored = model.run(values, "code_result")

    noncritical = model.predict_removal("lang", values, "code_result")
    lang = agi.lang
    model.unregister("lang")
    actual_noncritical = model.run(values, "code_result")
    model.register("lang", lang)

    # Canlı çift daha önce birlikte çıkarılmadı: tahmin önce, gerçek ablasyon sonra.
    pair = ("code_cortex", "lang")
    pair_prediction = model.predict_removal(pair, values, "code_result")
    pair_objects = {name: model._objects[name] for name in pair}
    for name in pair:
        model.unregister(name)
    actual_pair = model.run(values, "code_result")
    for name in pair:
        model.register(name, pair_objects[name])
    final_restored = model.run(values, "code_result")

    # Yedekli genel grafikte tekil ablasyonlar zararsız, bazı held-out çiftler
    # kritik olmalı. Bütün çift tahminleri herhangi bir çift çıkarılmadan önce alınır.
    root = _Root()
    redundant = ArchitectureModel(root)
    organs = {
        "tokenizer_a": _Tokenizer(), "tokenizer_b": _Tokenizer(),
        "counter_a": _Counter(), "counter_b": _Counter(),
    }
    for name, organ in organs.items():
        redundant.register(name, organ)
    text = "yedekli yol kombinasyon sınavı"
    single_ok = []
    for name in organs:
        redundant.unregister(name)
        outcome = redundant.run({"text": text}, "count")
        single_ok.append(outcome.get("status") == "ok" and outcome.get("value") == len(text.split()))
        redundant.register(name, organs[name])
    heldout_pairs = [("tokenizer_a", "tokenizer_b"),
                     ("counter_a", "counter_b"),
                     ("tokenizer_a", "counter_a")]
    pair_predictions = {pair: redundant.predict_removal(pair, {"text": text}, "count")
                        for pair in heldout_pairs}
    pair_actual = {}
    for pair in heldout_pairs:
        for name in pair:
            redundant.unregister(name)
        pair_actual[pair] = redundant.run({"text": text}, "count")
        for name in pair:
            redundant.register(name, organs[name])
    pair_match = all(
        pair_actual[pair].get("status") == pair_predictions[pair].get("status")
        and (pair_actual[pair].get("status") != "ok"
             or pair_actual[pair].get("value") == len(text.split()))
        for pair in heldout_pairs
    )
    synergistic = all(pair_predictions[pair].get("status") == "unresolved"
                      for pair in heldout_pairs[:2])
    surviving_pair = pair_predictions[heldout_pairs[-1]].get("status") == "ok"

    ok = (critical.get("status") == "unresolved" and actual_critical.get("status") == critical.get("status")
          and still_present and restored.get("status") == "ok" and restored["value"].get("stdout") == expected
          and noncritical.get("status") == "ok" and actual_noncritical.get("status") == noncritical.get("status")
          and actual_noncritical.get("value", {}).get("stdout") == expected
          and actual_pair.get("status") == pair_prediction.get("status")
          and final_restored.get("status") == "ok" and final_restored["value"].get("stdout") == expected
          and all(single_ok) and pair_match and synergistic and surviving_pair)
    detail = (f"kritik: tahmin={critical.get('status')} gerçek={actual_critical.get('status')} | "
              f"alternatif: tahmin={noncritical.get('status')} gerçek={actual_noncritical.get('status')} | "
              f"canlı-çift={pair_prediction.get('status')}:{actual_pair.get('status')} | "
              f"held-out-yedekli-çiftler={sum(pair_actual[p].get('status') == pair_predictions[p].get('status') for p in heldout_pairs)}/{len(heldout_pairs)} "
              f"tekil={sum(single_ok)}/{len(single_ok)} | geri={final_restored.get('status')}")
    return ok, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()
    tests = [("C1 dinamik tip-zinciri + ablasyon", dynamic_composition),
             ("C2 genel yayın/yarışma + kaynak + ablasyon", bus_auction_and_ablation),
             ("C3 hatalı çıktı başarı sayılmaz", invalid_output_is_not_success),
             ("C4 Nora canlı yayın → beden eylemi", live_nora_integration),
             ("C5 tekil + held-out kombine ablasyon tahmini ve doğrulama", live_counterfactual_removal)]
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

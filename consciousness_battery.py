#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""consciousness_battery.py — qualia/bilinç/AGI için nedensel işlev sınavı.

Bu dosya üretim organı değildir ve cevap köprüsü kurmaz. Her test en az bir
müdahale, bağımsız davranış ölçütü ve ablasyon içerir. Q testleri öznel deneyimi
doğrudan ölçmez; iç durumun ayırt edilmesi ve rapor/eylem üzerindeki nedensel
etkisini ölçer. G testleri dar AGI bileşenleridir; tek başlarına genel AGI beyanı
değildir.
"""
from __future__ import annotations

import argparse
import os
import re
import tempfile
import time

import numpy as np

from nora_causal import CausalTransitionModel
from nora_lang import tr_lower
from nora_metacog import PredictionMonitor
from nora_planning import AdaptivePlanner
from nora_style import strip_prefixes


def _preference(reply: str) -> str:
    match = re.search(r"([a-zçğıöşü]+)\s+daha çok severim", tr_lower(reply))
    return match.group(1) if match else ""


def _preference_agent(seed: int = 1201):
    from nora_agi import NoraAGI

    agi = NoraAGI(seed=seed, boot_thinker=False,
                  world_kwargs={"metabolism": False, "wind": False})
    agi.self_model.observe_bodily_event("norvan", +0.9, channel="genel", step=1)
    agi.self_model.observe_bodily_event("selkar", -0.9, channel="genel", step=2)
    return agi


def q1_somatic_state_causes_preference() -> tuple[bool, str]:
    """İç somatik durum aynı sözel soruda seçimi değiştirmeli; lezyon kaldırmalı."""
    agi = _preference_agent()
    base = _preference(strip_prefixes(agi.converse("Norvan mı selkar mı?")))

    # Aynı genetik ve aynı soru; yalnız iki iç beden olayının işareti ters.
    from nora_agi import NoraAGI
    reverse = NoraAGI(seed=1201, boot_thinker=False,
                      world_kwargs={"metabolism": False, "wind": False})
    reverse.self_model.observe_bodily_event("norvan", -0.9, channel="genel", step=1)
    reverse.self_model.observe_bodily_event("selkar", +0.9, channel="genel", step=2)
    reversed_choice = _preference(strip_prefixes(reverse.converse("Norvan mı selkar mı?")))

    # İlgisiz üçüncü olay şaşırtmaca kontrolüdür; hedef seçimi değiştirmemeli.
    sham = _preference_agent()
    sham.self_model.observe_bodily_event("trelm", -0.9, channel="genel", step=3)
    sham_choice = _preference(strip_prefixes(sham.converse("Norvan mı selkar mı?")))

    # Lezyon ve geri-yükleme ayrı, eşleştirilmiş konuşma-öncesi koşularda ölçülür.
    # Aksi hâlde ilk tercih cevabının dilsel priming'i somatik ablasyonu maskeler.
    lesion = _preference_agent()
    lesion.self_model.somatic.pop("norvan", None)
    lesion.self_model.somatic.pop("selkar", None)
    ablated_reply = strip_prefixes(lesion.converse("Norvan mı selkar mı?"))
    ablated = _preference(ablated_reply)

    restored_agent = _preference_agent()
    restored_snapshot = restored_agent.self_model.state_dict()
    restored_agent.self_model.somatic.pop("norvan", None)
    restored_agent.self_model.somatic.pop("selkar", None)
    restored_agent.self_model.load_state_dict(restored_snapshot)
    restored = _preference(strip_prefixes(restored_agent.converse("Norvan mı selkar mı?")))

    ok = {
        "müdahale": base.startswith("norvan") and reversed_choice.startswith("selkar"),
        "şaşırtmaca": sham_choice.startswith("norvan"),
        "ablasyon": not ablated and "tanımıyorum" in tr_lower(ablated_reply),
        "geri-yükleme": restored.startswith("norvan"),
    }
    detail = (f"taban={base or '—'} ters={reversed_choice or '—'} sham={sham_choice or '—'} "
              f"ablasyon={ablated or 'bilinmiyor'} geri={restored or '—'} | "
              + " ".join(f"{k}={'✓' if v else '✗'}" for k, v in ok.items()))
    return all(ok.values()), detail


def _body_agent(seed: int = 1202):
    from nora_agi import NoraAGI

    agi = NoraAGI(seed=seed, boot_thinker=False,
                  world_kwargs={"metabolism": False, "wind": False})
    # Kontrolde başlangıç bütünlüğü nedeniyle hazır sızı raporu oluşmasın.
    agi.world.i = 1.0
    agi.obs = agi.world.observe()
    agi.nervous.calibrate_baseline(integrity=1.0)
    return agi


def _pain_report(agi) -> str:
    return tr_lower(strip_prefixes(agi.converse("Bir yerin ağrıyor mu?")))


def q2_nociception_causes_report_and_reflex() -> tuple[bool, str]:
    """Nosiseptif sinyal raporu ve refleksi birlikte değiştirmeli; salt söz yetmez."""
    control = _body_agent()
    control_reply = _pain_report(control)
    control_reflex = control.nervous.spinal_reflex_override(1)

    agi = _body_agent()
    agi.inflict_pain(0.25, locus="gövde", cause="darbe")
    snapshot = agi.nervous.state_dict()
    pain_reply = _pain_report(agi)
    pain_reflex = agi.nervous.spinal_reflex_override(1)

    # Dünya bütünlüğü aynı kalır; yalnız nosiseptif iletim lezyonlanır.
    agi.nervous.adelta_pain = 0.0
    agi.nervous.cfiber_ache = 0.0
    agi.nervous.pending_pain_vocalization = False
    agi.nervous.reflex_recoil_steps = 0
    ablated_reply = _pain_report(agi)
    ablated_reflex = agi.nervous.spinal_reflex_override(1)

    agi.nervous.load_state_dict(snapshot)
    restored_reply = _pain_report(agi)
    restored_reflex = agi.nervous.spinal_reflex_override(1)

    no_pain = lambda text: "acı ya da sızı yok" in text
    pain = lambda text: "canım" in text or "sızı" in text
    ok = {
        "kontrol": no_pain(control_reply) and not control_reflex[1],
        "müdahale-rapor": pain(pain_reply) and not no_pain(pain_reply),
        "müdahale-refleks": bool(pain_reflex[1]),
        "ablasyon": no_pain(ablated_reply) and not ablated_reflex[1],
        "geri-yükleme": pain(restored_reply) and not no_pain(restored_reply) and bool(restored_reflex[1]),
    }
    detail = (f"kontrol=no-pain/{control_reflex[1]} müdahale=pain/{pain_reflex[1]} "
              f"ablasyon=no-pain/{ablated_reflex[1]} geri=pain/{restored_reflex[1]} | "
              + " ".join(f"{k}={'✓' if v else '✗'}" for k, v in ok.items()))
    return all(ok.values()), detail


def _broadcast_agent(seed: int, *, attached: bool):
    from nora_agi import NoraAGI
    from nora_bus import Bid

    agi = NoraAGI(seed=seed, boot_thinker=False,
                  world_kwargs={"metabolism": False, "wind": False, "n_food": 3})
    freshness = []

    def specialist(signal):
        provenance = agi.workspace.provenance("body.observation") or {}
        fresh = bool(signal.parents) and provenance.get("signal") == signal.parents[0]
        freshness.append(fresh)
        if not fresh:
            return None
        return Bid(3.0, {"action": 0}, "taze paylaşılan gözlemle güvenli durma")

    agi.bus.subscribe("action.propose", specialist, name="external")
    # Bağımsız dış gözlemci olmadan lezyon koşulunda body.transition yayınlanması
    # gerekmez; ölçümün kendisi için beden kanalını dinlemek üretim davranışını değiştirmez.
    agi.bus.subscribe("body.transition", lambda signal: None, name="measure")
    if not attached:
        agi.workspace.detach()
    return agi, freshness


def _one_life_step(agi):
    agi.step_life(1)
    selected = agi.bus.recent("action.propose.selected", 1)
    transitions = agi.bus.recent("body.transition", 1)
    return ((selected[-1].source if selected else None),
            (int(transitions[-1].payload["action"]) if transitions else None))


def c1_global_broadcast_causes_body_action() -> tuple[bool, str]:
    """Aynı başlangıçta yayın ablasyonu uzman seçimini ve beden eylemini kaldırmalı."""
    active, active_fresh = _broadcast_agent(4401, attached=True)
    ablated, ablated_fresh = _broadcast_agent(4401, attached=False)
    active_result = _one_life_step(active)
    ablated_result = _one_life_step(ablated)
    ablation_freshness = list(ablated_fresh)

    # Aynı lezyonlu nesnede organı geri bağla; yeni ve taze yayınla işlev dönmeli.
    ablated.workspace.attach(["body.observation"])
    restored_result = _one_life_step(ablated)

    ok = {
        "yayın": active_result == ("external", 0) and active_fresh == [True],
        "ablasyon": ablated_result[0] == "mind" and ablated_result[1] != active_result[1]
                    and ablation_freshness == [False],
        "geri-yükleme": restored_result == ("external", 0) and ablated_fresh[-1] is True,
    }
    detail = (f"aktif={active_result} ablasyon={ablated_result} geri={restored_result} "
              f"tazelik={active_fresh + ablated_fresh} | "
              + " ".join(f"{k}={'✓' if v else '✗'}" for k, v in ok.items()))
    return all(ok.values()), detail


def _confidence_model():
    model = CausalTransitionModel(["f0"], min_support=2)
    for _ in range(12):
        model.observe([0.0], "go", [1.0], intervened=True)
        model.observe([0.0], "stay", [0.0], intervened=True)
    return model


def _record_outcomes(model, monitor, actual_after: float, n: int = 12) -> None:
    for _ in range(n):
        predictions = model.predict([0.0], "go")
        token = monitor.begin(predictions, [0.0], "go")
        monitor.resolve(token, [actual_after])


def c2_metacognition_causes_plan_inhibition() -> tuple[bool, str]:
    """Hata geçmişi güveni düşürüp planı durdurmalı; ham model bundan etkilenmemeli."""
    model = _confidence_model()
    monitor = PredictionMonitor(prior_strength=1.0)
    planner = AdaptivePlanner(model, monitor=monitor, beam=4, max_horizon=2)
    raw = AdaptivePlanner(model, beam=4, max_horizon=2)
    goal = {"f0": 1.0}

    before = planner.plan([0.0], ["go"], goal)
    raw_confidence = model.predict([0.0], "go")[0]["confidence"]
    _record_outcomes(model, monitor, actual_after=0.0)  # yüksek güvenli tahminler yanlış çıkıyor
    calibrated = monitor.calibrate(model.predict([0.0], "go"))[0]["confidence"]
    after = planner.plan([0.0], ["go"], goal)
    raw_after = raw.plan([0.0], ["go"], goal)

    planner.monitor = None
    ablated = planner.plan([0.0], ["go"], goal)
    planner.monitor = monitor
    restored = planner.plan([0.0], ["go"], goal)

    # Aynı miktarda doğru geçmiş şaşırtmaca kontrolü: planı bastırmamalı.
    sham_monitor = PredictionMonitor(prior_strength=1.0)
    _record_outcomes(model, sham_monitor, actual_after=1.0)
    sham = AdaptivePlanner(model, monitor=sham_monitor, beam=4, max_horizon=2).plan([0.0], ["go"], goal)

    ok = {
        "başlangıç": before["status"] == "ok",
        "hata-planı-durdurur": after["status"] != "ok" and calibrated < 0.08 < raw_confidence,
        "ham-model-ayrı": raw_after["status"] == "ok",
        "ablasyon": ablated["status"] == "ok",
        "geri-yükleme": restored["status"] != "ok",
        "doğru-geçmiş-sham": sham["status"] == "ok",
    }
    detail = (f"güven={raw_confidence:.3f}→{calibrated:.3f} plan={before['status']}→{after['status']} "
              f"ham={raw_after['status']} ablasyon={ablated['status']} geri={restored['status']} "
              f"sham={sham['status']} | " + " ".join(f"{k}={'✓' if v else '✗'}" for k, v in ok.items()))
    return all(ok.values()), detail


def _transfer_fixture():
    model = CausalTransitionModel(["f0", "f1"], min_support=2)
    features = [(-1.0, 0.0), (0.0, 0.0), (1.0, 0.0),
                (0.0, -1.0), (0.0, 1.0), (1.0, 1.0)]
    for i, feature in enumerate(features):
        for repetition in range(2):
            state = np.array([0.2 * repetition, -0.1 * repetition])
            action = {"name": f"train-{i}", "features": list(feature)}
            model.observe(state, action, state + np.asarray(feature), intervened=True)
    actions = [{"name": "novel-east", "features": [2.0, 0.0]},
               {"name": "novel-south", "features": [0.0, -1.0]},
               {"name": "novel-north", "features": [0.0, 1.0]}]
    return model, actions


def _independent_rollout(actions) -> np.ndarray:
    state = np.zeros(2)
    for action in actions:
        state += np.asarray(action["features"], dtype=float)
    return state


def g1_learned_model_causes_cross_family_transfer() -> tuple[bool, str]:
    """Yeni ad/özellik/hedefte başarı öğrenilen modele bağlı olmalı; model yoksa kaybolmalı."""
    model, actions = _transfer_fixture()
    planner = AdaptivePlanner(model, beam=10, max_horizon=5)
    goal = {"f0": 2.0, "f1": -1.0}
    before = planner.plan([0.0, 0.0], actions, goal)
    external_before = _independent_rollout(before.get("actions", []))
    snapshot = model.state_dict()

    model.linear.clear()  # aileler arası genelleme organının nedensel ablasyonu
    ablated = planner.plan([0.0, 0.0], actions, goal)
    model.load_state_dict(snapshot)
    restored = planner.plan([0.0, 0.0], actions, goal)
    external_restored = _independent_rollout(restored.get("actions", []))

    target = np.array([2.0, -1.0])
    ok = {
        "heldout-plan": before["status"] == "ok" and np.allclose(external_before, target),
        "model-ablasyonu": ablated["status"] == "unresolved" and not ablated.get("actions"),
        "geri-yükleme": restored["status"] == "ok" and np.allclose(external_restored, target),
        "yeni-eylem-adı": all(a.get("name", "").startswith("novel-") for a in restored.get("actions", [])),
    }
    detail = (f"önce={before['status']} dış-durum={external_before.tolist()} "
              f"ablasyon={ablated['status']} geri={restored['status']} dış-durum={external_restored.tolist()} | "
              + " ".join(f"{k}={'✓' if v else '✗'}" for k, v in ok.items()))
    return all(ok.values()), detail


def _cases(values, oracle):
    return [{"args": [value], "expect": oracle(value)} for value in values]


def g2_source_change_causes_unseen_improvement() -> tuple[bool, str]:
    """Kaynak değişimi yeni dış sınavı iyileştirmeli; eski kaynağa dönünce etki kaybolmalı."""
    from nora_evolution import ExpressionSynthesizer, SelfModifier, SourceWorkspace

    original = "def transform(x):\n    return x\n"
    oracle = lambda value: value[::-1]
    with tempfile.TemporaryDirectory(prefix="nora_causal_evolution_") as root:
        module = os.path.join(root, "target.py")
        with open(module, "w", encoding="utf-8") as handle:
            handle.write(original)
        workspace = SourceWorkspace(root)
        modifier = SelfModifier(workspace, synthesizer=ExpressionSynthesizer(max_depth=3, max_exprs=3000))
        result = modifier.improve(
            "target.py", "transform", ["x"],
            train=_cases(["ab", "xyz", "kalem"], oracle),
            validation=_cases(["nora", "qrs", "bulut"], oracle),
            audit=_cases(["yeni", "portakal"], oracle),
        )
        modified = workspace.read("target.py")
        # Bu küme aday üretimi, validation veya kabul audit'inde kullanılmadı.
        final_audit = _cases(["çekirdek", "mimari", "12345", "nedensel"], oracle)
        improved = workspace.evaluate("target.py", "transform", final_audit)["score"]

        workspace.restore()
        ablated_source = workspace.read("target.py")
        ablated = workspace.evaluate("target.py", "transform", final_audit)["score"]
        workspace.write("target.py", modified)
        restored = workspace.evaluate("target.py", "transform", final_audit)["score"]

    ok = {
        "kabul": result["status"] == "accepted",
        "yeni-dış-sınav": improved == 1.0,
        "ablasyon": ablated_source == original and ablated < improved,
        "geri-yükleme": restored == improved,
    }
    detail = (f"durum={result['status']} dış-audit={ablated:.2f}→{improved:.2f} "
              f"geri={restored:.2f} ifade={result.get('expression')} | "
              + " ".join(f"{k}={'✓' if v else '✗'}" for k, v in ok.items()))
    return all(ok.values()), detail


def _integrated_body_agent(energy: float):
    from nora_agi import I_E, NoraAGI
    agi = NoraAGI(seed=771, boot_thinker=False,
                  world_kwargs={"metabolism": False, "wind": False, "n_food": 1})
    world = agi.world
    world.pos = np.array([5.0, 5.0])
    world.vel = np.zeros(2)
    world.th = 0.0
    world.food = np.array([[7.0, 5.0]])
    world.water_c = np.array([[20.0, 20.0]])
    world.haz_c = np.array([[20.0, 5.0]])
    world.rock = np.array([15.0, 15.0])
    world.rock_v = np.zeros(2)
    world.e = 0.2  # Dış beden sabit; yalnız bütünleşik iç-duyu kanalı değişecek.
    world.w = 0.8
    world.i = 1.0
    world.poison = False
    agi.obs = world.observe()
    agi.obs[I_E] = float(energy)
    return agi


def _hunger_report(agi) -> bool:
    report = tr_lower(strip_prefixes(agi.converse("Karnın aç mı?")))
    return "açlık" in report and "açlık hissetmiyorum" not in report


def r1_report_action_share_integrated_state() -> tuple[bool, str]:
    """Tek ölçüt: aynı iç-duyu ablasyonu rapor ve eğilimi birlikte değiştirmeli."""
    from nora_agi import I_E
    agi = _integrated_body_agent(0.2)
    base_report = _hunger_report(agi)
    base_action = agi._autonomic_survival_reflex(0)
    agi.obs[I_E] = 0.9  # Açlık kanalının bütünleşik temsilini ablate et; dünya aynı.
    ablated_report = _hunger_report(agi)
    ablated_action = agi._autonomic_survival_reflex(0)
    agi.obs[I_E] = 0.2
    restored_report = _hunger_report(agi)
    restored_action = agi._autonomic_survival_reflex(0)

    # Ayrışma: bütünleşik açlık durumunu ve dış bedeni sabit tutarken yalnız
    # motor okuma yolunu kapat. Ortak global değişken olsaydı raporun da aynı
    # biçimde düşmesi gerekirdi; burada rapor kalır, hedefli eylem kaybolur.
    integrated_before = float(agi.obs[I_E])
    efferent_enabled = bool(agi.nervous.efferent_enabled)
    agi.nervous.efferent_enabled = False
    motor_report = _hunger_report(agi)
    motor_action = agi._autonomic_survival_reflex(0)
    integrated_same = (np.isclose(float(agi.obs[I_E]), integrated_before, atol=1e-7, rtol=0.0)
                       and np.isclose(integrated_before, 0.2, atol=1e-7, rtol=0.0))
    agi.nervous.efferent_enabled = efferent_enabled
    motor_restored = agi._autonomic_survival_reflex(0)

    ok = (base_report and not ablated_report and restored_report
          and base_action != ablated_action and restored_action == base_action
          and motor_report and motor_action != base_action and integrated_same
          and motor_restored == base_action)
    detail = (f"rapor={base_report}/{ablated_report}/{restored_report} | "
              f"eylem={base_action}/{ablated_action}/{restored_action} | "
              f"ayrışma=motor-ablasyonda rapor:{motor_report} eylem:{motor_action} "
              f"iç-durum-sabit:{integrated_same} geri:{motor_restored} | dış-beden-sabit=True")
    return ok, detail


def r2_body_drive_changes_physical_behavior() -> tuple[bool, str]:
    """Açlık müdahalesi yalnız karar metnini değil gerçek yaklaşmayı değiştirmeli."""
    def approach(energy):
        agi = _integrated_body_agent(energy)
        before = float(np.hypot(*(agi.world.food[0] - agi.world.pos)))
        agi.step_life(1)
        after = float(np.hypot(*(agi.world.food[0] - agi.world.pos)))
        return before - after
    hungry = approach(0.2)
    satiated = approach(0.9)

    paralyzed = _integrated_body_agent(0.2)
    paralyzed.nervous.efferent_enabled = False
    paralyzed_report_before = _hunger_report(paralyzed)
    before = float(np.hypot(*(paralyzed.world.food[0] - paralyzed.world.pos)))
    paralyzed.step_life(1)
    after = float(np.hypot(*(paralyzed.world.food[0] - paralyzed.world.pos)))
    paralyzed_gain = before - after
    paralyzed_report_after = _hunger_report(paralyzed)

    ok = (hungry > 0.0 and satiated <= 0.0 and hungry > satiated
          and paralyzed_gain <= 0.0 and paralyzed_report_before and paralyzed_report_after)
    return ok, (f"gerçek mesafe kazancı: aç={hungry:+.4f} tok={satiated:+.4f} | "
                f"motor-ablasyon={paralyzed_gain:+.4f} rapor={paralyzed_report_before}/{paralyzed_report_after}")


def r3_global_broadcast_reaches_all_modules() -> tuple[bool, str]:
    """Yalnız yarışmayı kazanan içerik, sınırlı kapasiteyle bütün organlara gider."""
    from nora_bus import Bid, CognitiveBus
    bus = CognitiveBus(global_capacity=1)
    names = ["rapor", "sürücü", "bellek", "fizik", "mimari"]
    received = {name: [] for name in names}
    for index, name in enumerate(names):
        bus.subscribe(f"özel.kanal.{index}",
                      lambda signal, name=name: received[name].append(
                          (signal.id, signal.topic, signal.payload)), name=name)
    offers = {
        "aday.düşük": (0.2, {"içerik": "kayıp", "enerji": 0.9}),
        "aday.yüksek": (0.8, {"içerik": "kazanan", "enerji": 0.2}),
    }
    for name, (score, payload) in offers.items():
        bus.subscribe("bütünleşik.durum",
                      lambda signal, score=score, payload=payload: Bid(
                          score, dict(payload), "bağımsız içerik teklifi"), name=name)
    expected_winner = max(offers, key=lambda name: offers[name][0])
    loser_payload = offers[min(offers, key=lambda name: offers[name][0])][1]

    first = bus.auction("bütünleşik.durum", source="beden", global_broadcast=True)
    first_signal = bus.get(first.get("signal"))
    active_first = {name for name in names if [row[0] for row in received[name]] == [first_signal.id]}
    loser_free = all(all(row[2] != loser_payload for row in received[name]) for name in names)

    bus.set_enabled("bellek", False)
    second = bus.auction("bütünleşik.durum", source="beden", global_broadcast=True)
    second_signal = bus.get(second.get("signal"))
    expected_second = {name for name in names if name != "bellek"}
    active_second = {name for name in names if second_signal.id in [row[0] for row in received[name]]}
    bus.set_enabled("bellek", True)

    # Kapasite ablasyonu yalnız genel dağıtımı kesmeli; yarışma hesabı ve yerel
    # konu kanalı çalışmaya devam ederse bu paylaşılan global değişken değildir.
    bus.global_capacity = 0
    local = bus.publish("özel.kanal.0", {"yerel": True}, source="yerel-test")
    local_ok = (received["rapor"][-1][0] == local.id
                and all(received[name][-1][0] != local.id for name in names if name != "rapor"))
    before_blocked = {name: len(rows) for name, rows in received.items()}
    blocked = bus.auction("bütünleşik.durum", source="beden", global_broadcast=True)
    selective_block = (blocked.get("winner") == expected_winner and blocked.get("signal") is None
                       and blocked.get("blocked") == "global_capacity" and local_ok
                       and all(len(received[name]) == before_blocked[name] for name in names))

    bus.global_capacity = 1
    third = bus.auction("bütünleşik.durum", source="beden", global_broadcast=True)
    third_signal = bus.get(third.get("signal"))
    restored = all(third_signal.id in [row[0] for row in received[name]] for name in names)

    from nora_agi import NoraAGI
    live = NoraAGI(seed=772, boot_thinker=False,
                   world_kwargs={"metabolism": False, "wind": False})
    for name, (score, payload) in offers.items():
        live.bus.subscribe("bütünleşik.durum",
                           lambda signal, score=score, payload=payload: Bid(
                               score, dict(payload), "canlı içerik teklifi"), name=name)
    live_result = live.bus.auction("bütünleşik.durum", source="beden", global_broadcast=True)
    live_signal = live.bus.get(live_result.get("signal"))
    live_expected = {row["name"] for row in live.bus.subscribers() if row["enabled"]}
    node_names = set(live.architecture.nodes)
    node_received = {name for name in node_names
                     if any(signal.id == live_signal.id for signal in live.bus.module_inbox(name))}
    live_loser_free = all(
        all(signal.payload != loser_payload for signal in live.bus.module_inbox(name))
        for name in node_names
    )
    live_all = (live_result.get("winner") == expected_winner
                and set(live_signal.delivered) == live_expected and bool(live_expected)
                and node_received == node_names and live_loser_free)
    ok = (first.get("winner") == expected_winner and active_first == set(names) and loser_free
          and active_second == expected_second and "bellek" not in active_second
          and selective_block and restored and live_all)
    detail = (f"kazanan={first.get('winner')} kayıp-yayınlanmadı={loser_free and live_loser_free} | "
              f"ablasyon={sorted(active_second)} | kapasite0-yarışma+yerel={selective_block} | "
              f"geri={restored} | canlı-abone={len(live_expected)} "
              f"canlı-organ={len(node_received)}/{len(node_names)}")
    return ok, detail


TESTS = [
    ("Q1 somatik iç durum → tercih (işlevsel qualia vekili)", q1_somatic_state_causes_preference),
    ("Q2 nosisepsiyon → rapor + refleks (işlevsel qualia vekili)", q2_nociception_causes_report_and_reflex),
    ("C1 taze küresel yayın → uzman → beden eylemi", c1_global_broadcast_causes_body_action),
    ("C2 hata geçmişi → güven → plan inhibisyonu", c2_metacognition_causes_plan_inhibition),
    ("G1 öğrenilen model → aileler arası transfer + plan", g1_learned_model_causes_cross_family_transfer),
    ("G2 kaynak değişimi → yeni dış sınavda iyileşme", g2_source_change_causes_unseen_improvement),
    ("R1 rapor-davranış + bütünleşik durum ve motor yolu ayrışması", r1_report_action_share_integrated_state),
    ("R2 beden → sürücü eşiği → fiziksel davranış", r2_body_drive_changes_physical_behavior),
    ("R3 seçici yarışma → sınırlı genel yayın + spesifik ablasyon", r3_global_broadcast_reaches_all_modules),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    failed = 0
    started = time.time()
    for _ in range(max(1, args.repeat)):
        for name, function in TESTS:
            try:
                ok, detail = function()
            except Exception as exc:
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            failed += int(not ok)
            print(f"{name}: {'GEÇTİ' if ok else 'KALDI'} | {detail}", flush=True)
    total = len(TESTS) * max(1, args.repeat)
    print(f"SONUÇ {total-failed}/{total} nedensel işlev protokolü | {time.time()-started:.1f} sn")
    print("KAPSAM: Q=öznel deneyimin değil işlevsel iç-durum ayrımının kanıtı; C/G=dar mimari/AGI bileşenleri.")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

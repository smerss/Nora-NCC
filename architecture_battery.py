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


def mb_single_presentation_abc() -> tuple[bool, str]:
    """A→B→C tek-sunum sınavı; uyku, replay ve hazır cevap yolu kapalı.

    Beklenenler aşağıdaki bağımsız oracle satırlarıdır. Nora'nın grafındaki
    kenarları cevap oracle'ı olarak kullanmayız; ayrıca her sorunun gerçekten
    HDC/MB okumasıyla ve sembolik yön doğrulamasıyla çözülmesini isteriz.
    """
    from nora_agi import NoraAGI
    from nora_style import strip_prefixes

    agi = NoraAGI(seed=7311, boot_thinker=False, code_pretrain=False,
                  world_kwargs={"metabolism": False, "wind": False, "n_food": 1})
    agi.life.enabled = False
    forbidden_calls = []

    def forbidden_sleep(*_args, **_kwargs):
        forbidden_calls.append("sleep")
        raise AssertionError("uyku/replay kabul sınavında kapalı")

    agi.sleep = forbidden_sleep
    # Canlı turda herhangi bir organ gizli konsolidasyon çağırırsa sınav düşer.
    mb_organs = [agi.lang.graph.word_mb, agi.lang.wernicke.intent_mb,
                 agi.plastic_discourse.mb, agi.schema_cortex.mb, agi.fast.mb]
    relation_mb = getattr(getattr(agi.lang.graph, "relation_mb", None), "mb", None)
    if relation_mb is not None:
        mb_organs.append(relation_mb)
    for organ in mb_organs:
        organ.sleep_consolidate = forbidden_sleep

    teachings = [
        "Zorvath bir alet türüdür, kayaları keser.",
        "Blorg suda yaşar.",
        "Blorg bir Zorvath türüdür.",
    ]
    presentation_counts = {sentence: 0 for sentence in teachings}
    original_teach = agi.lang.teach_sentence

    def counted_teach(text, speaker="insan"):
        if text in presentation_counts:
            presentation_counts[text] += 1
        return original_teach(text, speaker=speaker)

    agi.lang.teach_sentence = counted_teach
    teaching_replies = [agi.converse(sentence) for sentence in teachings]
    # (soru, bağımsız beklenen tür, cevapta bulunması gereken yüzey)
    oracle = [
        ("Zorvath kayaları keser mi?", "yes", "kay"),
        ("Blorg suda mı yaşar?", "yes", "su"),
        ("Blorg nedir?", "contains", "zorvath"),
    ]
    replies, traces, failures = [], [], []
    for question, kind, expected in oracle:
        reply = agi.converse(question)
        clean = strip_prefixes(reply).strip().lower()
        replies.append(reply)
        trace = dict(getattr(agi.lang.graph, "last_decision", {}) or {})
        traces.append(trace)
        if kind == "yes" and not clean.startswith("evet"):
            failures.append(f"oracle yes değil: {question} -> {reply}")
        if kind == "contains" and expected not in clean:
            failures.append(f"oracle içerik yok: {question} -> {reply}")
        if kind == "yes" and expected not in clean:
            failures.append(f"cevap dayanağı yüzeyde yok: {question} -> {reply}")
        if trace.get("engine") != "hdc_mb" or not trace.get("graph_validated"):
            failures.append(f"MB+graf karar izi yok: {question} -> {trace}")
        if not trace.get("awake") or trace.get("replay") != 0:
            failures.append(f"uyanık/replay'siz iz yok: {question} -> {trace}")
    if any(count != 1 for count in presentation_counts.values()):
        failures.append(f"sunum sayıları: {presentation_counts}")
    if forbidden_calls:
        failures.append(f"uyku/replay çağrıları: {forbidden_calls}")
    if relation_mb is None:
        failures.append("merkezî ilişki MB'si yok")
    expected_edges = {
        ("zorvath", "CAN_DO", "kaya kes"),
        ("blorg", "LOCATED_AT", "su"),
        ("blorg", "ISA", "zorvath türü"),
    }
    actual_edges = {(edge.subj, edge.rel, edge.obj) for edge in agi.lang.graph.edges}
    if not expected_edges <= actual_edges:
        failures.append(f"bağımsız üçlü oracle'ı: eksik={sorted(expected_edges - actual_edges)}")
    detail = (f"cevaplar={replies} | sunum={list(presentation_counts.values())} | "
              f"uyku={len(forbidden_calls)} | MB_iz={len(traces)}")
    if failures:
        detail += " | " + " ; ".join(failures[:4])
    return not failures, detail


def mb_relation_invention_transfer() -> tuple[bool, str]:
    """Bilinmeyen yüklem ISA'ya ezilmez; probation operatörü yeni alana taşınır."""
    from nora_chem import NeuroChemistry
    from nora_lang import NoraLanguage

    lang = NoraLanguage(chem=NeuroChemistry(), seed=7312)
    graph = lang.graph
    failures = []
    first = lang.teach_sentence("Narvik torpları vorlar.")
    dynamic = [edge for edge in first if edge.rel.startswith("YENI_OPERATOR")]
    if len(dynamic) != 1:
        return False, f"bilinmeyen yüklem probation ilişkiye dönüşmedi: {first}"
    edge = dynamic[0]
    relation = edge.rel
    if (edge.subj, edge.obj, edge.status) != ("narvik", "torp", "probation"):
        failures.append(f"SVO/probation yanlış: {edge}")
    if any(item.rel == "ISA" and item.subj == "narvik" for item in first):
        failures.append("bilinmeyen yüklem ISA'ya zorlandı")
    presentations_before_repeat = graph.relation_mb.presentations
    lang.teach_sentence("Narvik torpları vorlar.")
    if (graph.last_invention.get("mode") != "reobservation"
            or graph.relation_mb.presentations != presentations_before_repeat):
        failures.append("aynı gözlem yeni icat/sunum gibi yazıldı")

    # İcat edilen ilişki doğrudan öğretilmemiş alt türe ancak ISA kanıtıyla taşınır.
    lang.teach_sentence("Blorg bir Narvik türüdür.")
    transferred = graph.check_relation("blorg", "torpları vorlar")
    if transferred.get("answer") is not True or not transferred.get("isa_path"):
        failures.append(f"yeni türe transfer yok: {transferred}")
    proof = graph.prove_candidate("blorg", relation, "torp")
    if proof.get("status") != "proved" or proof.get("rule") != "isa_inheritance":
        failures.append(f"bağımsız sembolik doğrulama yok: {proof}")

    # Aynı yapının başka özne/nesne alanında görülmesi keşif; ilk operatör icattır.
    lang.teach_sentence("Derva kumları vorlar.")
    discovery = dict(getattr(graph, "last_invention", {}) or {})
    if discovery.get("mode") != "discovery" or discovery.get("relation") != relation:
        failures.append(f"tekrarlanan yapı keşif sayılmadı: {discovery}")
    lang.teach_sentence("Sena bir Derva türüdür.")
    new_domain = graph.check_relation("sena", "kumları vorlar")
    if new_domain.get("answer") is not True:
        failures.append(f"yeni nesne alanına transfer yok: {new_domain}")
    unrelated = graph.check_relation("merta", "torpları vorlar")
    if unrelated.get("answer") is not None:
        failures.append(f"ilişkisiz kontrol uydurma cevap verdi: {unrelated}")

    # Yön çelişkisi eski bilgiyi silmeden reddedilir.
    reverse = graph.invent_relation("torp", "narvik", "Torpları Narvik vorlar.", "vorl")
    if reverse.get("status") != "conflict" or reverse.get("edge") is not None:
        failures.append(f"ters yön çelişkisi yakalanmadı: {reverse}")
    if graph._edge_map.get(("narvik", relation, "torp")) is None:
        failures.append("çelişki denetimi eski kenarı sildi")

    rules = getattr(graph.relation_interpreter, "relation_rules", {})
    rule = next((item for item in rules.values() if item.relation == relation), None)
    if rule is None or rule.status != "probation":
        failures.append("uygulanabilir probation operatörü Interpreter'da yok")
    else:
        triple, _trace = rule.apply("sena", "kum")
        if triple != ("sena", relation, "kum"):
            failures.append(f"operatör yeni bağlanışa uygulanamadı: {triple}")
        empty_triple, _empty_trace = rule.apply("", "kum")
        if empty_triple is not None:
            failures.append("operatör boş özneyle geçerli üçlü üretti")
    trace = dict(getattr(graph, "last_decision", {}) or {})
    if trace.get("engine") != "hdc_mb" or trace.get("replay") != 0:
        failures.append(f"transfer MB'den karar vermedi: {trace}")
    detail = (f"rel={relation} | icat+keşif={discovery.get('mode')} | "
              f"transfer={transferred.get('answer')}/{new_domain.get('answer')} | "
              f"kontrol={unrelated.get('answer')} | çelişki={reverse.get('status')}")
    if failures:
        detail += " | " + " ; ".join(failures[:4])
    return not failures, detail


def mb_ablation_retention_multiseed() -> tuple[bool, str]:
    """MB kararın nedeni mi; yeni yazımlar ve bakım eski bilgiyi koruyor mu?"""
    import numpy as np
    from nora_chem import NeuroChemistry
    from nora_lang import NoraLanguage

    failures, seed_rows = [], []
    for seed in (9101, 9202, 9303, 9404, 9505):
        lang = NoraLanguage(chem=NeuroChemistry(), seed=seed)
        graph = lang.graph
        # Erken yazılan bütün bootstrap kenarları nörogenezden sonra da kendi
        # olay normuyla erişilebilir olmalı; yalnız son yazılan A/B/C'yi sınamak
        # eski kodun seyreltilmesini gözden kaçırıyordu.
        missing_bootstrap = 0
        for subject in sorted({edge.subj for edge in graph.edges if edge.weight > 0.25}):
            recalled = graph.query(subject)
            direct_ids = {id(edge) for edge, path in recalled if not path}
            missing_bootstrap += sum(
                1 for edge in graph.edges
                if edge.subj == subject and edge.weight > 0.25 and id(edge) not in direct_ids)
        if missing_bootstrap:
            failures.append(f"{seed}: nörogenez sonrası {missing_bootstrap} eski kenar okunamadı")
        lang.teach_sentence("Zorvath bir alet türüdür, kayaları keser.")
        lang.teach_sentence("Blorg suda yaşar.")
        lang.teach_sentence("Blorg bir Zorvath türüdür.")

        def oracle():
            first = graph.check_relation("zorvath", "kayaları keser").get("answer") is True
            second = graph.check_relation("blorg", "suda").get("answer") is True
            third = any(edge.obj == "zorvath türü"
                        for edge, _path in graph.query("blorg", ("ISA",)))
            return first and second and third

        if not oracle():
            failures.append(f"{seed}: başlangıç A/B/C başarısız")
            continue
        fast = graph.relation_mb.mb.W_fast.copy()
        slow = graph.relation_mb.mb.W_slow.copy()
        edge_count = len(graph.edges)
        graph.relation_mb.mb.W_fast.fill(0.0)
        graph.relation_mb.mb.W_slow.fill(0.0)
        ablated = (graph.check_relation("zorvath", "kayaları keser").get("answer"),
                   graph.check_relation("blorg", "suda").get("answer"))
        if ablated != (None, None) or len(graph.edges) != edge_count:
            failures.append(f"{seed}: graf varken MB ablasyonu kararı kaldırmadı: {ablated}")
        graph.relation_mb.mb.W_fast[:, :] = fast
        graph.relation_mb.mb.W_slow[:, :] = slow
        if not oracle():
            failures.append(f"{seed}: ağırlık restorasyonu öğretimsiz dönmedi")

        for index in range(40):
            lang.teach_sentence(f"Nesne{index} kayaları keser.")
        if not oracle():
            failures.append(f"{seed}: 40 yeni bilgi sonrası eski A/B/C kayboldu")
        maintenance = graph.relation_mb.sleep_maintenance()
        if maintenance.get("replayed") != 0.0 or not oracle():
            failures.append(f"{seed}: replay'siz bakım sonucu bozuk: {maintenance}")
        if float(np.linalg.norm(graph.relation_mb.mb.W_slow)) <= 0.0:
            failures.append(f"{seed}: bakım W_slow yazmadı")
        sparsity = float(graph.last_decision.get("actual_sparsity", 0.0))
        if not 0.055 <= sparsity <= 0.065:
            failures.append(f"{seed}: seyreklik hedefi sapması: {sparsity}")
        if graph.relation_mb.mb.growth_events <= 0:
            failures.append(f"{seed}: kapasite baskısında nörogenez gözlenmedi")
        seed_rows.append(f"{seed}:{'ok' if not any(str(seed) in row for row in failures) else 'fail'}"
                         f"/KC{graph.relation_mb.mb.n_kc}/s{sparsity:.4f}/eski-eksik{missing_bootstrap}")
    detail = "; ".join(seed_rows)
    if failures:
        detail += " | " + " ; ".join(failures[:6])
    return not failures, detail


def mb_sleep_and_persistence() -> tuple[bool, str]:
    """Canlı uyku replay üretmez; kayıt/yükleme probation ve transferi korur."""
    import os
    import tempfile
    from nora_agi import NoraAGI
    from nora_lang import tr_lower

    agi = NoraAGI(seed=9606, boot_thinker=False, code_pretrain=False,
                  world_kwargs={"metabolism": False, "wind": False, "n_food": 1})
    agi.life.enabled = False
    failures = []
    replay_calls = []

    def forbidden_replay(*_args, **_kwargs):
        replay_calls.append("mind._sleep")
        raise AssertionError("eski replay yolu çağrıldı")

    agi.mind._sleep = forbidden_replay
    if agi.mind.use_replay:
        failures.append("ana zihin periyodik replay ile açıldı")
    agi.step_life(501)  # eski otomatik 500-adım replay sınırını bağımsız aş
    if replay_calls:
        failures.append("500-adım yaşam döngüsü gizli replay çağırdı")
    agi.converse("Narvik torpları vorlar.")
    agi.converse("Blorg bir Narvik türüdür.")
    agi.converse("Zorvath kayaları keser.")
    relation = agi.lang.graph.relation_by_verb.get("vorl", "")
    if not relation:
        failures.append("canlı yeni ilişki kurulamadı")
    if agi.lang.graph.check_relation("blorg", "torpları vorlar").get("answer") is not True:
        failures.append("uyku öncesi transfer yok")
    direct_reply = agi.converse("Narvik torpları vorlar mı?")
    transfer_reply = agi.converse("Blorg torpları vorlar mı?")
    control_reply = agi.converse("Torp kumları vorlar mı?")
    if not tr_lower(direct_reply).startswith("evet") or not tr_lower(transfer_reply).startswith("evet"):
        failures.append("canlı Wernicke–MB–Broca evet/çıkarım yolu başarısız")
    if tr_lower(control_reply).startswith("evet"):
        failures.append("canlı ilişkisiz kontrol sahte evet verdi")
    if agi.lang.graph.last_decision.get("engine") != "hdc_mb":
        failures.append("canlı soru MB dışında karar verdi")
    result = agi.sleep()
    if result.get("replay") != 0 or result.get("s2_replayed") != 0 or replay_calls:
        failures.append(f"uyku replay içerdi: {result}, {replay_calls}")
    if agi.lang.graph.check_relation("zorvath", "kayaları keser").get("answer") is not True:
        failures.append("uyku eski Zorvath bilgisini kaybettirdi")
    if agi.lang.graph.check_relation("blorg", "torpları vorlar").get("answer") is not True:
        failures.append("uyku probation transferini kaybettirdi")

    with tempfile.TemporaryDirectory(prefix="nora_mb_accept_") as folder:
        prefix = os.path.join(folder, "mind")
        agi.save(prefix)
        loaded = NoraAGI.load(prefix)
        loaded.life.enabled = False
        if loaded.mind.use_replay:
            failures.append("save/load periyodik replay'i yeniden açtı")
        graph = loaded.lang.graph
        if graph.relation_by_verb.get("vorl") != relation:
            failures.append("yüklem→ilişki eşlemi kalıcı değil")
        if graph.check_relation("blorg", "torpları vorlar").get("answer") is not True:
            failures.append("yeniden yüklemede held-out transfer yok")
        if graph.check_relation("zorvath", "kayaları keser").get("answer") is not True:
            failures.append("yeniden yüklemede eski bilgi yok")
        edge = graph._edge_map.get(("narvik", relation, "torp"))
        if edge is None or edge.status != "probation":
            failures.append("probation durumu kalıcı değil")
        rule = graph.relation_interpreter.relation_rules.get(f"relation::{relation}")
        if rule is None or rule.status != "probation":
            failures.append("uygulanabilir operatör kalıcı değil")
        loaded_reply = loaded.converse("Blorg torpları vorlar mı?")
        if not tr_lower(loaded_reply).startswith("evet"):
            failures.append("yeniden yüklemede canlı transfer cevabı yok")
    detail = f"uyku_replay={result.get('replay')} | eski_replay_çağrısı={len(replay_calls)} | hata={len(failures)}"
    if failures:
        detail += " | " + " ; ".join(failures[:5])
    return not failures, detail


def mb_multiple_relation_inventions() -> tuple[bool, str]:
    """Farklı -Ar kökleri ayrı operatör olur; bilinen fiil ve eski kökler bozulmaz."""
    from nora_chem import NeuroChemistry
    from nora_lang import NoraLanguage, tr_lower

    cases = [
        ("Narvik", "torpları", "torp", "vorlar", "vorl"),
        ("Zimre", "kumları", "kum", "quresar", "qures"),
        ("Pırna", "taşları", "taş", "pırpler", "pırpl"),
        ("Drelm", "suları", "su", "zimrer", "zimr"),
        ("Korvi", "buzları", "buz", "sündürür", "sündür"),
    ]
    failures, rows = [], []
    for seed in (1201, 1302, 1403):
        lang = NoraLanguage(chem=NeuroChemistry(), seed=seed)
        graph = lang.graph
        relation_ids, retained = [], []
        for index, (source, obj_surface, obj, surface_verb, root) in enumerate(cases):
            frame = lang.wernicke.parse(f"{source} {obj_surface} {surface_verb}.")
            if (frame.kind != "invent_relation" or frame.subj != tr_lower(source)
                    or frame.obj != obj or frame.verb != root):
                failures.append(f"{seed}/{root}: SVO çözümleme yanlış: {frame}")
                continue
            lang.teach_sentence(frame.raw)
            invented = graph.last_invention
            expected_rel = graph._relation_id(root)
            relation_ids.append(invented.get("relation", ""))
            if (invented.get("relation") != expected_rel or invented.get("mode") != "invention"
                    or invented.get("status") != "probation"):
                failures.append(f"{seed}/{root}: icat/probation kaydı yanlış: {invented}")
            derived = f"Aktarım{index}"
            lang.teach_sentence(f"{derived} bir {source} türüdür.")
            target = f"{obj_surface} {surface_verb}"
            if graph.check_relation(source, target).get("answer") is not True:
                failures.append(f"{seed}/{root}: doğrudan ilişki yok")
            if graph.check_relation(derived, target).get("answer") is not True:
                failures.append(f"{seed}/{root}: yeni türe transfer yok")
            if graph.check_relation("yabancı", target).get("answer") is not None:
                failures.append(f"{seed}/{root}: ilişkisiz kontrol None değil")
            retained.append((source, derived, target))
            for old_source, old_derived, old_target in retained:
                if (graph.check_relation(old_source, old_target).get("answer") is not True
                        or graph.check_relation(old_derived, old_target).get("answer") is not True):
                    failures.append(f"{seed}/{root}: önceki ilişki unutuldu: {old_target}")
                    break
        if len(set(relation_ids)) != len(cases):
            failures.append(f"{seed}: farklı kökler aynı operatöre çöktü: {relation_ids}")
        before_known = set(graph.relation_by_verb)
        lang.teach_sentence("Narvik kayaları keser.")
        if set(graph.relation_by_verb) != before_known:
            failures.append(f"{seed}: bilinen keser fiili yeni operatöre dönüştü")
        if graph.check_relation("narvik", "kayaları keser").get("answer") is not True:
            failures.append(f"{seed}: bilinen CAN_DO yolu geriledi")
        lang.teach_sentence("Narvik mavidir.")
        if set(graph.relation_by_verb) != before_known:
            failures.append(f"{seed}: koşaçlı sıfat yeni operatöre dönüştü")
        rows.append(f"{seed}:{len(set(relation_ids))}/5 kök")
    detail = ", ".join(rows)
    if failures:
        detail += " | " + " ; ".join(failures[:7])
    return not failures, detail


def epistemic_gate_zero_broca() -> tuple[bool, str]:
    """Bilinmeyen anlam Broca'ya gitmez; soğuk tanım yanıtı 20 ms altındadır."""
    from nora_agi import NoraAGI

    agi = NoraAGI(seed=1501, boot_thinker=False, code_pretrain=False,
                  broca="data/broca_tr.pt",
                  world_kwargs={"metabolism": False, "wind": False, "n_food": 1})
    agi.life.enabled = False
    organ = agi.life.voice.organ
    calls = {"generate": 0, "learn_online": 0}
    original_generate = organ.generate
    original_learn = organ.learn_online

    def counted_generate(*args, **kwargs):
        calls["generate"] += 1
        return original_generate(*args, **kwargs)

    def counted_learn(*args, **kwargs):
        calls["learn_online"] += 1
        return original_learn(*args, **kwargs)

    organ.generate = counted_generate
    organ.learn_online = counted_learn
    times, answers = [], []
    for question in ("Kuş nedir?", "Nivara nedir?", "Tormek nedir?"):
        started = time.perf_counter()
        answers.append(agi.converse(question))
        times.append(time.perf_counter() - started)
    honest = all("bilgim yok" in answer.casefold() and "tanımım yok" in answer.casefold()
                 and "ne olduğunu biliyorum" not in answer.casefold()
                 for answer in answers)
    traces = agi.last_gw_trace.get("winner") == "epistemic_gate" and \
        agi.last_epistemic_gate.get("known") is False
    # Hızlı tanım dışındaki bilinmeyen ilişkide de ortak çapa kapısı nöral
    # üretimi/öğretimi açmamalı; uzman yollar erken reddedilmez.
    dynamic_unknown = agi.converse("Torp kumları vorlar mı?")
    dynamic_honest = "henüz" in dynamic_unknown.casefold() or "bilgim yok" in dynamic_unknown.casefold()
    ok = (honest and traces and dynamic_honest and max(times) < 0.020
          and calls["generate"] == 0 and calls["learn_online"] == 0)
    return ok, (f"ms={[round(x * 1000, 2) for x in times]} | neural={calls} | "
                f"dürüst={honest and dynamic_honest} | anı={len(agi.broca_organ.episodic.episodes)}")


def graph_corpus_single_pass() -> tuple[bool, str]:
    """Corpus önermesi Broca'ya değil Graph/HDC'ye yazılır ve hemen kullanılır."""
    import tempfile
    from pathlib import Path
    from nora_agi import NoraAGI
    from nora_style import strip_prefixes

    agi = NoraAGI(seed=1502, boot_thinker=False, code_pretrain=False,
                  broca="data/broca_tr.pt",
                  world_kwargs={"metabolism": False, "wind": False, "n_food": 1})
    agi.life.enabled = False
    sleep_calls = []
    agi.sleep = lambda *a, **k: sleep_calls.append((a, k))
    organ = agi.life.voice.organ
    calls = {"generate": 0, "learn_online": 0}
    original_generate = organ.generate
    original_learn = organ.learn_online

    def counted_generate(*args, **kwargs):
        calls["generate"] += 1
        return original_generate(*args, **kwargs)

    def counted_learn(*args, **kwargs):
        calls["learn_online"] += 1
        return original_learn(*args, **kwargs)

    organ.generate = counted_generate
    organ.learn_online = counted_learn
    with tempfile.TemporaryDirectory(prefix="nora-graph-corpus-") as td:
        path = Path(td) / "corpus.txt"
        path.write_text("Kuşlar uçar.\nKuş bir hayvandır.\n", encoding="utf-8")
        stats = agi.read_corpus(str(path))
    presentations = int(agi.lang.graph.relation_mb.presentations)
    direct = agi.lang.graph.check_relation("kuş", "uçar").get("answer") is True
    started = time.perf_counter()
    definitions = agi.converse("Kuş nedir?")
    definition_seconds = time.perf_counter() - started
    definition_gate = dict(agi.last_epistemic_gate)
    neural_definition = dict(calls)
    yes_no = strip_prefixes(agi.converse("Kuşlar uçar mı?"))
    neural_after = dict(calls)
    # Sorular yeni sunum/replay değildir; yalnız okuma yapar.
    presentations_after = int(agi.lang.graph.relation_mb.presentations)
    ok = (stats.get("sentences") == 2 and stats.get("errors") == 0
          and stats.get("organ_seen") == 0 and stats.get("heard_tokens") == 0
          and stats.get("graph_edges", 0) >= 2
          and stats.get("mb_presentations") == stats.get("graph_edges")
          and direct and "hayvan" in definitions.casefold() and "uç" in definitions.casefold()
          and "bilgim yetersiz" in definitions.casefold()
          and definition_gate.get("insufficient") is True
          and definition_gate.get("semantic_dimensions", {}).get("işlev") is True
          and definition_gate.get("semantic_dimensions", {}).get("parça") is False
          and definition_gate.get("semantic_dimensions", {}).get("nedensellik") is False
          and definition_seconds <= 0.020
          and neural_definition == {"generate": 0, "learn_online": 0}
          and yes_no.casefold().startswith("evet") and not sleep_calls
          and presentations_after == presentations)
    return ok, (f"corpus={stats} | doğrudan={direct} | tanım={definitions} | "
                f"tanım_s={definition_seconds:.4f} | boyut={definition_gate.get('semantic_dimensions')} | "
                f"tanım_nöral={neural_definition} | tüm_nöral={neural_after} | "
                f"evet-hayır={yes_no} | uyku={len(sleep_calls)}")


def passive_hdc_reactivation() -> tuple[bool, str]:
    """Diskteki pasif olay HDC ipucuyla RAM'e gelir; graf yine doğrular."""
    import pickle
    import sqlite3
    from nora_chem import NeuroChemistry
    from nora_lang import NoraLanguage, tr_lower

    lang = NoraLanguage(chem=NeuroChemistry(), seed=1503)
    graph = lang.graph
    mb = graph.relation_mb
    mb.set_working_set_size(4)
    lang.teach_sentence("Nivara bir varlıktır.")
    slot = next(i for i, edge in enumerate(graph.edges)
                if edge.subj == "nivara" and edge.rel == "ISA" and edge.obj == "varlık")
    others = [f"Kalıcı{i}" for i in range(24)]
    for subject in others:
        lang.teach_sentence(f"{subject} bir varlıktır.")
    was_passive = slot not in mb.active_slots and slot in mb.passive_slots
    storage = mb.passive_storage_info()
    with sqlite3.connect(storage["path"]) as independent_db:
        disk_row = independent_db.execute(
            "SELECT fast, slow FROM passive WHERE slot=?", (slot,)).fetchone()
    disk_proved = bool(storage["file_exists"] and storage["backend"] == "sqlite"
                       and storage["ram_passive_weight_rows"] == 0
                       and disk_row is not None and len(disk_row[0]) > 0 and len(disk_row[1]) > 0)
    before_presentations = int(mb.presentations)
    before_reactivations = int(mb.passive_reactivations)
    hits = graph.query("nivara", ("ISA",))
    recalled = any(edge.subj == "nivara" and edge.obj == "varlık" for edge, _path in hits)
    retained = all(any(edge.subj == tr_lower(subject) and edge.obj == "varlık"
                       for edge, _path in graph.query(subject, ("ISA",))) for subject in others)
    # Aynı pasif sinapslar bağımsız bir serileştirme turundan sonra da okunur.
    restored = pickle.loads(pickle.dumps(mb))
    restored_storage = restored.passive_storage_info()
    graph.relation_mb = restored
    restored_hits = graph.query("nivara", ("ISA",))
    restored_ok = any(edge.subj == "nivara" and edge.obj == "varlık"
                      for edge, _path in restored_hits)
    # Kopya farklı SQLite dosyası kullanır; onun okuması ilk nesnenin pasif
    # satırını silmez. İki nesneyi art arda bağımsız olarak doğrula.
    graph.relation_mb = mb
    original_after = graph.query("nivara", ("ISA",))
    original_ok = any(edge.subj == "nivara" and edge.obj == "varlık"
                      for edge, _path in original_after)
    independent_stores = (restored_storage["path"] != storage["path"]
                          and restored_storage["file_exists"] and storage["file_exists"])
    ok = (was_passive and disk_proved and recalled and retained and restored_ok and original_ok
          and independent_stores and len(mb.active_slots) <= 4 and len(restored.active_slots) <= 4
          and mb.passive_reactivations > before_reactivations
          and mb.presentations == before_presentations
          and restored.presentations == before_presentations)
    return ok, (f"pasif={was_passive} disk={disk_proved} | geri={recalled}/{restored_ok}/{original_ok} | "
                f"24_kayıt={retained} | aktif={len(mb.active_slots)}/4 | "
                f"uyandırma={mb.passive_reactivations - before_reactivations} | "
                f"sunum={mb.presentations - before_presentations} | depo=sqlite")


def epistemic_event_symbolic_broca() -> tuple[bool, str]:
    """Bilinmezlik Graph olayından Broca'ya gider; dünya kanıtına dönüşmez."""
    import tempfile
    from pathlib import Path
    from nora_agi import NoraAGI

    agi = NoraAGI(seed=1504, boot_thinker=False, code_pretrain=False,
                  broca="data/broca_tr.pt",
                  world_kwargs={"metabolism": False, "wind": False, "n_food": 1})
    agi.life.enabled = False
    calls = {"generate": 0, "learn_online": 0, "symbolic_broca": 0}

    def instrument(current):
        organ = current.life.voice.organ
        if organ is not None:
            original_generate = organ.generate
            original_learn = organ.learn_online

            def counted_generate(*args, **kwargs):
                calls["generate"] += 1
                return original_generate(*args, **kwargs)

            def counted_learn(*args, **kwargs):
                calls["learn_online"] += 1
                return original_learn(*args, **kwargs)

            organ.generate = counted_generate
            organ.learn_online = counted_learn
        original_realize = current.lang.broca.realize_epistemic

        def counted_realize(*args, **kwargs):
            calls["symbolic_broca"] += 1
            return original_realize(*args, **kwargs)

        current.lang.broca.realize_epistemic = counted_realize

    instrument(agi)
    answers = [agi.converse("Nivara nedir?"), agi.converse("Nivara nedir?")]
    semantic_before = agi.lang.graph.query("nivara", None)
    unknown_events = [e for e in agi.lang.graph.epistemic_events
                      if e.subj == "nivara" and e.rel == "HAS_STATE" and e.obj == "unknown"]
    before_presentations = int(agi.lang.graph.relation_mb.presentations)
    with tempfile.TemporaryDirectory(prefix="nora-epistemic-event-") as td:
        prefix = str(Path(td) / "state")
        agi.save(prefix)
        restored = NoraAGI.load(prefix)
        restored.life.enabled = False
        instrument(restored)
        answers.append(restored.converse("Nivara nedir?"))
        loaded_unknown = [e for e in restored.lang.graph.epistemic_events
                          if e.subj == "nivara" and e.obj == "unknown"]
        still_semantic_empty = not restored.lang.graph.query("nivara", None)
        restored.lang.teach_sentence("Nivara bir varlıktır.")
        state = restored.lang.graph.epistemic_state("nivara")
        unknown_retained = any(e.subj == "nivara" and e.obj == "unknown"
                               for e in restored.lang.graph.epistemic_events)
        learned = any(e.subj == "nivara" and e.rel == "ISA" and e.obj == "varlık"
                      for e, _path in restored.lang.graph.query("nivara", ("ISA",)))
        restored_presentations = int(restored.lang.graph.relation_mb.presentations)
    honest = all("nivara" in answer.casefold() and "bilgim yok" in answer.casefold()
                 and "tanımım yok" in answer.casefold() for answer in answers)
    ok = (honest and not semantic_before and still_semantic_empty
          and unknown_events and loaded_unknown and unknown_retained
          and state is not None and state.obj == "known" and learned
          and restored_presentations == before_presentations + 1
          and calls["generate"] == 0 and calls["learn_online"] == 0
          and calls["symbolic_broca"] >= 3)
    return ok, (f"yanıt={answers[-1]} | olay={len(unknown_events)}/{len(loaded_unknown)} "
                f"durum={state.obj if state else None} | öğrenme={learned} | çağrı={calls}")


def graph_system2_composition() -> tuple[bool, str]:
    """Çok-adım, karşılaştırma ve birleşik ters sorgu; HDC ablasyonuyla."""
    from nora_agi import NoraAGI

    agi = NoraAGI(seed=1505, boot_thinker=False, code_pretrain=False,
                  world_kwargs={"metabolism": False, "wind": False, "n_food": 1})
    agi.life.enabled = False
    write_deltas = []

    def teach(text):
        return agi.converse(text)

    def ask(text):
        before = int(agi.lang.graph.relation_mb.presentations)
        answer = agi.converse(text)
        write_deltas.append(int(agi.lang.graph.relation_mb.presentations) - before)
        return answer

    teach("Zorvath bir memelidir.")
    teach("Memeli sıcakkanlıdır.")
    inherited = ask("Zorvath sıcakkanlı mıdır?")
    inherited_trace = dict(agi.last_gw_trace)
    inherited_not_memorized = ("zorvath", "HAS_PROP", "sıcakkanlı") not in agi.lang.graph._edge_map
    direct = ask("Zorvath memeli midir?")
    direct_trace = dict(agi.last_gw_trace)
    teach("Tera, kivaya yol açar.")
    teach("Kiva, nivaya yol açar.")
    causal = ask("Tera nivaya yol açar mı?")
    causal_trace = dict(agi.last_gw_trace)
    causal_not_memorized = ("tera", "CAUSES", "niva") not in agi.lang.graph._edge_map

    for text in ("Mira bir hayvandır.", "Mira evcil bir varlıktır.", "Mira miyavlar.",
                 "Dora bir hayvandır.", "Dora evcil bir varlıktır.", "Dora havlar."):
        teach(text)
    comparison = ask("Mira ile Doranın ortak noktası nedir?")
    comparison_result = dict(agi.last_graph_reasoning)
    comparison_trace = dict(agi.last_gw_trace)
    shared_keys = {(row["rel"], row["obj"]) for row in comparison_result.get("shared") or []}
    original_recall = agi.lang.graph.relation_mb.recall_relation_object
    try:
        agi.lang.graph.relation_mb.recall_relation_object = lambda *_args, **_kwargs: []
        comparison_ablated = agi.lang.graph.compare_concepts("mira", "dora")
    finally:
        agi.lang.graph.relation_mb.recall_relation_object = original_recall
    try:
        def without_dora(relation, obj):
            return [(slot, activation) for slot, activation in original_recall(relation, obj)
                    if agi.lang.graph.edges[slot].subj != "dora"]

        agi.lang.graph.relation_mb.recall_relation_object = without_dora
        comparison_partial = agi.lang.graph.compare_concepts("mira", "dora")
    finally:
        agi.lang.graph.relation_mb.recall_relation_object = original_recall
    partial_shared = {(row["rel"], row["obj"])
                      for row in comparison_partial.get("shared") or []}

    teach("Tivaranın kanadı vardır.")
    teach("Tivara uçar.")
    inverse = ask("Kanadı olan ve uçan şey nedir?")
    inverse_result = dict(agi.last_graph_reasoning)
    inverse_trace = dict(agi.last_gw_trace)
    inverse_subjects = [row["subject"] for row in inverse_result.get("candidates") or []]
    negative = ask("Kanadı olan ve havlayan şey nedir?")
    negative_result = dict(agi.last_graph_reasoning)
    negative_trace = dict(agi.last_gw_trace)
    try:
        agi.lang.graph.relation_mb.recall_relation_object = lambda *_args, **_kwargs: []
        inverse_ablated = agi.lang.graph.find_by_requirements(
            [("HAS_PART", "kanat"), ("CAN_DO", "uç")])
    finally:
        agi.lang.graph.relation_mb.recall_relation_object = original_recall
    definition = ask("Tivara nedir?")

    ok = ("dolayısıyla" in inherited.casefold() and inherited_not_memorized
          and inherited_trace.get("used_s2") is True
          and direct_trace.get("used_s2") is False
          and "dolayısıyla" in causal.casefold() and causal_not_memorized
          and causal_trace.get("used_s2") is True
          and ("ISA", "hayvan") in shared_keys and ("HAS_PROP", "evcil") in shared_keys
          and "miyav" not in comparison.casefold() and "havla" not in comparison.casefold()
          and comparison_trace.get("used_s2") is True
          and comparison_ablated.get("status") == "unresolved"
          and not comparison_ablated.get("shared")
          and ("ISA", "hayvan") not in partial_shared
          and ("HAS_PROP", "evcil") not in partial_shared
          and inverse_subjects == ["tivara"] and "tivara" in inverse.casefold()
          and "kanad" in inverse.casefold() and "uç" in inverse.casefold()
          and inverse_trace.get("used_s2") is True
          and negative_result.get("status") == "unresolved"
          and not negative_result.get("candidates")
          and negative_trace.get("used_s2") is False
          and "bul" in negative.casefold()
          and inverse_ablated.get("status") == "unresolved"
          and "yetersiz" not in definition.casefold()
          and "kanad" in definition.casefold() and "uç" in definition.casefold()
          and all(delta == 0 for delta in write_deltas))
    return ok, (f"kalıtım={inherited} | doğrudan_S1={direct_trace.get('used_s2') is False} | "
                f"neden={causal} | ortak={sorted(shared_keys)} | "
                f"kısmi_ablasyon={sorted(partial_shared)} | ters={inverse} | olumsuz={negative} | tanım={definition} | "
                f"yazım={write_deltas}")


def bounded_retention_generalization_multiseed() -> tuple[bool, str]:
    """Üç tohumda 96 olay; bakım ve bağımsız depo sonrası bağımsız oracle."""
    import pickle
    from nora_chem import NeuroChemistry
    from nora_lang import NoraLanguage

    failures, rows = [], []
    for seed, base in ((1601, "miraf"), (1702, "tovik"), (1803, "selvara")):
        lang = NoraLanguage(chem=NeuroChemistry(), seed=seed)
        graph = lang.graph
        mb = graph.relation_mb
        mb.set_working_set_size(3)
        lang.teach_sentence(f"{base} bir memelidir.")
        lang.teach_sentence("Memeli sıcakkanlıdır.")
        expected = [f"nesne{i}" for i in range(96)]
        for concept in expected:
            lang.teach_sentence(f"{concept} bir varlıktır.")
        before = int(mb.presentations)

        def missing_items():
            missing = []
            for concept in expected:
                hits = graph.query(concept, ("ISA",))
                if not any(edge.subj == concept and edge.rel == "ISA" and edge.obj == "varlık"
                           for edge, _path in hits):
                    missing.append(concept)
            return missing

        live_missing = missing_items()
        inherited = graph.check_relation(base, "sıcakkanlı")
        maintenance = mb.sleep_maintenance()
        sleep_missing = missing_items()
        sleep_inherited = graph.check_relation(base, "sıcakkanlı")
        passive_before = len(mb.passive_slots)
        active_live = len(mb.active_slots)
        restored = pickle.loads(pickle.dumps(mb))
        graph.relation_mb = restored
        restored_missing = missing_items()
        restored_inherited = graph.check_relation(base, "sıcakkanlı")
        not_memorized = (base, "HAS_PROP", "sıcakkanlı") not in graph._edge_map
        counts = (96 - len(live_missing), 96 - len(sleep_missing), 96 - len(restored_missing))
        seed_ok = (counts == (96, 96, 96) and passive_before >= 96
                   and active_live <= 3 and len(restored.active_slots) <= 3
                   and inherited.get("answer") is True and inherited.get("isa_path")
                   and sleep_inherited.get("answer") is True
                   and restored_inherited.get("answer") is True and restored_inherited.get("isa_path")
                   and maintenance.get("replayed") == 0.0 and not_memorized
                   and int(mb.presentations) == before and int(restored.presentations) == before)
        if not seed_ok:
            failures.append(f"{seed}: sayı={counts}, pasif={passive_before}, "
                            f"kalıtım={inherited.get('answer')}/{restored_inherited.get('answer')}")
        rows.append(f"{seed}:{counts}/96 aktif{active_live}/3 pasif{passive_before}")
    return not failures, " | ".join(rows) + (" | " + " ; ".join(failures) if failures else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--only", default="", help="yalnız adı bu metni içeren sınavları çalıştır")
    args = ap.parse_args()
    tests = [("C1 dinamik tip-zinciri + ablasyon", dynamic_composition),
             ("C2 genel yayın/yarışma + kaynak + ablasyon", bus_auction_and_ablation),
             ("C3 hatalı çıktı başarı sayılmaz", invalid_output_is_not_success),
             ("C4 Nora canlı yayın → beden eylemi", live_nora_integration),
             ("C5 tekil + held-out kombine ablasyon tahmini ve doğrulama", live_counterfactual_removal),
             ("C6 MB tek-sunum A→B→C; uyku/replay kapalı", mb_single_presentation_abc),
             ("C7 probation ilişki icadı + held-out alan transferi", mb_relation_invention_transfer),
             ("C8 MB ablasyonu + çoklu tohum + unutmama", mb_ablation_retention_multiseed),
             ("C9 replay'siz uyku + operatör kalıcılığı", mb_sleep_and_persistence),
             ("C10 çok-köklü ilişki icadı + kontroller", mb_multiple_relation_inventions),
             ("C11 epistemik kapı + sıfır Broca + 20ms", epistemic_gate_zero_broca),
             ("C12 graph-corpus tek geçiş + uyanık kullanım", graph_corpus_single_pass),
             ("C13 pasif HDC bellek + sınırlı çalışma kümesi", passive_hdc_reactivation),
             ("C14 epistemik olay + sembolik Broca + kalıcılık", epistemic_event_symbolic_broca),
             ("C15 System-2 birleşik anlam + HDC ablasyonu", graph_system2_composition),
             ("C16 sınırlı tutma + genelleme + çoklu tohum", bounded_retention_generalization_multiseed)]
    if args.only:
        tests = [(name, fn) for name, fn in tests if args.only.casefold() in name.casefold()]
    if not tests:
        raise SystemExit("seçime uyan sınav yok")
    t0 = time.time()
    failed = 0
    run_count = 0
    for rep in range(max(1, args.repeat)):
        for name, fn in tests:
            try:
                ok, detail = fn()
            except Exception as exc:
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            failed += int(not ok)
            run_count += 1
            print(f"{name}: {'GEÇTİ' if ok else 'KALDI'} | {detail}", flush=True)
    print(f"SONUÇ {run_count - failed}/{run_count} | {time.time()-t0:.1f} sn")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

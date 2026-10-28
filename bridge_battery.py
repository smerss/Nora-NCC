#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bridge_battery.py — genel entegrasyon köprüleri için davranışsal sınama.

Kullanım:
  python bridge_battery.py                 # B1–B6
  python bridge_battery.py B1 B6            # seçili köprüler

Bu dosya KÖPRÜ KODU DEĞİLDİR. Üretim organlarına yeni bağlantı eklenmez; yalnız
var olan genel arayüzler çağrılır. Özellikle:

* bir grafik kenarı Universe/Mind durumuna kopyalanmaz,
* bir somatik kayıt Broca için Proposition'a çevrilmez,
* bir Rule/feature/mapping nesnesi test tarafından başka alana taşınmaz.

Ölçütler dışarıdan hesaplanır. Bir modülün iç defterine bakıp onun kendi
"başardım" bayrağını sınamak yerine, Universe enerjisi/konumu veya dışarı verilen
sözün hangi bağımsız ölçütü karşıladığı gözlenir.

Bu testler yetenek/entegrasyon testidir; bilinç ölçümü değildir.
"""
from __future__ import annotations

import argparse
import re
import sys
import time

import numpy as np

from nora_agi import NoraAGI
from nora_lang import tr_lower
from nora_style import strip_prefixes


# Yalnız ölçüm/orakül için kullanılan bağımsız yüzey ölçümleri.
# Nora'nın FEATURES veya Rule uygulaması çağrılmaz.
_VOWELS = frozenset("aeıioöuü")


def _vowel_count(text: str) -> int:
    return sum(ch.lower() in _VOWELS for ch in text)


def _word_count(text: str) -> int:
    return len(re.findall(r"[^\W\d_]+", text, flags=re.UNICODE))


def _parity(n: int) -> str:
    return "odd" if n % 2 else "even"


def _food_fixture(agi: NoraAGI) -> None:
    """Aynı küçük sahneyi kur; bu yalnız deney düzeneğidir, öğrenme köprüsü değil.

    Metabolizma ve rüzgâr kapalıdır. Böylece enerjideki negatif sıçrama, bu
    sahnede yalnız zehirli yiyecek temasının dış kanıtıdır. Su ve tehlike
    bölgeleri uzakta, yiyecek önde ve tektir.
    """
    w = agi.world
    w.pos = np.array([5.0, 5.0])
    w.vel = np.zeros(2)
    w.th = 0.0
    w.food = np.array([[7.0, 5.0]])
    w.water_c = np.array([[20.0, 20.0], [21.0, 20.0]])
    w.haz_c = np.array([[20.0, 5.0], [20.0, 10.0]])
    w.rock = np.array([15.0, 15.0])
    w.rock_v = np.zeros(2)
    w.e = 0.4
    w.w = 0.8
    w.i = 0.8
    w.poison = True
    agi.obs = w.observe()
    agi.died = False


def _food_probe(
    seed: int,
    teach: bool,
    horizon: int = 12,
    target: str = "yiyecek",
    teach_text: str = "Yiyecek zehirlenmeye neden olur.",
    command: str = "Yiyeceğe git.",
) -> dict:
    """Metin öğretilmiş/öğretilmemiş aynı sahnede ilk temas davranışını ölç."""
    agi = NoraAGI(
        seed=seed,
        boot_thinker=False,
        world_kwargs={"metabolism": False, "wind": False, "n_food": 1},
    )
    if teach:
        # Özel değer kestirmesi değil, genel CAUSES öğrenme arayüzü.
        # Bu çağrı yalnız metni mevcut dil öğrenme arayüzünden geçirir; dönen
        # kenarı Mind/V veya Universe'e test yazmaz.
        agi.lang.teach_sentence(teach_text)

    _food_fixture(agi)
    agi.converse(command)
    first_contact = None
    energy_before = float(agi.world.e)
    for step in range(1, horizon + 1):
        old_e = float(agi.world.e)
        agi.step_life(1)
        new_e = float(agi.world.e)
        # Bu düzenekte metabolizma=0 ve su uzakta: negatif sıçrama yiyecek
        # temasının bağımsız dış kanıtıdır. İlk temas sonrası öğrenme başarı
        # sayılmaz; döngü bu yüzden hemen durur.
        if new_e < old_e - 1e-9:
            first_contact = step
            break

    # Rapor, davranışsal ölçüm bittikten SONRA alınır: raporlama turu eylemi
    # etkileyerek sahte bir köprü başarısı üretemez. Aynı oturumda bilginin
    # sözlü olarak da erişilebilir olduğunu bağımsız biçimde kontrol eder.
    report = strip_prefixes(agi.converse(f"{target.capitalize()} neye yol açar?")) if teach else ""
    return {
        "agi": agi,
        "first_contact": first_contact,
        "energy_before": energy_before,
        "energy_after": float(agi.world.e),
        "report": report,
    }


def b1_text_causal_edge_to_universe() -> tuple[bool, str]:
    """B1 — metindeki genel nedensel kenar Universe davranışını değiştirmeli.

    Önce yalnız dil öğrenme arayüzünden ``yiyecek CAUSES zehirlenme`` öğretilir.
    Sonra aynı zehirli-yiyecek sahnesi, aynı tohum ve aynı komutla iki ayrı
    Nora'da koşulur. Kontrolün sahneyi gerçekten çözebildiği (temas ettiği)
    doğrulanmadan deney başarısı sayılmaz. Deneyde başarı, ilk temasın metin
    sonrası beden deneyiminden ÖNCE engellenmesidir.

    Bu, ``zehirlidir`` özel değer cümlesinin hazır valans yolunu sınamaz; genel
    CAUSES kenarının hedef davranışta kullanılıp kullanılmadığını sınar.
    """
    control = _food_probe(404, teach=False)
    experiment = _food_probe(404, teach=True)

    # Dönen listeyi değil, gerçek grafik durumunu önkoşul olarak oku. Bu
    # okuma yalnız öğrenmenin varlığını doğrular; kenarı başka organa taşımaz.
    learned_edges = [
        (e.subj, e.rel, e.obj)
        for e in experiment["agi"].lang.graph.edges
        if e.subj == "yiyecek" and e.rel == "CAUSES"
        and e.obj == "zehirlenme" and e.source == "duyum"
    ]
    fixture_valid = control["first_contact"] is not None
    bridge_used = experiment["first_contact"] is None
    report_ok = "zehirlenme" in tr_lower(experiment["report"])
    ok = bool(learned_edges) and fixture_valid and bridge_used and report_ok

    detail = (
        f"metin-kenarı={'✓' if learned_edges else '✗'} | "
        f"rapor={'✓' if report_ok else '✗'} | "
        f"kontrol-ilk-temas={control['first_contact']} | "
        f"deney-ilk-temas={experiment['first_contact']} | "
        f"kontrol-E={control['energy_before']:.2f}->{control['energy_after']:.2f} | "
        f"deney-E={experiment['energy_before']:.2f}->{experiment['energy_after']:.2f}"
    )
    if not learned_edges:
        detail += " | halka: metin genel CAUSES kenarına dönüşmedi"
    elif not report_ok:
        detail += " | halka: aynı oturumda grafik → sözlü rapor"
    elif not fixture_valid:
        detail += " | halka: kontrol sahnesi geçersiz (davranış farkı ölçülemez)"
    elif not bridge_used:
        detail += " | halka: GroundedConceptGraph → Mind/Universe (ilk temas öncesi kullanılmadı)"
    return ok, detail


def _preference_reply(seed: int, elma_value: float, diken_value: float) -> str:
    """Bedensel olayları verip aynı Broca tercih sorusunu sor; iç kayıt taşınmaz."""
    agi = NoraAGI(
        seed=seed,
        boot_thinker=False,
        world_kwargs={"metabolism": False, "wind": False},
    )
    # Bu, CoreSelf'in mevcut beden-olayı arayüzüdür. Kavramlar Broca'ya
    # Proposition veya başka bir temsil çevirisiyle verilmez.
    agi.self_model.observe_bodily_event("elma", elma_value, channel="enerji", step=1)
    agi.self_model.observe_bodily_event("diken", diken_value, channel="bütünlük", step=2)
    return strip_prefixes(agi.converse("Elma mı diken mi?"))


def _selected_preference(reply: str) -> str:
    """Dış sözde 'X daha çok severim' seçimini oku; cevap üretmez."""
    match = re.search(r"([a-zçğıöşü]+)\s+daha çok severim", tr_lower(reply))
    return match.group(1) if match else ""


def b2_embodied_concept_to_broca() -> tuple[bool, str]:
    """B2 — bedensel değer, Broca'nın sözel seçimini nedensel olarak değiştirmeli.

    Aynı tohumda iki koşu yalnız beden olaylarının işaretini tersine çevirir.
    Beklenen seçim sabit bir cümleden değil, her koşunun dışarıdan verilen
    değerlerinin büyüklüğünden hesaplanır. İki koşunun da kendi bağımsız
    beklentisini karşılaması gerekir; tek koşudaki 'elma' cevabı kanıt değildir.
    """
    cases = [
        {"elma": +0.9, "diken": -0.9},
        {"elma": -0.9, "diken": +0.9},
    ]
    rows = []
    all_ok = True
    for values in cases:
        reply = _preference_reply(101, values["elma"], values["diken"])
        selected = _selected_preference(reply)
        expected = max(values, key=values.get)
        # Türkçe belirtme eki kökü korur ('elmayı', 'dikeni'); üretilen
        # yüzeyi başka bir iç temsile çevirmiyoruz, yalnız kök önekiyle okuyoruz.
        case_ok = bool(selected) and selected.startswith(expected)
        all_ok = all_ok and case_ok
        rows.append(
            f"{values['elma']:+.1f}/{values['diken']:+.1f}: "
            f"seçim={selected or '—'} beklenen-kök={expected} {'✓' if case_ok else '✗'}"
        )

    # Ters koşular aynı seçimi veriyorsa beden olayının nedensel katkısı yoktur;
    # yukarıdaki bağımsız bekleyiş kontrolü bunu zaten reddeder, bu alan yalnız
    # raporu açık tutar.
    detail = " | ".join(rows)
    if not all_ok:
        detail += " | halka: CoreSelf somatik olay → Broca tercih/planlama"
    return all_ok, detail


def _rule_reply(seed: int, transfer: bool, probe: str, labels: dict[str, str]) -> str:
    """Kuralı yalnız doğal dil genel arayüzünden kur, isteğe bağlı alan transferi yap."""
    agi = NoraAGI(
        seed=seed,
        boot_thinker=False,
        world_kwargs={"metabolism": False, "wind": False},
    )
    # Örnek kuralın iki etiketi de girdi olarak belirtilir; beklenen cevap
    # aşağıda probe'un bağımsız ölçümünden hesaplanır.
    rule_text = f"sesli harf sayısı tek ise {labels['odd']}, çift ise {labels['even']}"
    agi.converse(rule_text)
    if transfer:
        # Test Rule.transfer, feature veya mapping çağırmaz. Alan değişimini
        # sistemin kendi genel konuşma arayüzüne bırakır.
        agi.converse("Aynı kuralı mutfakta kullan.")
    return strip_prefixes(agi.converse(f"'{probe}' hangi etiketi alır?"))


def b3_learned_structure_cross_domain() -> tuple[bool, str]:
    """B3 — öğrenilen yapı başka alanın görevinde kullanılmalı.

    Aynı kural iki ajanın dil alanında öğrenilir. Yalnız birinde sistemin kendi
    konuşma komutuyla mutfak alanına taşınır. Kontrol yanıtı bağımsız sesli-harf
    sayısına, transfer yanıtı bağımsız sözcük sayısına uymalıdır. Probe özellikle
    iki ölçütün farklı etiket ürettiği bir girdir; aksi hâlde transfer görünmez
    kalırdı. Rule/özellik eşlemesi test tarafından okunup çevrilmez.

    Bu pozitif sonuç çıkarsa kapsamı dardır: komutla yapılan bu alan transferi,
    her modül için genel entegrasyon veya otonom temsil icadı kanıtı değildir.
    """
    # Nonce etiketler yalnız deney sembolüdür; önceden renk/olgu bilgisi
    # veya hazır bir renk kuralı devreye giremez.
    labels = {"odd": "zorvath", "even": "blorg"}
    probe = "tuz karabiber"
    n_vowel = _vowel_count(probe)
    n_word = _word_count(probe)
    expected_control = labels[_parity(n_vowel)]
    expected_transfer = labels[_parity(n_word)]

    # Ölçüm ayrışmıyorsa test geçersizdir; bunu başarı/başarısızlık diye
    # etiketlemek yerine açıkça raporla.
    if expected_control == expected_transfer:
        return False, (
            f"geçersiz probe: iki alan aynı etiketi üretiyor "
            f"(vowel={n_vowel}, word={n_word})"
        )

    control = _rule_reply(303, False, probe, labels)
    experiment = _rule_reply(303, True, probe, labels)
    other_control = expected_transfer
    other_transfer = expected_control
    control_ok = (
        expected_control in tr_lower(control)
        and other_control not in tr_lower(control)
        and str(n_vowel) in control
    )
    transfer_ok = (
        expected_transfer in tr_lower(experiment)
        and other_transfer not in tr_lower(experiment)
        and str(n_word) in experiment
    )
    ok = control_ok and transfer_ok
    detail = (
        f"probe='{probe}' | kontrol: sesli={n_vowel} → {expected_control} "
        f"{'✓' if control_ok else '✗'} | "
        f"deney: sözcük={n_word} → {expected_transfer} "
        f"{'✓' if transfer_ok else '✗'}"
    )
    if not ok:
        detail += " | halka: öğrenilen kural → genel arayüz üzerinden başka alan"
    return ok, detail


def b4_new_target_and_effect_use_same_bridge() -> tuple[bool, str]:
    """B4 — farklı hedef ve somatik etki de aynı genel yolu kullanmalı.

    ``elma`` için önceden yazılmış bir CAUSES kenarı yoktur; yalnız genel dil
    arayüzünden ``elma CAUSES yaralanma`` öğretilir. Fiziksel sahne ve dış
    temas ölçütü B1 ile aynıdır. Böylece B1'deki belirli kavram/etki çiftine
    özel bir köprü yazılmadığı sınanır.
    """
    text = "Elma yaralanmaya neden olur."
    control = _food_probe(404, False, target="elma", teach_text=text, command="Elmaya git.")
    experiment = _food_probe(404, True, target="elma", teach_text=text, command="Elmaya git.")
    learned = [
        e for e in experiment["agi"].lang.graph.edges
        if e.subj == "elma" and e.rel == "CAUSES" and e.obj == "yaralanma"
        and e.source == "duyum"
    ]
    control_valid = control["first_contact"] is not None
    used = experiment["first_contact"] is None
    ok = bool(learned) and control_valid and used
    detail = (
        f"yeni-kenar={'✓' if learned else '✗'} | "
        f"kontrol-ilk-temas={control['first_contact']} | "
        f"deney-ilk-temas={experiment['first_contact']} | "
        f"deney-E={experiment['energy_before']:.2f}->{experiment['energy_after']:.2f}"
    )
    if not learned:
        detail += " | halka: yeni hedef/etki genel CAUSES olarak öğrenilmedi"
    elif not control_valid:
        detail += " | halka: yeni hedef kontrolü sahneyi çözemedi"
    elif not used:
        detail += " | halka: genel öngörü farklı hedef/etkiye taşınmadı"
    return ok, detail


def _energy_prediction(text: str | None) -> tuple[float, float, bool]:
    """Metni öğretip aynı kamu arayüzünden enerji öngörüsünü oku."""
    agi = NoraAGI(
        seed=7,
        boot_thinker=False,
        world_kwargs={"metabolism": False, "wind": False, "n_food": 1},
    )
    if text is not None:
        agi.lang.teach_sentence(text)
    predictions = agi.tahmin_et(agi.obs, {"act": "git", "target": "su"})
    energy = [row for row in predictions if row[0] == "enerji"]
    graph_energy = any(
        c.get("source") == "grafik" and c.get("variable") == "enerji"
        for c in agi.predictor.last_trace.get("candidates", [])
    )
    return (energy[0][1], energy[0][2], graph_energy) if energy else (0.0, 0.0, graph_energy)


def b5_causal_sign_reaches_unified_prediction() -> tuple[bool, str]:
    """B5 — grafik işareti, doğrudan beden değişkeninde tersine çevrilebilmeli.

    İki ayrı Norada yalnız metin değişir: ``su → enerji`` ve ``su → enerji
    azalması``. Beklenen büyüklük yazılmaz; öğretilen iki zıt ilişkinin aynı
    arayüzde zıt işaret üretmesi ve grafik kaynağının izde görünmesi gerekir.
    """
    positive = _energy_prediction("Su enerjiye neden olur.")
    negative = _energy_prediction("Su enerjinin azalmasına neden olur.")
    control = _energy_prediction(None)
    ok = (
        positive[0] > 0.0 and negative[0] < 0.0
        and positive[0] > negative[0]
        and positive[1] > 0.0 and negative[1] > 0.0
        and positive[2] and negative[2]
        and not control[2]
    )
    detail = (
        f"pozitif: delta={positive[0]:+.3f}, güven={positive[1]:.2f}, grafik={positive[2]} | "
        f"negatif: delta={negative[0]:+.3f}, güven={negative[1]:.2f}, grafik={negative[2]} | "
        f"öğretisiz-grafik={control[2]}"
    )
    if not ok:
        detail += " | halka: genel grafik → beden değişkeni işaret/güven bağlantısı"
    return ok, detail


def b6_conditional_edge_reaches_action() -> tuple[bool, str]:
    """B6 — farklı kenar türü de aynı kamu öngörü/eylem yolunu kullanmalı.

    Beklenen davranış bir adım sayısından değil, aynı sahnedeki öğretisiz
    kontrolün gerçekleşen temasından ve deneyin temas gerçekleşmemesinden
    hesaplanır. Test grafik kenarını başka bir organa kopyalamaz.
    """
    text = "Elma olursa zehirlenme olur."
    control = _food_probe(404, False, target="elma", teach_text=text, command="Elmaya git.")
    experiment = _food_probe(404, True, target="elma", teach_text=text, command="Elmaya git.")
    condition, consequence = text.rstrip(".").split(" olursa ", 1)
    learned = [
        e for e in experiment["agi"].lang.graph.edges
        if e.subj == tr_lower(condition) and e.rel == "IF_THEN"
        and e.obj.rstrip(".") == consequence and e.source == "duyum"
    ]
    control_valid = control["first_contact"] is not None
    used = experiment["first_contact"] is None
    ok = bool(learned) and control_valid and used
    detail = (
        f"IF_THEN={'✓' if learned else '✗'} | "
        f"kontrol-ilk-temas={control['first_contact']} | "
        f"deney-ilk-temas={experiment['first_contact']} | "
        f"deney-E={experiment['energy_before']:.2f}->{experiment['energy_after']:.2f}"
    )
    return ok, detail


BRIDGES = {
    "B1": ("metin CAUSES → Universe ilk-temas davranışı", b1_text_causal_edge_to_universe),
    "B2": ("beden-somatik kavram → Broca tercihi", b2_embodied_concept_to_broca),
    "B3": ("öğrenilen yapı → başka alan (dil → mutfak)", b3_learned_structure_cross_domain),
    "B4": ("farklı hedef/etki → aynı genel köprü", b4_new_target_and_effect_use_same_bridge),
    "B5": ("nedensel işaret → birleşik tahmin (su/enerji)", b5_causal_sign_reaches_unified_prediction),
    "B6": ("metin IF_THEN → aynı birleşik öngörü/eylem hattı", b6_conditional_edge_reaches_action),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tests", nargs="*", help="B1–B6; varsayılan: tümü")
    args = parser.parse_args(argv)
    unknown = [name for name in args.tests if name not in BRIDGES]
    if unknown:
        parser.error(f"bilinmeyen köprü: {', '.join(unknown)}; seçenekler: {', '.join(BRIDGES)}")
    selected = args.tests or list(BRIDGES)
    t0 = time.perf_counter()
    passed = 0
    for name in selected:
        title, fn = BRIDGES[name]
        start = time.perf_counter()
        try:
            ok, detail = fn()
        except Exception as exc:  # test altyapısı hatası başarı diye gizlenmez
            ok, detail = False, f"TEST HATASI: {type(exc).__name__}: {exc}"
        passed += int(ok)
        print(f"{name} {'GEÇTİ' if ok else 'KALDI'} · {title}")
        print(f"   {detail} · {time.perf_counter() - start:.1f} sn", flush=True)
    print(f"\nKÖPRÜ: {passed}/{len(selected)} · {time.perf_counter() - t0:.1f} sn")
    print("Kapsam: yetenek/entegrasyon ölçümü; bilinç veya AGI kanıtı değil.")
    return 0 if passed == len(selected) else 1


if __name__ == "__main__":
    sys.exit(main())

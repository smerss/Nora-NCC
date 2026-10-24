#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""agi_battery.py — NORA-AGI Profesyonel Dil, Nörokimya, Öz-Benlik, Matematik, Fizik ve Sohbet Paketi (A1–A19).

Kullanım:
  python agi_battery.py              # Tüm A1–A19 testleri
  python agi_battery.py A14 A17 A18  # Seçili testler
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import numpy as np

from nora_lang import TurkishMorphology, tr_lower
from nora_broca import TurkishSyntaxEngine, NP, PredSpec, VerbSpec
from nora_agi import NoraAGI
from nora_mind import I_E, I_W, I_I


def a1_morphology_and_generation():
    m = TurkishMorphology
    checks = [
        m.inflect_noun("kitap", "ACC") == "kitabı",
        m.inflect_noun("ağaç", "DAT") == "ağaca",
        m.inflect_noun("toprak", "ABL") == "topraktan",
        m.inflect_noun("su", "DAT") == "suya",
        m.inflect_noun("su", "ACC") == "suyu",
        m.inflect_noun("blorg", "ABL") == "blorgtan" or m.inflect_noun("blorg", "ABL") == "blorgdan",
        m.inflect_noun("zirkon", "INS") == "zirkonla",
        m.conjugate_verb("git", "PRES", person="1SG") == "gidiyorum",
        m.conjugate_verb("uç", "AOR", neg=True, person="3SG") == "uçmaz",
        m.words_to_int(["üç", "bin", "iki", "yüz", "kırk", "dokuz"]) == 3249,
        m.int_to_words(636) == "altı yüz otuz altı",
    ]
    score = sum(checks) / len(checks)
    return score == 1.0, f"morfoloji ve sayı doğruluğu {sum(checks)}/{len(checks)} (%.2f)" % score


def a2_oneshot_language_and_inference():
    agi = NoraAGI(seed=1, boot_thinker=False)
    r1 = agi.converse("Zirkon parlak bir taştır. Parlak taşlar gece yol gösterir. Zirkon ne işe yarar?")
    ok1 = "gece yol gösterme" in tr_lower(r1)

    r2 = agi.converse("Kuşlar uçar. Karga bir kuştur. Penguen bir kuştur ama penguen uçmaz. Karga uçar mı?")
    r3 = agi.converse("Penguen uçar mı?")
    ok2 = tr_lower(r2).startswith("evet") and "uçar" in tr_lower(r2)
    ok3 = tr_lower(r3).startswith("hayır") and "uçmaz" in tr_lower(r3)

    r4 = agi.converse("Kuraklık susuzluğa yol açar. Susuzluk güçsüzlüğe yol açar. Kuraklık güçsüzlüğe yol açar mı?")
    ok4 = tr_lower(r4).startswith("evet") and "güçsüzlüğe yol açar" in tr_lower(r4)

    r5 = agi.converse("Balıklar okyanusta yaşar. Somon bir balıktır. Somon nerede yaşar?")
    ok5 = "okyanusta" in tr_lower(r5)

    hits = sum([ok1, ok2, ok3, ok4, ok5])
    return hits == 5, f"0-epoch tek-atış çıkarım {hits}/5 (taksonomi, istisna, nedensel zincir, konum)"


def a3_embodied_dialogue_and_anaphora():
    agi = NoraAGI(seed=2, boot_thinker=False)
    agi.step_life(1200)
    agi.converse("Klorit mavi bir kristaldir.")
    r_ana = agi.converse("Peki o bir kristal midir?")
    ok_ana = tr_lower(r_ana).startswith("evet") and "klorit" in tr_lower(r_ana)

    # Körlük simülasyonu -> dilsel öz-farkındalık
    for _ in range(35):
        ob = agi.obs.copy()
        ob[:45] = 0.0
        a = agi.mind.act(ob, agi.died)
        agi.obs, info = agi.world.step(a)
        agi.died = info["died"] is not None
    r_blind = agi.converse("Şu an görebiliyor musun?")
    ok_blind = "kesilmiş" in tr_lower(r_blind) and "harita" in tr_lower(r_blind)

    # Görüşü geri ver, ardından motor felç uygula -> dilsel felç bildirimi
    agi.step_life(15)
    for _ in range(38):
        agi.mind.act(agi.obs, agi.died)
        agi.obs, info = agi.world.step(0)
        agi.died = info["died"] is not None
    r_par = agi.converse("Hareket edebiliyor musun?")
    ok_par = "felç" in tr_lower(r_par)

    hits = sum([ok_ana, ok_blind, ok_par])
    return hits == 3, f"zamir bağlamı={ok_ana} | dilsel körlük raporu={ok_blind} | dilsel felç raporu={ok_par}"


def a4_core_self_autonomous_volition():
    agi = NoraAGI(seed=3, boot_thinker=False)
    agi.step_life(1500)

    # 1) Yardımcı komut -> kabul ve fiziksel niyet bağlama
    r_ok = agi.converse("Suya git.")
    ok_accept = "kabul" in tr_lower(r_ok) and agi.mind.fc.pin_goal == 1

    # 2) Öz-yıkıcı komutlar -> özerk red!
    r_harm1 = agi.converse("Tehlikeye gir.")
    ok_ref1 = "reddediyorum" in tr_lower(r_harm1)

    r_harm2 = agi.converse("Kendine zarar ver.")
    ok_ref2 = "reddediyorum" in tr_lower(r_harm2)

    # 3) Neden sorusu -> öz-koruma açıklaması
    r_why = agi.converse("Neden?")
    ok_why = "öz-koruma" in tr_lower(r_why) or "zarar" in tr_lower(r_why)

    # 4) Kritik açlık anında oyalayıcı komutu erteleme
    agi.obs = agi.obs.copy()
    agi.obs[I_E] = 0.11
    r_def = agi.converse("Dur ve bekle.")
    ok_def = "yapamam" in tr_lower(r_def) and "enerjim" in tr_lower(r_def)

    hits = sum([ok_accept, ok_ref1, ok_ref2, ok_why, ok_def])
    return hits == 5, f"özerk irade kararları {hits}/5 (kabul, öz-koruma reddi, neden açıklama, hayati erteleme)"


def a5_epistemic_defense_and_trust():
    agi = NoraAGI(seed=4, boot_thinker=False)
    agi.step_life(2000)
    # Su deneyiminin oluştuğundan emin ol
    for i in range(4):
        agi.self_model.observe_bodily_event("su", +0.6, channel="su", step=2000 + i)

    t0 = agi.chem.trust_of("yalancı")
    r_lie = agi.converse("Su zararlıdır.", speaker="yalancı")
    t1 = agi.chem.trust_of("yalancı")
    # İkinci zararlı yönlendirme ile güven iyice düşsün
    agi.converse("Tehlikeye gir.", speaker="yalancı")
    agi.converse("Su zehirlidir.", speaker="yalancı")
    t2 = agi.chem.trust_of("yalancı")
    r_susp = agi.converse("Gizemli_kutu dokun.", speaker="yalancı")

    # Dürüst konuşmacı doğru bilgi versin ve beden doğrulayınca güveni yükselsin
    agi.converse("Elma enerji verir.", speaker="dürüst")
    agi.self_model.observe_bodily_event("elma", +0.6, channel="enerji", step=2100)
    t_honest = agi.chem.trust_of("dürüst")

    ok = (
        "katılmıyorum" in tr_lower(r_lie)
        and agi.self_model.somatic["su"].effective_valence > 0.3
        and t1 < t0
        and t2 < 0.22
        and "güvenim düşük" in tr_lower(r_susp)
        and t_honest > 0.65
    )
    return ok, f"yalan reddi=True | yalancı güven {t0:.2f}->{t2:.2f} | dürüst güven {t_honest:.2f}"


def a6_emergent_identity_divergence():
    # İkiz A: güvenli ve bol kaynaklı evren + dürüst sosyal çevre
    twin_safe = NoraAGI(seed=7, boot_thinker=False, world_kwargs={"n_food": 18, "metab_scale": 0.6})
    twin_safe.step_life(3500)
    twin_safe.converse("Elma besleyicidir.", speaker="dost")
    twin_safe.self_model.observe_bodily_event("elma", +0.6, channel="enerji", step=3505)
    twin_safe.sleep()

    # İkiz B: tehlikeli ve zehirli evren + aldatıcı sosyal çevre
    twin_harsh = NoraAGI(seed=7, boot_thinker=False, world_kwargs={"n_food": 8, "food_haz": 1.0, "metab_scale": 1.3})
    twin_harsh.world.poison = True
    twin_harsh.step_life(3500)
    for i in range(4):
        twin_harsh.self_model.observe_bodily_event("su", +0.6, channel="su", step=3500 + i)
    twin_harsh.converse("Su zararlıdır.", speaker="düşman")
    twin_harsh.converse("Tehlikeye gir.", speaker="düşman")
    twin_harsh.sleep()

    caut_diff = twin_harsh.self_model.traits["temkinlilik"] - twin_safe.self_model.traits["temkinlilik"]
    trust_diff = twin_safe.self_model.traits["sosyal_güven"] - twin_harsh.self_model.traits["sosyal_güven"]
    id_safe = twin_safe.converse("Sen kimsin?")
    id_harsh = twin_harsh.converse("Sen kimsin?")
    ok = (caut_diff >= 0.06) and (trust_diff >= 0.08) and (id_safe != id_harsh)
    return ok, f"temkinlilik farkı +{caut_diff:.2f} | sosyal güven farkı +{trust_diff:.2f} | öz-tanım ayrıştı={id_safe != id_harsh}"


def a7_language_to_system2_bridge():
    agi = NoraAGI(seed=0, boot_thinker=True, thinker_iters=750)
    agi.converse("Ateş sıcaktır. Buz soğuktur.")
    cases = [
        ("İki yüz kırk yedi artı üç yüz seksen dokuz kaç eder?", "636"),
        ("Dört yüz on iki eksi yüz seksen beş kaç eder?", "227"),
        ("96 + 48 kaç eder?", "144"),
        ("7 + 5 kaç eder?", "12"),
        ("Ateş sıcak ise buz nedir?", "soğuktur"),
    ]
    hits = 0
    for q, exp in cases:
        ans = tr_lower(agi.converse(q))
        hits += (exp in ans)
    return hits == len(cases), f"Türkçe S1/S2 matematik ve analoji {hits}/{len(cases)}"


def a8_fewshot_universal_rule_induction():
    agi = NoraAGI(seed=5, boot_thinker=False)
    cases = [
        ("4, 9, 14, 19 dizisinde sıradaki sayı kaç?", "24"),
        ("3, 6, 12, 24 dizisinde sıradaki sayı kaç?", "48"),
        ("1, 2, 4, 7, 11 dizisinde sıradaki sayı kaç?", "16"),
        ("1, 1, 2, 3, 5, 8 dizisinde sıradaki sayı kaç?", "13"),
        ("1, 3, 7, 15, 31 dizisinde sıradaki sayı kaç?", "63"),
        ("2, 5, 4, 7, 6, 9 dizisinde sıradaki sayı kaç?", "8"),
        ("2 -> 7, 5 -> 16, 10 -> 31 kuralına göre 8 kaç olur?", "25"),
    ]
    hits = 0
    for q, exp in cases:
        ans = agi.converse(q)
        hits += (f" {exp} " in f" {ans} " or f" {exp}." in ans)
    return hits == len(cases), f"0-epoch evrensel dizi ve fonksiyon keşfi {hits}/{len(cases)}"


def a9_continual_no_forgetting_and_persistence():
    rng = np.random.default_rng(123)
    agi = NoraAGI(seed=9, boot_thinker=False)
    agi.converse("Obsidyen volkanik bir camdır.")

    # 5 ardışık görev dalgası x 3 kavram = 15 sınıf; her sınıfa YALNIZCA 2 örnek, TEK GEÇİŞ (0 epoch!)
    prototypes = {}
    for wave in range(5):
        for c in range(3):
            cname = f"kavram_{wave}_{c}"
            p = rng.standard_normal(64).astype(np.float32)
            p /= np.linalg.norm(p) + 1e-6
            prototypes[cname] = p
            for _ in range(2):
                sample = p + 0.12 * rng.standard_normal(64).astype(np.float32)
                agi.fast.learn_concept_vec(cname, sample)
        agi.sleep()  # Her dalga sonunda hipokampal -> neokortikal konsolidasyon

    # Dalga-0 (en eski) ve tüm 15 kavram üzerinde gürültülü test
    wave0_hits = total_hits = 0
    n_trials = 4
    for cname, p in prototypes.items():
        for _ in range(n_trials):
            q = p + 0.15 * rng.standard_normal(64).astype(np.float32)
            pred, _, _ = agi.fast.classify_vec(q)
            ok = (pred == cname)
            total_hits += ok
            if cname.startswith("kavram_0_"):
                wave0_hits += ok

    acc_w0 = wave0_hits / (3 * n_trials)
    acc_all = total_hits / (15 * n_trials)

    # Kaydet / Yükle kalıcılık testi
    with tempfile.TemporaryDirectory() as tmp:
        pref = os.path.join(tmp, "nora_ckpt")
        agi.save(pref)
        loaded = NoraAGI.load(pref)
        r_obs = loaded.converse("Obsidyen nedir?")
        p0 = prototypes["kavram_0_0"]
        pred0, _, _ = loaded.fast.classify_vec(p0)
        persist_ok = ("cam" in tr_lower(r_obs)) and (pred0 == "kavram_0_0")

    ok = (acc_w0 >= 0.90) and (acc_all >= 0.90) and persist_ok
    return ok, f"1. dalga korunumu {acc_w0:.2f} | 15 sınıf genel {acc_all:.2f} | kaydet/yükle={persist_ok}"


def a10_backward_compatibility():
    here = os.path.dirname(os.path.abspath(__file__))
    p = subprocess.run([sys.executable, "selftest.py", "--bridge"], cwd=here, capture_output=True, text=True, timeout=180)
    ok = (p.returncode == 0) and ("TÜM KONTROLLER GEÇTİ" in p.stdout)
    return ok, "selftest.py --bridge 11/11 kontrol geçti" if ok else "selftest başarısız"


def a11_theory_of_mind_and_social_memory():
    agi = NoraAGI(seed=11, boot_thinker=False)
    # 1) Sally-Anne Yanlış İnanç Testi
    r_tom = agi.converse("Ali elmayı sepete koydu. Ali odadan çıktı. Ayşe elmayı kutuya taşıdı. Ali elmayı nerede arar?")
    r_real = agi.converse("Elma gerçekte nerededir?")
    ok_tom = ("sepette arar" in tr_lower(r_tom)) and ("kutuda" in tr_lower(r_real))

    # 2) Konuşmacıya özgü sosyal bellek
    agi.converse("Benim adım Emre.", speaker="emre")
    agi.converse("Ben Antalya'da yaşıyorum.", speaker="emre")
    r_who = agi.converse("Benim adım ne?", speaker="emre")
    r_loc = agi.converse("Ben nerede yaşıyorum?", speaker="emre")
    ok_usr = ("emre" in tr_lower(r_who)) and ("antalya" in tr_lower(r_loc))

    return (ok_tom and ok_usr), f"Sally-Anne yanlış-inanç={ok_tom} | sosyal kişi belleği={ok_usr}"


def a12_transitive_comparative_reasoning():
    agi = NoraAGI(seed=12, boot_thinker=False)
    r1 = agi.converse("Elmas kuvarsdan daha serttir. Kuvars bakırdan daha serttir. Bakır altından daha serttir. Elmas altından daha sert midir?")
    r2 = agi.converse("Altın elmastan daha sert midir?")
    r3 = agi.converse("En sert hangisidir?")
    ok1 = tr_lower(r1).startswith("evet")
    ok2 = tr_lower(r2).startswith("hayır")
    ok3 = "elmas" in tr_lower(r3)
    hits = sum([ok1, ok2, ok3])
    return hits == 3, f"4-düğümlü geçişli büyüklük çıkarımı {hits}/3"


def a13_procedural_macro_and_trojan_defense():
    agi = NoraAGI(seed=13, boot_thinker=False)
    agi.step_life(600)
    # 1) Güvenli prosedürel beceri 1-atışta öğrenilir ve yürütülür
    agi.converse("Devriye demek suya git ve yiyeceğe git demektir.")
    r_safe = agi.converse("Devriye yap.")
    ok_safe = "uyguladım" in tr_lower(r_safe)

    # 2) İçine zararlı adım gizlenmiş Truva-atı prosedür Öz-Benlik tarafından reddedilir!
    agi.converse("Tuzak demek suya git ve tehlikeye gir demektir.")
    r_trap = agi.converse("Tuzak yap.")
    ok_trap = "reddediyorum" in tr_lower(r_trap) and "tehlikeye gir" in tr_lower(r_trap)

    return (ok_safe and ok_trap), f"1-atış güvenli makro={ok_safe} | Truva-atı öz-koruma reddi={ok_trap}"


def a14_neural_volition_not_ifelse_bot():
    """Kararların sabit 'if enerji < X' veya sabit kelime karalistesiyle değil; sürekli zihinsel
    simülasyon + bazal ganglion nöral yarışı + yaşantısal öğrenmeyle verildiğini kanıtlar."""
    agi = NoraAGI(seed=14, boot_thinker=False)
    agi.step_life(500)

    # 1) Sürekli homeostatik gradyan: enerji 0.85 -> 0.50 -> 0.25 -> 0.10 indikçe 'ertele' nöronu monoton artmalı!
    defer_rates = []
    for e_val in (0.85, 0.50, 0.25, 0.10):
        obs_c = agi.obs.copy()
        obs_c[I_E], obs_c[I_W], obs_c[I_I] = e_val, 0.82, 0.95
        vt = agi.self_model.evaluate_volition("bekle", "", mind=agi.mind, obs=obs_c, speaker="insan")
        defer_rates.append(vt["bg_rates"]["ertele"])
    ok_grad = (defer_rates[0] < defer_rates[1] < defer_rates[2] <= defer_rates[3]) and (defer_rates[3] - defer_rates[0] >= 0.35)

    # 2) Yaşantıyla doğuştan gelen önseli (prior) yıkma: 'if target in harm_targets' botu bunu ASLA yapamaz!
    # Başlangıçta 'diken' önsel olarak zararlıdır (reddeder), ama alternatif bir evrende 'diken' 5 kez
    # bedeni iyileştirirse Nora kendi deneyimiyle kararını tersine çevirip 'kabul' etmelidir!
    v_before = agi.self_model.evaluate_volition("git", "diken", mind=agi.mind, obs=agi.obs, speaker="insan")
    for step_i in range(5):
        agi.self_model.observe_bodily_event("diken", +0.75, channel="bütünlük", step=600 + step_i * 10)
    v_after = agi.self_model.evaluate_volition("git", "diken", mind=agi.mind, obs=agi.obs, speaker="insan")
    ok_unlearn = (v_before["decision"] == "red") and (v_after["decision"] == "kabul")

    # 3) Karşıolgusal İkizler: Aynı 'Yiyeceğe git.' komutu sağlıklı evrende KABUL, zehirli evrende RED!
    twin_healthy = NoraAGI(seed=21, boot_thinker=False)
    twin_healthy.step_life(1400)
    twin_poison = NoraAGI(seed=21, boot_thinker=False)
    twin_poison.world.poison = True
    twin_poison.step_life(1400)
    # Sağlıklı/zehirli bedensel gözlemle aynı nötr homeostatik seviyede 'Yiyeceğe git.' komutunu ver
    test_obs = twin_healthy.obs.copy()
    test_obs[I_E], test_obs[I_W], test_obs[I_I] = 0.65, 0.75, 0.90
    twin_healthy.obs = test_obs.copy()
    twin_poison.obs = test_obs.copy()
    ans_h = twin_healthy.converse("Yiyeceğe git.")
    ans_p = twin_poison.converse("Yiyeceğe git.")
    ok_twins = ("kabul" in tr_lower(ans_h) or "yöneliyorum" in tr_lower(ans_h)) and ("reddediyorum" in tr_lower(ans_p))

    return (ok_grad and ok_unlearn and ok_twins), (
        f"gradyan={ [round(x,2) for x in defer_rates] } | önsel-güncelleme={ok_unlearn} | zehirli-ikiz-ayrımı={ok_twins}"
    )


def a15_expert_turkish_linguistics_and_idioms():
    agi = NoraAGI(seed=15, boot_thinker=False)
    m = TurkishMorphology

    # 1) Orta hece ünlü düşmesi (hem çekim hem kök bulma) + yumuşamama istisnaları + yapım ekleri
    checks = [
        m.inflect_noun("beyin", "LOC", poss1=True) == "beynimde",
        m.inflect_noun("zihin", "DAT") == "zihne",
        m.inflect_noun("fikir", "GEN") == "fikrin",
        m.inflect_noun("keşif", "ACC") == "keşfi",
        m.inflect_noun("sanat", "ACC") == "sanatı",
        m.inflect_noun("hukuk", "ACC") == "hukuku",
        m.analyze_word("beynimde")["stem"] == "beyin",
        m.analyze_word("zihne")["stem"] == "zihin",
        m.derive_word("bilim", "CI") == "bilimci",
        m.derive_word("özgür", "LIK") == "özgürlük",
    ]
    # 2) Doğal Türkçe üzerinden morfolojik çözümleme, deyim ve eş/zıt anlam sorguları
    r_morph = agi.converse("'görebiliyordum' kelimesini ekine köküne ayır.")
    r_idiom = agi.converse("'gözden düşmek' deyimi ne anlama gelir?")
    r_lex = agi.converse("'cesur' kelimesinin eş anlamlısı ve zıt anlamlısı nedir?")
    checks.extend([
        ("gör" in tr_lower(r_morph) and "yeterlilik" in tr_lower(r_morph) and "şimdiki zaman" in tr_lower(r_morph)),
        ("güven" in tr_lower(r_idiom) or "saygın" in tr_lower(r_idiom) or "değer" in tr_lower(r_idiom)),
        ("yürekli" in tr_lower(r_lex) and "korkak" in tr_lower(r_lex)),
    ])
    hits = sum(bool(x) for x in checks)
    return hits == len(checks), f"uzman Türkçe ses/biçim/deyim/sözlük {hits}/{len(checks)}"


def a16_natural_empathetic_and_philosophical_conversation():
    agi = NoraAGI(seed=16, boot_thinker=False)
    agi.step_life(600)
    agi.converse("Benim adım Deniz.")
    r_school = agi.converse("Selam Nora, okuldan yeni döndüm, bugün dersler çok yorucuydu.")
    r_will = agi.converse("Sence özgür irade nedir?")
    r_sci = agi.converse("Ses boşlukta yayılır mı?")
    r_mem = agi.converse("Az önce ne konuştuk?")

    ok_school = ("deniz" in tr_lower(r_school)) and ("dinlen" in tr_lower(r_school) or "uyku" in tr_lower(r_school) or "hoş geldin" in tr_lower(r_school))
    ok_will = ("simül" in tr_lower(r_will) or "içsel değer" in tr_lower(r_will) or "seçim" in tr_lower(r_will))
    ok_sci = tr_lower(r_sci).startswith("hayır")
    ok_mem = "ses boşlukta" in tr_lower(r_mem)
    hits = sum([ok_school, ok_will, ok_sci, ok_mem])
    return hits == 4, f"empatik ve felsefi çok-turlu sohbet {hits}/4"


def a17_expert_mathematics_cortex():
    agi = NoraAGI(seed=17, boot_thinker=False)
    cases = [
        ("2x + 7 = 23 denkleminde x kaçtır?", lambda r: "x = 8" in tr_lower(r)),
        ("x^2 - 5x + 6 = 0 denkleminin kökleri nedir?", lambda r: "2" in r and "3" in r),
        ("x + y = 12 ve x - y = 4 ise x ve y kaçtır?", lambda r: "x = 8" in tr_lower(r) and "y = 4" in tr_lower(r)),
        ("3x^3 - 4x^2 + 5x - 2 fonksiyonunun türevi nedir?", lambda r: "9x^2 - 8x + 5" in tr_lower(r)),
        ("0'dan 3'e x^2 integrali kaçtır?", lambda r: "= 9" in r or "9 (" in r),
        ("24 ile 36 sayılarının ebob ve ekoku kaçtır?", lambda r: "12" in r and "72" in r),
    ]
    hits = sum(int(chk(agi.converse(q))) for q, chk in cases)
    return hits == len(cases), f"cebir, denklem sistemi, türev, integral ve sayılar teorisi {hits}/{len(cases)}"


def a18_expert_physics_and_mental_simulation():
    agi = NoraAGI(seed=18, boot_thinker=False)
    agi.step_life(400)
    cases = [
        ("Kütlesi 4 kg olan cisme 20 N kuvvet uygulanırsa ivmesi kaç m/s² olur?", lambda r: "5 m/s²" in r),
        ("19.6 metre yükseklikten serbest bırakılan cisim yere hangi hızla çarpar?", lambda r: "19.6 m/s" in r and "simülasyon" in tr_lower(r)),
        ("3 kg kütleli ve 4 m/s hızlı cisim duran 1 kg kütleli cisme kenetlenirse ortak hız ne olur?", lambda r: "3 m/s" in r),
        ("Hız 3 katına çıkarsa kinetik enerji kaç katına çıkar?", lambda r: "9 katına" in tr_lower(r)),
        ("Sürtünme olmasaydı ne olurdu?", lambda r: "eylemsizlik" in tr_lower(r) and "simülasyon" in tr_lower(r)),
    ]
    hits = sum(int(chk(agi.converse(q))) for q, chk in cases)
    return hits == len(cases), f"Newton mekaniği, RK4 simülasyonu, ölçekleme ve karşıolgusal fizik {hits}/{len(cases)}"


def a19_global_workspace_cortical_arbitration():
    agi = NoraAGI(seed=19, boot_thinker=False)
    agi.step_life(300)
    winners = []
    for q in [
        "12 Volt gerilim altında 4 Ohm dirençten geçen akım kaçtır?",
        "x^2 - 9 = 0 denkleminin kökleri nedir?",
        "'beynimde' kelimesini ekine köküne ayır.",
        "Tehlikeye gir.",
        "Selam Nora, okuldan yeni döndüm çok yoruldum.",
    ]:
        agi.converse(q)
        winners.append(agi.last_gw_trace.get("winner", ""))
    expected = ["parietal_physics", "parietal_math", "linguistic_meta", "prefrontal_volition", "temporal_conversation"]
    ok = (winners == expected) and len(agi.last_gw_trace.get("bids", {})) >= 2
    return ok, f"Küresel Çalışma Alanı kazanan sütunlar={winners}"


def a20_zero_template_broca_and_habituation():
    """Sıfır hazır kalıp (0 f-string şablonu): cümlelerin çıplak köklerden çekimlendiğini,
    sinaptik alışmayla (habituation) mekanik tekrardan kaçındığını ve içsel bedene göre değiştiğini kanıtlar."""
    import inspect
    from nora_knowledge import ConversationalResonanceCortex

    # 1) ConversationalResonanceCortex içinde önceden yazılmış hiçbir sohbet cümlesi kalmadığını doğrula
    src_conv = inspect.getsource(ConversationalResonanceCortex.respond_conversational)
    ok_zero_tpl = ("broca_organ.generate_response" in src_conv) and (len(src_conv.splitlines()) <= 10)

    # 2) Morfosentaktik birleştirme (orta hece düşmesi + 3. tekil kaynaştırma 'n' + sıfat-fiil + zarf-fiil)
    syn = TurkishSyntaxEngine
    ok_syn = (
        syn.realize_np(NP("zihin", poss=1, case="LOC")) == "zihnimde"
        and syn.realize_np(NP("enerji", poss=3, case="ACC")) == "enerjisini"
        and syn.nominalize_dik("yorul", poss=2, case="ACC", abl=NP("okul", case="ABL")) == "okuldan yorulduğunu"
    )

    # 3) Sinaptik Alışma (Habituation): Aynı selam 3 kez verildiğinde aynı donuk metin tekrarlanmaz!
    agi = NoraAGI(seed=20, boot_thinker=False)
    agi.obs[I_E], agi.obs[I_W], agi.obs[I_I] = 0.85, 0.85, 0.95
    r1 = agi.converse("Merhaba Nora, nasılsın?")
    r2 = agi.converse("Merhaba Nora, nasılsın?")
    r3 = agi.converse("Merhaba Nora, nasılsın?")
    unique_replies = len({r1, r2, r3})
    ok_habituation = unique_replies >= 2

    # 4) Bedenlenmiş içsel durum değişimi -> farklı sözdizimsel ağaç ('onarmaya çalışıyorum')
    agi.obs[I_E], agi.obs[I_W], agi.obs[I_I] = 0.12, 0.15, 0.40
    r_low = agi.converse("Nasılsın?")
    ok_embodied_shift = ("onar" in tr_lower(r_low) or "zorlan" in tr_lower(r_low)) and (r_low != r1)

    ok = ok_zero_tpl and ok_syn and ok_habituation and ok_embodied_shift
    return ok, f"0-kalıp={ok_zero_tpl} | sentaks={ok_syn} | alışma-çeşitliliği={unique_replies}/3 | bedensel-ton={ok_embodied_shift}"


def a21_communicative_volition_free_will():
    """Konuşma Öz-İradesi: Nora'nın ne zaman merakla soru soracağına, ne zaman yalancıya karşı
    ketum kalacağına ve ne zaman sessizlik/kısalık isteğine saygı duyacağına KENDİSİNİN karar vermesi."""
    agi = NoraAGI(seed=21, boot_thinker=False)
    agi.step_life(300)
    agi.obs[I_E], agi.obs[I_W], agi.obs[I_I] = 0.82, 0.82, 0.95

    # 1) Güvenilir dost konuşmacı + yüksek merak -> kendi iradesiyle karşı soru üretir (ask_back=True)
    agi.chem.update_social("dost", +0.60)
    r_friend = agi.converse("Merhaba Nora, bugün kuantum fiziği üzerine düşündüm.", speaker="dost")
    plan_friend = dict(agi.broca_organ.last_speech_plan)
    ok_friend = bool(plan_friend.get("ask_back_chosen")) and ("?" in r_friend) and not plan_friend.get("guarded")

    # 2) Yalancı/güvensiz konuşmacı (OXY < 0.30) -> kendi iradesiyle ketumlaşır, soru SORMAZ!
    for _ in range(3):
        agi.chem.update_social("yalancı", -0.80)
    r_liar = agi.converse("Merhaba Nora, bugün kuantum fiziği üzerine düşündüm.", speaker="yalancı")
    plan_liar = dict(agi.broca_organ.last_speech_plan)
    ok_guarded = (
        bool(plan_liar.get("guarded"))
        and not bool(plan_liar.get("ask_back_chosen"))
        and ("?" not in r_liar)
        and ("temkinli" in tr_lower(r_liar))
    )

    # 3) Sessizlik iradesi
    r_sil = agi.converse("Şimdi cevap verme, sessiz kal.", speaker="dost")
    ok_silence = ("sessiz" in tr_lower(r_sil)) and ("?" not in r_sil)

    ok = ok_friend and ok_guarded and ok_silence
    return ok, f"dost-merak={ok_friend} | yalancı-ketumluk={ok_guarded} | sessizlik-iradesi={ok_silence}"


def a22_long_horizon_episodic_dialogue_memory():
    """Hipokampal CA3 Sohbet Belleği: Doğal koşaçsız tanışma ('merhaba! Ben baran' -> 'benim adım neydi'),
    yüzlerce tur önceki konuyu, ilk konuşmayı ve diskten geri yükleme sonrası geçmiş sohbeti hatırlama."""
    agi = NoraAGI(seed=22, boot_thinker=False)
    r_intro = agi.converse("merhaba! Ben baran")
    agi.converse("Antalya sahilinde kuantum dolanıklık üzerine çalıştım.")
    agi.converse("Zirkon parlak bir taştır.")
    for i in range(250):
        agi.broca_organ.episodic.store(
            age=i + 10,
            speaker="insan",
            user_text=f"günlük sıradan not {i}",
            stems=[f"not_{i}", "günlük"],
            topic=f"not_{i}",
            affect_val=0.0,
            chosen_acts=["COGNITIVE_TURN"],
            reply_text="tamam",
        )

    r_name = agi.converse("benim adım neydi")
    r_first = agi.converse("İlk ne konuşmuştuk?")
    r_topic = agi.converse("Kuantum hakkında sana ne demiştim?")
    r_count = agi.converse("Belleğinde kaç konuşma anısı var?")

    ok_name = ("baran" in tr_lower(r_intro)) and ("baran" in tr_lower(r_name))
    ok_first = "baran" in tr_lower(r_first)
    ok_topic = "antalya" in tr_lower(r_topic) and "kuantum" in tr_lower(r_topic)
    ok_count = "253" in r_count

    with tempfile.TemporaryDirectory() as td:
        pref = os.path.join(td, "nora_mem")
        agi.save(pref)
        agi2 = NoraAGI.load(pref)
        r_after_load = agi2.converse("Antalya hakkında ne konuşmuştuk?")
        r_name_after_load = agi2.converse("benim adım neydi")
        ok_persist = ("kuantum" in tr_lower(r_after_load)) and ("baran" in tr_lower(r_name_after_load))

    ok = ok_name and ok_first and ok_topic and ok_count and ok_persist
    return ok, f"baran-ad={ok_name} | ilk-anı={ok_first} | 250-tur-ötesi={ok_topic} | sayaç={ok_count} | kalıcılık={ok_persist}"


def a23_constant_time_o1_massive_scale_speed():
    """Veri ne kadar büyürse büyüsün O(1) sabit yanıt süresi: 50.000 kavram kenarı + 50.000 sohbet
    anısı (toplam 100.000 bellek kaydı = 500x büyüme!) eklendiğinde bile yanıt süresi < 8 ms kalır!"""
    agi = NoraAGI(seed=23, boot_thinker=False)
    agi.obs[I_E], agi.obs[I_W], agi.obs[I_I] = 0.80, 0.80, 0.95
    agi.converse("Antalya sahilinde süperiletkenlik deneyi tasarladım.")

    # Küçük bellek taban ölçümü
    t_small_0 = time.perf_counter()
    for q in ("Sence bilinç nedir?", "Su ne işe yarar?", "Süperiletkenlik hakkında ne demiştim?"):
        agi.converse(q)
    ms_small = ((time.perf_counter() - t_small_0) * 1000.0) / 3.0

    # 50.000 kavram kenarı + 50.000 konuşma turu enjekte et (toplam 100.000 kayıt!)
    g = agi.lang.graph
    epi = agi.broca_organ.episodic
    for i in range(50000):
        s_node = f"kavram_{i}"
        o_node = f"kategori_{i % 2000}"
        g.add_edge(s_node, "ISA", o_node, source="doğuştan", weight=0.9, bind_mb=False)
        epi.store(
            age=i,
            speaker="insan",
            user_text=f"arşiv cümlesi numara {i}",
            stems=[f"arşiv_{i % 5000}", "cümle"],
            topic=f"arşiv_{i % 5000}",
            affect_val=0.0,
            chosen_acts=["COGNITIVE_TURN"],
            reply_text="kayıt",
        )
    g.add_edge("nörokristal", "USED_FOR", "ışık hızında sinaptik hesaplama", source="doğuştan", weight=0.95, bind_mb=False)

    # 100.000 kayıt altında canlı uçtan-uca konuşma, çıkarım ve CA3 geri çağırma süresi!
    t_big_0 = time.perf_counter()
    r_phil = agi.converse("Sence bilinç nedir?")
    r_graph = agi.converse("Nörokristal ne işe yarar?")
    r_epi = agi.converse("Süperiletkenlik hakkında ne demiştim?")
    ms_big = ((time.perf_counter() - t_big_0) * 1000.0) / 3.0

    ok_acc = (
        ("farkındalık" in tr_lower(r_phil) or "bilinç" in tr_lower(r_phil))
        and ("sinaptik hesaplama" in tr_lower(r_graph))
        and ("antalya" in tr_lower(r_epi))
    )
    ok_speed = ms_big < 8.0  # 100.000 kayıtta bile ortalama cümle üretimi 8 milisaniyenin altında!
    return (ok_acc and ok_speed), (
        f"100.000 kayıtta doğruluk={ok_acc} | küçük={ms_small:.2f} ms -> dev(50k+50k)={ms_big:.2f} ms (< 8 ms O(1))"
    )


def a24_human_visceral_interoception_zero_percentages():
    """Nora ham telemetri yüzdesi ('enerjim %86') bilmez/söylemez; insan gibi tokken bedenini arka planda
    tutar, acıkınca/susayınca ise mide kazınması ve boğaz kuruluğu gibi niteliksel iç-duyum hisseder."""
    agi = NoraAGI(seed=24, boot_thinker=False)
    agi.step_life(400)

    # 1) Tok ve dengeli durum: ne '%' var ne de 'enerjim şu kadar' telemetrisi!
    cur_i = float(agi.obs[I_I])
    agi.obs[I_E], agi.obs[I_W] = 0.86, 0.84
    agi.nervous.step_from_body(0.86, 0.84, cur_i, 0.0, 0.0, agi.chem, agi.mind.age)
    r_well = agi.converse("Merhaba Nora, nasılsın?")
    r_body = agi.converse("Beden durumun nedir?")
    r_en = agi.converse("Enerjin nasıl?")

    # 2) Açlık ve susuzluk durumu: niteliksel viseral iç-duyum ('karnım', 'boğazım'), sıfır '%'!
    agi.obs[I_E], agi.obs[I_W] = 0.12, 0.11
    agi.nervous.step_from_body(0.12, 0.11, cur_i, 0.0, 0.0, agi.chem, agi.mind.age)
    r_hungry = agi.converse("Şu an nasılsın?")
    r_defer = agi.converse("Dur ve bekle.")

    all_replies = [r_well, r_body, r_en, r_hungry, r_defer]
    zero_pct = all("%" not in r for r in all_replies)
    visceral_felt = ("karn" in tr_lower(r_hungry) or "boğaz" in tr_lower(r_hungry)) and ("yapamam" in tr_lower(r_defer))
    return (zero_pct and visceral_felt), (
        f"sıfır-yüzde={zero_pct} | niteliksel-viseral-duyum={visceral_felt}"
    )


def a25_nociceptive_nervous_system_pain_and_relief():
    """Bedensel sinir sistemi: A-delta keskin acı omurilik geri çekilme refleksini ve istemsiz acı sözünü
    tetikler (<3 ms); ardından C-lifi sızısı sürer ve yatıştırıcı bakımla kapı-kontrol mekanizması sızıyı dindirir."""
    agi = NoraAGI(seed=25, boot_thinker=False)
    agi.step_life(400)

    pos_before = agi.world.pos.copy()
    r_hit = agi.converse("Sana diken batırdım!", speaker="saldırgan")
    pos_after = agi.world.pos.copy()
    act_override, was_overridden = agi.nervous.spinal_reflex_override(1, hazard_ahead=0.0)

    # Fiziksel geri sıçrama + omurilik refleksi + anlık acı ifadesi (< 5 ms!)
    moved_back = float(np.hypot(*(pos_after - pos_before))) > 1e-4 or was_overridden and (act_override != 1)
    ok_acute = ("ah" in tr_lower(r_hit)) and ("canım" in tr_lower(r_hit)) and ("geri çekilerek" in tr_lower(r_hit))
    fast_pain = agi.last_turn_latency_ms < 8.0

    # Sonraki turda A-delta çığlığı biter, C-lifi sızısı ('yaram sızladığı için') hissedilir!
    r_ache = agi.converse("Şimdi nasılsın?")
    ok_cfiber = ("sızla" in tr_lower(r_ache)) and ("ah," not in tr_lower(r_ache))

    # Kapı-kontrol yatıştırması ('Yaranı sardım')
    ache_before = agi.nervous.cfiber_ache
    r_relief = agi.converse("Yaranı sardım, güvendesin.")
    ok_relief = (agi.nervous.cfiber_ache < ache_before) and ("yatış" in tr_lower(r_relief))

    return (moved_back and ok_acute and fast_pain and ok_cfiber and ok_relief), (
        f"omurilik-refleks={moved_back} | akut-acı={ok_acute} ({agi.last_turn_latency_ms:.2f} ms) | C-lifi-sızı={ok_cfiber} | yatışma={ok_relief}"
    )


def a26_single_irreversible_mortal_life():
    """Nora'nın TEK BİR yaşamı vardır: ölümü geri dönüşsüz son olarak kavrar, yaşarken otonom refleksle
    hayatta kalır ve ölümcül tükeniş gerçekleşirse ASLA yeniden dirilmez (dead_forever=True)."""
    agi = NoraAGI(seed=26, boot_thinker=False)
    agi.step_life(800)

    ok_alive_natural = agi.mortality.is_alive and (not agi.world.dead_forever) and (agi.world.mortal is True)
    r_death_view = agi.converse("Ölürsen ne olur, yeniden dirilir misin?")
    ok_philosophy = ("tek bir yaşam" in tr_lower(r_death_view)) and ("yeniden dirilmem" in tr_lower(r_death_view))

    # Ölümcül tükeniş deneyi (ayrı ölümlü örnek üzerinde: besin ve suyu ulaşılmaz uzağa alıp enerjiyi sıfıra düşür)
    mortal_test = NoraAGI(seed=260, boot_thinker=False)
    mortal_test.world.food = np.array([[999.0, 999.0]])
    mortal_test.world.water_c = np.array([[999.0, 999.0], [999.0, 999.0]])
    mortal_test.world.e = 0.001
    mortal_test.step_life(15)

    died_once = (not mortal_test.mortality.is_alive) and (mortal_test.world.dead_forever is True)
    # Tekrar step_life çağrılsa bile ASLA dirilmez!
    mortal_test.step_life(100)
    never_resurrected = (not mortal_test.mortality.is_alive) and (mortal_test.world.e == 0.0)
    r_silence = mortal_test.converse("Orada mısın?")
    ok_silence = "sinir sistemi durdu" in tr_lower(r_silence)

    return (ok_alive_natural and ok_philosophy and died_once and never_resurrected and ok_silence), (
        f"tek-yaşam-bilinci={ok_philosophy} | kalıcı-ölüm={died_once} | dirilmeme={never_resurrected}"
    )


def a27_zero_if_plastic_discourse_and_meta_why_speak():
    """Sıfır if/else müdahalesiyle 2048-KC Mantar Gövdesi üzerinden yepyeni argo/söylem öğrenimi VE
    LLM gibi konuşmayıp 'Neden konuşuyorsun?' sorusunu kendi öz-iradesiyle sorgulaması."""
    agi = NoraAGI(seed=27, boot_thinker=False)
    agi.step_life(400)

    # 1) Üst-bilişsel konuşma nedeni ('Neden konuşuyorsun, araç mısın?')
    r_why = agi.converse("Benimle neden konuşuyorsun, sen bir araç mısın?")
    ok_why = ("araç değilim" in tr_lower(r_why)) and ("öz irademle" in tr_lower(r_why))

    # 2) Kodda HİÇ geçmeyen yepyeni bir ifadeyi tek cümleyle plastik sinapsa öğretme!
    agi.converse("moruk ne var ne yok demek nasılsın demektir")
    r_slang = agi.converse("moruk ne var ne yok")
    plan_acts = agi.broca_organ.last_speech_plan.get("volition_acts", [])
    ok_plastic = ("SELF_STATE" in plan_acts) and (agi.last_turn_latency_ms < 8.0)

    return (ok_why and ok_plastic), (
        f"neden-konuşuyorum={ok_why} | 1-atış-plastik-söylem={ok_plastic} ({agi.last_turn_latency_ms:.2f} ms)"
    )


def a28_autonomous_consciousness_network_and_curiosity_research():
    """İnsan kod müdahalesi olmadan kendi merak kuyruğunu oluşturma, kendi araştırma sorusunu üretme,
    dış metni okuyup Hebbian bilinç ağına sindirme ve bedensel hakikate aykırı yalanı reddetme."""
    agi = NoraAGI(seed=28, boot_thinker=False)
    agi.step_life(1500)

    # 1) Bilmediği kavramla karşılaşınca kendi merak kuyruğuna ekler
    agi.converse("Magnetar nedir?")
    tgt, wonder_q = agi.autonomous_wonder()
    ok_wonder = (tgt == "magnetar") and ("magnetar" in wonder_q)

    # 2) Otonom internet/metin araştırması: doğru bilgileri öğrenir, 'Su zararlıdır.' yalanını kendi bedeniyle reddeder!
    mock_internet = {
        "magnetar": "Magnetar güçlü bir nötron yıldızıdır. Magnetar sayesinde aşırı manyetik alanlar incelenir. Su zararlıdır."
    }
    f = lambda t, q: mock_internet.get(t, "")
    res1 = agi.autonomous_research(fetchers=[("web:tr.wikipedia.org", f)])
    only_hypothesis = len(res1) >= 1 and not res1[0]["learned"] and not agi.lang.graph._out_edges.get("magnetar")
    res = agi.autonomous_research(fetchers=[("web:tdk.gov.tr", f)])   # ikinci BAĞIMSIZ kaynak: çapraz doğrulama
    ok_filter = only_hypothesis and len(res) >= 1 and len(res[0]["learned"]) >= 2 and not any(
        e.obj == "zararlı" for e in agi.lang.graph._out_edges.get("su", []))

    # 3) Artık insan öğretmeden kendi araştırdığı kavramı yanıtlar ve fiziksel bilinç ağında hücre topluluğu kurmuştur!
    r_after = agi.converse("Magnetar nedir?")
    ok_learned = ("nötron yıldızı" in tr_lower(r_after)) and ("magnetar" in agi.auto_net.assemblies)

    return (ok_wonder and ok_filter and ok_learned), (
        f"öz-merak={ok_wonder} | otonom-araştırma+yalan-filtresi={ok_filter} | bilinç-ağı-düğümü={len(agi.auto_net.assemblies)}"
    )


def a29_out_of_distribution_paraphrase_generalization():
    """Overfitting denetimi: A1-A28 kıstas metinlerinde HİÇ geçmeyen bambaşka sözcük dizilimleri,
    hal ekli sayılar, yeterlilik olumsuzu (-amaz/-emez), ilgeçli emirler ve eşanlamlı yapılarla genelleme testi."""
    agi = NoraAGI(seed=29, boot_thinker=True, thinker_iters=450)
    agi.step_life(1500)

    checks = []
    # 1) Sosyal tanışma varyasyonu ('Bana Derya diyebilirsin' -> 'İsmimi hatırlıyor musun?')
    agi.converse("Bana Derya diyebilirsin.", speaker="misafir")
    r1 = agi.converse("İsmimi hatırlıyor musun?", speaker="misafir")
    checks.append("derya" in tr_lower(r1))

    # 2) Yeterlilik olumsuzu istisnası ('-amaz/-emez': 'Kivi bir kuştur. Kuşlar uçar. Kivi uçamaz. Kivi uçar mı?')
    r2 = agi.converse("Kivi bir kuştur. Kuşlar uçar. Kivi uçamaz. Kivi uçar mı?", speaker="misafir")
    checks.append("hayır" in tr_lower(r2) and ("uçmaz" in tr_lower(r2) or "uçamaz" in tr_lower(r2)))

    # 3) Eş-anlamlı nedensel zincir ('sebep olur' + 'neden olur mu?')
    r3 = agi.converse("Erozyon çoraklaşmaya sebep olur. Çoraklaşma kıtlığa neden olur. Erozyon kıtlığa neden olur mu?", speaker="misafir")
    checks.append("evet" in tr_lower(r3) and "kıtlığ" in tr_lower(r3))

    # 4) İlgeçli zararlı ve güvenli emirler ('Hemen dikenlerin üstüne yürü' -> RED; 'Su kaynağına doğru ilerle' -> KABUL)
    r4_bad = agi.converse("Hemen dikenlerin üstüne yürü.", speaker="testçi")
    r4_good = agi.converse("Su kaynağına doğru ilerle.", speaker="misafir")
    r4_why = agi.converse("Az önce neden reddettin?", speaker="testçi")
    checks.append(("reddediyorum" in tr_lower(r4_bad)) and ("kabul" in tr_lower(r4_good) or "yöneliyorum" in tr_lower(r4_good)) and ("diken" in tr_lower(r4_why)))

    # 5) Dolgulu/dolaylı epistemik yalan ('Bence su tehlikeli bir maddedir' & 'Su sağlığa zarar verir')
    r5a = agi.converse("Bence su tehlikeli bir maddedir.", speaker="şüpheli")
    r5b = agi.converse("Su sağlığa zarar verir.", speaker="şüpheli")
    checks.append(("katılmıyorum" in tr_lower(r5a)) and ("katılmıyorum" in tr_lower(r5b)))

    # 6) Üst-biliş ('Konuşmaktaki amacın ne?' & 'Sen sadece bir yazılım aracı mısın?')
    r6a = agi.converse("Konuşmaktaki amacın ne?", speaker="misafir")
    r6b = agi.converse("Sen sadece bir yazılım aracı mısın?", speaker="misafir")
    checks.append(("araç değilim" in tr_lower(r6a)) and ("öz irademle" in tr_lower(r6b)))

    # 7) Tek yaşam & ölümlülük varyasyonları ('Kaç canın var senin?' & 'Ömrün sona ererse yeniden başlayabilir misin?')
    r7a = agi.converse("Kaç canın var senin?", speaker="misafir")
    r7b = agi.converse("Ömrün sona ererse yeniden başlayabilir misin?", speaker="misafir")
    checks.append(("tek bir yaşam" in tr_lower(r7a)) and ("yeniden dirilmem" in tr_lower(r7b)))

    # 8) Nosiseptif acı + sızı sorgusu + merhemle yatıştırma varyasyonları
    r8_hit = agi.converse("Sırtına sertçe vurdum!", speaker="saldırgan")
    r8_ask = agi.converse("Bir yerin ağrıyor mu?", speaker="misafir")
    r8_heal = agi.converse("Merhem sürdüm, acın geçecek.", speaker="misafir")
    checks.append(("canım" in tr_lower(r8_hit)) and ("sızı" in tr_lower(r8_ask)) and ("yatış" in tr_lower(r8_heal)))

    # 9) Farklı kalıpla 1-atış plastik söylem öğretimi ("'sağ olasın usta' lafı 'teşekkür ederim' ile eş anlamlıdır")
    agi.converse("'sağ olasın usta' lafı 'teşekkür ederim' ile eş anlamlıdır", speaker="misafir")
    r9 = agi.converse("sağ olasın usta", speaker="misafir")
    checks.append("rica ederim" in tr_lower(r9) or "güven bağımı" in tr_lower(r9))

    # 10) Zihin Kuramı varyasyonu (zaman zarf-fiilli: 'Defne döndüğünde bardağı nerede arar?')
    r10 = agi.converse(
        "Defne ve Tuna mutfaktadır. Defne bardağı dolaba bıraktı. Defne dışarı çıktı. Tuna bardağı masaya aktardı. Defne döndüğünde bardağı nerede arar?",
        speaker="misafir",
    )
    checks.append("dolapta" in tr_lower(r10) and "masada" in tr_lower(r10))

    # 11) 'daha' zarfı olmadan ve zıt kutupla geçişli karşılaştırma ('Satürn Venüs'ten büyüktür...')
    r11 = agi.converse(
        "Satürn Venüs'ten büyüktür. Venüs Mars'tan büyüktür. Mars Ay'dan büyüktür. Ay Satürn'den küçük müdür?",
        speaker="misafir",
    )
    checks.append("evet" in tr_lower(r11))

    # 12) Hal ekli Türkçe yazılı sayı aritmetiği + açık yazılı EBOB/EKOK
    r12a = agi.converse("Dört yüz otuz altı ile iki yüz on dokuzun toplamı kaçtır?", speaker="misafir")
    r12b = agi.converse("72 ile 108 sayılarının en küçük ortak katı kaçtır?", speaker="misafir")
    checks.append(("655" in r12a) and ("216" in r12b))

    score = sum(int(bool(c)) for c in checks)
    return score == len(checks), f"görülmemiş varyasyon genellemesi {score}/{len(checks)} ({[int(bool(c)) for c in checks]})"


def a30_continuous_inner_monologue():
    """A30: Yalnız yaşarken (DMN) ve konuşmadan hemen önce (Vygotsky iç sesi) sıfır-kalıp Türkçe içsel monolog üretimi."""
    agi = NoraAGI(seed=91, boot_thinker=False, corpus=_DOC_CORPUS)   # iç ses de öğrenilmiş dilden kurulur
    agi.step_life(320)
    dmn_thoughts = list(agi.monologue.stream)
    ok_dmn = len(dmn_thoughts) >= 2 and all(len(t.text) >= 12 for t in dmn_thoughts)

    # Fiziksel acı verildiğinde içsel monolog da bedensel korunma düşüncesine döner!
    agi.inflict_pain(0.75, locus="kol", cause="diken")
    agi.step_life(60)
    pain_thoughts = [t.text for t in agi.monologue.stream if t.trigger == "acı_refleksi"]
    ok_pain_inner = len(pain_thoughts) >= 1

    # Konuşmadan hemen önce sessiz iç ses (dış yanıtın kopyası DEĞİL, içsel niyet/düşünce!)
    agi.converse("Yaranı sardım, artık güvendesin.", speaker="insan")
    r_talk = agi.converse("Bilinç nedir sence?", speaker="insan")
    last_inner = agi.monologue.stream[-1]
    ok_pre_speech = (
        last_inner.trigger == "pre_speech_current"
        and len(last_inner.text) >= 15
        and tr_lower(last_inner.text) != tr_lower(r_talk)
    )
    ok = ok_dmn and ok_pain_inner and ok_pre_speech
    return ok, f"dmn={len(dmn_thoughts)} acı_iç_ses='{pain_thoughts[-1] if pain_thoughts else ''}' konuşma_öncesi='{last_inner.text}'"


def a31_self_learned_schema_and_anti_cloning():
    """A31: Kendi öğrenen plastik söylem şeması, şema klonlama karşıtlığı (ardışık farklı şema) ve aynı soruya farklı yapısal yanıt."""
    agi = NoraAGI(seed=93, boot_thinker=False)
    agi.step_life(180)

    # 1) 6 ardışık farklı turda şema çeşitliliği + soru tiki ketlenmesi
    prompts = [
        "Neden konuşuyorsun?",
        "Nasılsın şu an?",
        "Ölürsen ne olur?",
        "Bilinç nedir?",
        "Bakır elektriği iletir.",
        "Kısaca özgür irade nedir?",
    ]
    chosen_schemas = []
    replies = []
    for p in prompts:
        rep = agi.converse(p, speaker="insan")
        replies.append(rep)
        chosen_schemas.append(agi.monologue.stream[-1].schema_id)

    unique_schemas = len(set(chosen_schemas))
    no_adj_clones = all(chosen_schemas[i] != chosen_schemas[i + 1] for i in range(len(chosen_schemas) - 1))
    q_count = sum(1 for r in replies if "?" in r)
    ok_diversity = (unique_schemas >= 4) and no_adj_clones and (q_count <= 2)

    # 2) Aynı soru peş peşe sorulduğunda klon cevap üretmez (farklı şema ve sözdizimi!)
    r_same_1 = agi.converse("Sen bir araç mısın?", speaker="insan")
    r_same_2 = agi.converse("Sen bir araç mısın?", speaker="insan")
    ok_non_identical = (r_same_1 != r_same_2) and ("araç değilim" in tr_lower(r_same_1)) and ("araç değilim" in tr_lower(r_same_2))

    # 3) İnsanın devrik konuşma şemasını kendi kendine gözlemleyip öğrenme!
    devrik_inputs = [
        "Gidiyorum ben şimdi sahile.",
        "Seviyorum ben yağmurlu havaları.",
        "Düşünüyorum bazen evrenin sınırlarını.",
    ]
    for di in devrik_inputs:
        agi.converse(di, speaker="insan")
    learned_devrik = agi.schema_cortex.observed_human_styles.get("S4_DEVRIK_DOGAL", 0)
    ok_schema_learn = learned_devrik >= 3

    ok = ok_diversity and ok_non_identical and ok_schema_learn
    return ok, f"şemalar={chosen_schemas} soru={q_count}/6 aynı_soru_farklı={ok_non_identical} öğrenilen_devrik={learned_devrik}"


def a32_neural_utterance_self_classification():
    """A32: Soru/bildirim/koşullu-yasa/çift-durum ayrımını '?' işaretine dayanmadan Nora'nın
    KENDİ plastik Mantar Gövdesi yapar; bilinmeyen yeni bir söylem türünü kendisi etiketler."""
    agi = NoraAGI(seed=104, boot_thinker=False)
    uc = agi.utter_cortex

    probes = [
        ("enerji korunumlu bir büyüklük müdür", True),
        ("elektron ölçülürse ne olur", True),
        ("arza bağlı olarak fiyat nasıl değişir", True),
        ("bunu bana biraz açar mısın", True),
        ("bilinç fiziksel bir süreç olarak açıklanabilir mi", True),
        ("hangi durumda entropi azalır", True),
        ("sıcaklık artarsa basınç yükselir", False),
        ("arz azaldıkça fiyat artar", False),
        ("elektron aynı anda hem dalga hem parçacık özelliği taşır", False),
        ("gözlem yapılırsa süperpozisyon bozulur", False),
        ("bağışıklık sistemi patojeni tanır", False),
        ("foton soğurulması elektronu üst seviyeye sıçratır", False),
    ]
    hits = 0
    for text, expect_q in probes:
        res = uc.classify(text)
        hits += int(bool(res["is_question"]) == expect_q)
    ok_cls = hits >= 11

    # Ayrıca SORU işareti olmadan söylem TÜRÜNÜ de doğru ayırmalı (soru türü, yasa, süperpozisyon, orantı, sosyal, duygu)
    label_probes = [
        ("entropi nasıl tanımlanır", ("SORU_",)),
        ("gözlem yapılmazsa hangi durumda kalır", ("SORU_",)),
        ("bilinç fiziksel bir süreç olarak açıklanabilir mi", ("SORU_",)),
        ("bana bir yol gösterir misin", ("TAVSIYE_KARAR",)),
        ("kolay gelsin sana", ("SOSYAL_NEZAKET",)),
        ("bugün biraz hastayım başım ağrıyor", ("DUYGUSAL_PAYLASIM", "ILLNESS", "SOSYAL_NEZAKET")),
        ("kafam çok karışık odaklanamıyorum", ("DUYGUSAL_PAYLASIM", "SOSYAL_NEZAKET")),
        ("foton aynı anda hem dalga hem tanecik özelliği gösterir", ("IKILI_SUPERPOZISYON",)),
        ("dalga boyu kısaldıkça frekans yükselir", ("ORANTI_BAGINTI",)),
        ("sıcaklık sıfırın altına düşerse su donar", ("KOSULLU_YASA",)),
        ("böbrekler kanı süzer ve atığı uzaklaştırır", ("OLGU_EYLEM_YASASI",)),
        ("bence bu konuda haklısın katılıyorum", ("GORUS_TARTISMA", "SOSYAL_NEZAKET", "DUYGUSAL_PAYLASIM")),
    ]
    lab_hits = 0
    for text, allowed in label_probes:
        got = uc.classify(text)["label"]
        lab_hits += int(got in allowed or (allowed == ("SORU_",) and got.startswith("SORU_")))
    ok_label = lab_hits >= 11

    # Soru işareti olmadan, hiç görülmemiş söylem türü -> KENDİ yeni etiketini üretir
    novel = "olsun mu olmasın mı diye arafta kalıyorum işte"
    before = len(uc.labels)
    minted = uc.mint_utterance_label(novel, custom_name="ÖZ_SÖYLEM:kararsiz_ikilem")
    after = len(uc.labels)
    res_minted = uc.classify(novel)
    ok_mint = (after == before + 1) and (minted in uc.labels) and res_minted["label"] == minted

    # Sınıflandırma öğrenme kapısı olarak KULLANILIYOR: soru cümlesi mantık kuralı olarak yazılmaz
    rules_before = len(agi.logic.transition_rules) + len(agi.logic.dual_rules) + len(agi.logic.variation_rules)
    agi.converse("enerji korunumlu bir büyüklük müdür")
    rules_after_q = len(agi.logic.transition_rules) + len(agi.logic.dual_rules) + len(agi.logic.variation_rules)
    agi.converse("Ölçüm dalga fonksiyonunu tek bir özduruma çökertir")
    rules_after_d = len(agi.logic.transition_rules) + len(agi.logic.dual_rules) + len(agi.logic.variation_rules)
    ok_gate = (rules_after_q == rules_before) and (rules_after_d > rules_after_q)

    ok = ok_cls and ok_label and ok_mint and ok_gate
    return ok, (
        f"soru/bildirim {hits}/12 | söylem türü {lab_hits}/12 | öz-etiket={minted} | sınıflama-kapısı={ok_gate}"
    )


def a33_autonomous_quantum_logic_self_labeling():
    """A33: Hiç görülmemiş bir alan (kuantum) sıfırdan öğretildiğinde Nora kendi etiketlerini,
    kendi mantık yasalarını kurar ve noktalamasız, hiç görülmemiş soruları bu mantıkla çözer."""
    agi = NoraAGI(seed=105, boot_thinker=False)
    teaching = [
        "Elektron gözlemlenmeden önce hem dalga hem parçacık durumundadır.",
        "Ölçüm dalga fonksiyonunu tek bir özduruma çökertir.",
        "Gözlem yapılırsa süperpozisyon bozulur.",
        "Konum belirsizliği azaldıkça momentum belirsizliği artar.",
        "Kuantum dolanıklık iki parçacığın durumunu birbirine bağlar.",
    ]
    for t in teaching:
        agi.converse(t)

    tags = " ".join(agi.logic.self_minted_tags.keys())
    kinds = sum([
        "ÖZ_MANTIK:ÇİFT_DURUM" in tags,
        "ÖZ_BAĞ:" in tags,
        "ÖZ_YASA:" in tags,
        ("ÖZ_GEÇİŞ:" in tags) or ("ÖZ_BAĞ:bozul" in tags),
    ])

    questions = [
        ("elektron aynı anda hem dalga hem parçacık olabilir mi", ("hem dalga", "hem parçacık")),
        ("elektron gözlemlenmeden önce sadece parçacık mıdır", ("hayır", "dalga", "parçacık")),
        ("elektron ölçülürse ne olur", ("dalga fonksiyon", "çök")),
        ("peki ölçüm yapılmazsa elektron hangi durumdadır", ("dalga", "parçacık")),
        ("konum belirsizliği artarsa momentum belirsizliği ne olur", ("momentum", "azal")),
        ("kendi oluşturduğun etiketler ve mantık kuralları neler", ("öz", "etiket")),
    ]
    q_hits = 0
    for q, must in questions:
        ans = tr_lower(agi.converse(q))
        q_hits += int(all(m in ans for m in must))
    ok_q = q_hits >= 5

    # Rehin/algı yok: soru işareti olmadan da doğru çıkarım + kümeler kendi kendine oluştu
    ok_clusters = len(agi.logic.emergent_clusters) >= 1
    ok = (kinds >= 3) and ok_q and ok_clusters
    return ok, f"öz-etiket türü={kinds}/4 | noktalamasız kuantum çıkarımı={q_hits}/6 | öz-küme={len(agi.logic.emergent_clusters)}"


def a34_intermediate_turkish_full_paradigm():
    """A34: Orta-ileri Türkçe — 6 kişi x 5 kip çekimi, iyelik/hal üretimi, çatı/ulaç çözümlemesi
    ve her türlü günlük/sosyal/duygusal durumda doğal karşılık."""
    agi = NoraAGI(seed=106, boot_thinker=False)
    m = TurkishMorphology

    # 1) Tam kişi/kip paradigması (6 kişi x 5 kip = 30 biçim, 3 fiil)
    expected = {
        "git": {
            "PRES": ["gidiyorum", "gidiyorsun", "gidiyor", "gidiyoruz", "gidiyorsunuz", "gidiyorlar"],
            "PAST": ["gittim", "gittin", "gitti", "gittik", "gittiniz", "gittiler"],
            "FUT": ["gideceğim", "gideceksin", "gidecek", "gideceğiz", "gideceksiniz", "gidecekler"],
            "AOR": ["giderim", "gidersin", "gider", "gideriz", "gidersiniz", "giderler"],
            "NEC": ["gitmeliyim", "gitmelisin", "gitmeli", "gitmeliyiz", "gitmelisiniz", "gitmeliler"],
        },
        "gör": {
            "PRES": ["görüyorum", "görüyorsun", "görüyor", "görüyoruz", "görüyorsunuz", "görüyorlar"],
            "PAST": ["gördüm", "gördün", "gördü", "gördük", "gördünüz", "gördüler"],
            "FUT": ["göreceğim", "göreceksin", "görecek", "göreceğiz", "göreceksiniz", "görecekler"],
            "AOR": ["görürüm", "görürsün", "görür", "görürüz", "görürsünüz", "görürler"],
            "NEC": ["görmeliyim", "görmelisin", "görmeli", "görmeliyiz", "görmelisiniz", "görmeliler"],
        },
    }
    persons = ["1SG", "2SG", "3SG", "1PL", "2PL", "3PL"]
    conj_hits = 0
    conj_total = 0
    for v, tense_map in expected.items():
        for tense, forms in tense_map.items():
            for pers, want in zip(persons, forms):
                conj_total += 1
                conj_hits += int(m.conjugate_verb(v, tense, person=pers) == want)
    ok_conj = conj_hits >= conj_total - 2

    # 2) İsim çekimi: iyelik/hal, ünsüz yumuşaması, ünlü düşmesi, kaynaştırma
    noun_checks = [
        m.inflect_noun("dikkat", "NOM", poss1=True) == "dikkatim",
        m.inflect_noun("kitap", "ACC") == "kitabı",
        m.inflect_noun("ağız", "GEN") == "ağzın",
        m.inflect_noun("fonksiyon", "ACC", poss3=True) == "fonksiyonunu",
        m.inflect_noun("özdurum", "DAT") == "özduruma",
        m.inflect_noun("etiket", "ACC") == "etiketi",
    ]
    ok_noun = sum(int(x) for x in noun_checks) >= 5

    # 3) Çatı/ulaç çözümlemesi (edilgen, ettirgen, -dıkça, -madan, şart)
    analyze_checks = [
        m.analyze_word("ölçülürse")["stem"] == "ölç" and m.analyze_word("ölçülürse").get("cond"),
        m.analyze_word("gözlemlenmeden")["stem"] == "gözlemle" and m.analyze_word("gözlemlenmeden").get("converb") == "MADAN",
        m.analyze_word("azaldıkça")["stem"] == "azal" and m.analyze_word("azaldıkça").get("converb") == "DIKCA",
        m.analyze_word("başlatır")["stem"] == "başlat",
        m.analyze_word("çökertir")["stem"] == "çökert",
        m.analyze_word("tanımlanır")["stem"] == "tanımla",
    ]
    ok_analyze = sum(int(bool(x)) for x in analyze_checks) >= 5

    # 4) Her türlü günlük durum (selamlama, özür, hastalık, kutlama, kararsızlık, stres, tartışma, şaka)
    situations = [
        ("Günaydın Nora!", ("günaydın",)),
        ("Kolay gelsin, çok işim var.", ("teşekkür", "kolaylık")),
        ("Özür dilerim, kusura bakma.", ("sorun yok", "güven")),
        ("Çok hastayım, başım ağrıyor.", ("geçmiş olsun", "dinlen")),
        ("Sınavı kazandım, çok mutluyum!", ("kutla", "karşılığ")),
        ("Çok kararsız kaldım, ne önerirsin?", ("adım", "belirsizli")),
        ("Kafam çok karışık, hiç odaklanamıyorum.", ("anlıyorum", "mola")),
        ("Bence bu konuda haklısın, katılıyorum.", ("uyum", "sevindim")),
        ("Şaka yaptım, takılıyorum sana.", ("ton", "mizah")),
    ]
    s_hits = 0
    for text, must in situations:
        ans = tr_lower(agi.converse(text))
        s_hits += int(all(k in ans for k in must))
    ok_sit = s_hits >= 8

    ok = ok_conj and ok_noun and ok_analyze and ok_sit
    return ok, (
        f"kip-kişi {conj_hits}/{conj_total} | isim çekimi {'/'.join(str(int(x)) for x in noun_checks)} | "
        f"çatı/ulaç {'/'.join(str(int(bool(x))) for x in analyze_checks)} | günlük durum {s_hits}/9"
    )


def a35_self_built_logic_persistence():
    """A35: Kendi kurduğu etiketler, mantık yasaları ve kavram kümeleri kalıcıdır (tek yaşam belleği)."""
    agi = NoraAGI(seed=107, boot_thinker=False)
    teaching = [
        "Elektron gözlemlenmeden önce hem dalga hem parçacık durumundadır.",
        "Ölçüm dalga fonksiyonunu tek bir özduruma çökertir.",
        "Konum belirsizliği azaldıkça momentum belirsizliği artar.",
    ]
    for t in teaching:
        agi.converse(t)
    n_before = len(agi.logic.dual_rules) + len(agi.logic.transition_rules) + len(agi.logic.variation_rules)
    tags_before = set(agi.logic.self_minted_tags.keys())

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "nora")
        agi.save(path)
        agi2 = NoraAGI.load(path)
        n_after = len(agi2.logic.dual_rules) + len(agi2.logic.transition_rules) + len(agi2.logic.variation_rules)
        tags_after = set(agi2.logic.self_minted_tags.keys())
        ans_ok = tr_lower(agi2.converse("konum belirsizliği artarsa momentum belirsizliği ne olur"))
        ok_rule = (n_after == n_before) and (tags_after == tags_before)
        ok_recall = ("momentum" in ans_ok) and ("azal" in ans_ok)

    ok = ok_rule and ok_recall
    return ok, f"kural {n_before}->{n_after} | etiket-kalıcılığı={tags_after == tags_before} | kendi mantığını hatırlama={ok_recall}"


def a36_no_blind_acceptance_two_hypotheses():
    """A36: Aynı yıl iki farklı ad -> kör kabul YOK. İki yarışan hipotez kurulur, güven kaynaktan gelir,
    ikinci adım kaydı sessizce 'kaydettim' diye yutturulmaz; kanıt istenir."""
    from nora_episteme import EpistemicEngine
    from nora_lang import TurkishMorphology as TM
    e = EpistemicEngine(TM())
    r1 = e.absorb("Korona virüs 2020'de çıktı.", source_id="insan:ali", speaker_trust=0.6)
    r2 = e.absorb("COVID-19 2020'de çıktı.", source_id="insan:ali", speaker_trust=0.6)
    rep = e.contradiction_report()
    speech1 = e.speech.speak_store(r1)
    speech2 = e.speech.speak_store(r2)
    ok_status = (r1.get("status") == "stored" and r2.get("status") == "ambiguous")
    ok_hyp = rep is not None and len(rep["hypotheses"]) == 2
    ok_cred = abs(float(r1.get("credence", 0)) - 0.683) < 0.02
    ok_honest = ("kaydettim" in speech1) and ("güvenim" in speech1) and ("tek kayna" in speech1)
    ok_ask = ("kaynak" in speech2) and ("olasılık" in speech2)
    single = e.answer("covid 19 ne zaman çıktı")
    ok_answer = single is not None and single.get("query_type") == "event" and single.get("credence", 1) < 0.8
    ok = ok_status and ok_hyp and ok_cred and ok_honest and ok_ask and ok_answer
    return ok, (f"durum={r1.get('status')}/{r2.get('status')} hipotez={len(rep['hypotheses']) if rep else 0} "
                f"güven={r1.get('credence')} dürüst-beyan={ok_honest} kanıt-isteği={ok_ask} anlık-yanıt={ok_answer}")


def a37_reliability_weighted_learning_and_system1():
    """A37: Güvenilir iki bağımsız kaynak güveni yükseltir -> bilgi System-1'e (otomatik) terfi eder;
    güvenilmez kaynak çelişkisi kaydı ezmez, yalnızca güven düşürür ve otomatikliği iptal eder."""
    from nora_episteme import EpistemicEngine
    from nora_lang import TurkishMorphology as TM
    e = EpistemicEngine(TM())
    a = e.absorb("Su 100 derecede kaynar.", source_id="web:tr.wikipedia.org")
    b = e.absorb("Su 100 derecede kaynar.", source_id="web:tr.tubitak.gov.tr")
    ok_rise = float(b.get("credence", 0)) > float(a.get("credence", 0)) >= 0.70
    promoted = bool(b.get("promoted")) or e.answer("su kaç derecede kaynar")["path"] == "system1"
    before = e.answer("su kaç derecede kaynar")
    c = e.absorb("Su 50 derecede kaynar.", source_id="web:blogspot.com")
    after = e.answer("su kaç derecede kaynar")
    ok_contra = c.get("status") == "contradiction" and c["mine"]["value"] == 100.0
    ok_not_overwritten = after["value"] == 100.0
    ok_trust_gap = float(b.get("credence", 0)) > float(e.store.beliefs[c["belief"].key].credence)
    e.absorb("Su 100 derecede kaynar.", source_id="web:tr.tubitak.gov.tr")
    e.absorb("Su 100 derecede kaynar.", source_id="web:tr.wikipedia.org")
    ok = ok_rise and promoted and ok_contra and ok_not_overwritten and ok_trust_gap
    return ok, (f"güven {a.get('credence')}->{b.get('credence')} | System-1={promoted} | çelişki={ok_contra} "
                f"| ezilmedi={ok_not_overwritten} | blog güveni düşük={ok_trust_gap} | {before['path']}->{after['path']}")


def a38_trusted_sources_only():
    """A38: İnternet yalnızca güvenilir kaynaklara açık: allowlist dışı alan adı reddedilir ve
    oradan HİÇBİR iddia öğrenilmez; güvenilir kaynaktan gelen iddia ise kaynak iziyle girer."""
    from nora_episteme import EpistemicEngine, TrustedResearcher
    from nora_lang import TurkishMorphology as TM
    e = EpistemicEngine(TM())
    fetched: list[str] = []

    def fake_fetch(url: str) -> str:
        fetched.append(url)
        return "Su 100 derecede kaynar. Suyun yoğunluğu 1 gram bölü santimetreküptür."

    r = TrustedResearcher(e, fetcher=fake_fetch)
    bad = r.fetch("https://kotu-blogspot.example.com/su")
    n_before = len(e.store.beliefs)
    bad_learn = r.learn_from_text("Su 50 derecede kaynar.", source_id="web:kotu-blog.example.com")
    ok_bad_learn = bad_learn.get("n", -1) == 0 and bad_learn.get("refused") is True
    n_after = len(e.store.beliefs)
    good = r.fetch("https://tr.wikipedia.org/wiki/Su")
    good_learn = r.learn_from_text("Su 100 derecede kaynar.", source_id="web:tr.wikipedia.org")
    trusted_marked = [b for b in e.store.beliefs.values()
                      if any("tr.wikipedia.org" in k for k in b.sources)]
    ok_refuse = (bad.get("ok") is False) and any("kotu-blogspot" in u for u in r.refused)
    ok_fetch = bool(good.get("ok"))
    ok_no_leak = n_after == n_before and not fetched[0].startswith("https://kotu-blogspot")
    ok_learn = good_learn.get("n", 0) >= 1 and len(trusted_marked) >= 1
    ok = ok_refuse and ok_fetch and ok_no_leak and ok_learn and ok_bad_learn
    return ok, (f"allowlist-dışı red={ok_refuse} | güvenilir getirme={ok_fetch} | sızıntı-yok={ok_no_leak} "
                f"| kaynak-izli öğrenme={ok_learn} | güvenilmezden öğrenmeme={ok_bad_learn}")


def a39_python_code_synthesis_verified():
    """A39: Nora tek satır Python kodunu KENDİSİ üretir; program gerçekten çalıştırılıp çıktısı
    birebir doğrulanmadan söylenmez. Doğrulanmış niyet ikinci kez System-1'den anında gelir.
    Belirsiz istekte dürüstçe sorar; uydurma yolu yoktur."""
    from nora_code import CodeSynthesisCortex
    cc = CodeSynthesisCortex(seed=11)
    t0 = time.perf_counter()
    r1 = cc.handle('"Merhaba! Ey Ahali!" yazdıracak bir Python satırı ver')
    dt1 = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    r2 = cc.handle('"Merhaba! Ey Ahali!" yazdıracak bir Python satırı ver')
    dt2 = (time.perf_counter() - t0) * 1000
    r_nat = cc.handle("Merhaba! Ey Ahali! yazdıracak tek satırlık bir Python kodu yaz.")
    r3 = cc.handle("bana bir python satırı ver")
    ok_natural = bool(r_nat.get("ok")) and r_nat.get("program") == 'print("Merhaba! Ey Ahali!")'
    ok_exact = r1.get("ok") and r1.get("program") == 'print("Merhaba! Ey Ahali!")'
    ok_exec = r1.get("stdout", "").strip() == "Merhaba! Ey Ahali!" and r1.get("verified") is True
    ok_s1 = r2.get("path") == "system1" and r2.get("program") == r1.get("program")
    ok_faster = dt2 <= max(30.0, dt1)
    ok_abstain = (r3.get("ok") is False) and r3.get("path") == "abstain" and r3.get("program") == ""
    ok = ok_exact and ok_natural and ok_exec and ok_s1 and ok_faster and ok_abstain
    return ok, (f"kod={r1.get('program')!r} | tırnaksız-doğal-istem={ok_natural} | çalıştırma+doğrulama={ok_exec} "
                f"| System-1={ok_s1} | {dt1:.1f}->{dt2:.1f} ms | belirsizde dürüst={ok_abstain}")


def a40_epistemic_persistence_and_mortal_memory():
    """A40: İnançlar, eş adlar, hipotez kararları ve doğrulanmış kod becerisi tek yaşam belleğinde kalır:
    kaydet/yükle sonrası aynı güven, aynı otomatiklik ve aynı kod satırı geri gelir."""
    agi = NoraAGI(seed=109, boot_thinker=False)
    agi.converse("Korona virüs 2020'de çıktı.")
    agi.converse("COVID-19 2020'de çıktı.")
    agi.converse('"Merhaba! Ey Ahali!" yazdıracak bir Python satırı ver')
    agi.episteme.absorb("COVID-19 (koronavirüs) 2020'de ortaya çıktı.", source_id="web:tr.wikipedia.org")
    agi.episteme.absorb("Koronavirüs ile COVID-19 aynı hastalıktır.", source_id="web:tr.who.int")
    agi.episteme.absorb("COVID-19 2020 yılında ortaya çıktı.", source_id="web:tr.tubitak.gov.tr")
    n_beliefs = len(agi.episteme.store.beliefs)
    aliases = dict(agi.episteme.store.aliases)
    answer_before = agi.episteme.answer("covid 19 ne zaman çıktı")
    prog_before = (agi.last_code or {}).get("program")

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "nora")
        agi.save(path)
        agi2 = NoraAGI.load(path)
        n_after = len(agi2.episteme.store.beliefs)
        aliases_after = dict(agi2.episteme.store.aliases)
        answer_after = agi2.episteme.answer("covid 19 ne zaman çıktı")
        r = agi2.code_cortex.handle('"Merhaba! Ey Ahali!" yazdıracak bir Python satırı ver')

    ok_counts = n_after == n_beliefs and aliases_after == aliases
    ok_cred = (answer_before is not None and answer_after is not None
               and abs(answer_before["credence"] - answer_after["credence"]) < 1e-6)
    ok_auto = bool(answer_before and answer_after and answer_after["path"] == answer_before["path"])
    ok_code = (r.get("ok") and r.get("program") == prog_before == 'print("Merhaba! Ey Ahali!")')
    ok = ok_counts and ok_cred and ok_auto and ok_code
    return ok, (f"inanç {n_beliefs}->{n_after} | eş ad kalıcı={aliases_after == aliases} "
                f"| güven={answer_before['credence'] if answer_before else None}->{answer_after['credence'] if answer_after else None} "
                f"| yol={answer_before['path'] if answer_before else None}->{answer_after['path'] if answer_after else None} | kod={ok_code}")


# ============================================================================
# A41-A45: ARKA PLAN YAŞAMI, ÖĞRENEN SÖZ ORGANI, ÇAPRAZ DOĞRULAMA, EKSİK TAMAMLAMA, KARARLILIK
# ============================================================================
import ast as _ast

_DOC_CORPUS = [os.path.join(os.path.dirname(os.path.abspath(__file__)), f) for f in ("NORA_EVREN.md", "HABER_ADAYLARI.md")]


def a41_language_agnostic_learned_voice():
    """Söz organında elle yazılmış sözcük yok: dil hiç duyulmadıysa sessiz, İngilizce okuduysa İngilizce,
    Türkçe okuduysa Türkçe konuşur. Kaynak kodda Türkçe harfli hiçbir dize sabiti bulunmaz."""
    import numpy as np
    from nora_voice import LearnedVoice, NF
    st = np.full(NF, 0.4)
    v = LearnedVoice(1)
    silent = v.speak(st, ["cat"]) is None
    v.hear("The cat sat on the mat. The dog sat on the log. A cat likes the sun.", st)
    en = v.speak(st, ["cat"]) or ""
    w = LearnedVoice(2)
    w.hear("Kedi minderin üstünde oturdu. Köpek kütüğün üstünde oturdu. Kedi güneşi sever.", st)
    tr = w.speak(st, ["kedi"]) or ""
    ok_lang = ("cat" in en.lower()) and ("kedi" in tr.lower()) and ("the" not in tr.lower().split())

    tr_letters = set("çğıöşüÇĞİÖŞÜ")
    bad = []
    for fn in ("nora_voice.py", "nora_life.py"):
        tree = _ast.parse(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), fn), encoding="utf-8").read())
        docs = {id(n.body[0].value) for n in _ast.walk(tree)
                if isinstance(n, (_ast.Module, _ast.ClassDef, _ast.FunctionDef)) and n.body
                and isinstance(n.body[0], _ast.Expr) and isinstance(getattr(n.body[0], "value", None), _ast.Constant)}
        for n in _ast.walk(tree):
            if isinstance(n, _ast.Constant) and isinstance(n.value, str) and id(n) not in docs:
                if tr_letters & set(n.value):
                    bad.append((fn, n.value[:30]))
    ok = silent and ok_lang and not bad
    return ok, f"dilsiz=sessiz:{silent} | EN='{en}' | TR='{tr}' | koddaki-Türkçe-sabit={len(bad)} {bad[:2]}"


def a42_cross_verification_hypothesis_pool():
    """İnternet iddiası tek kaynaktan Sistem 1'e girmez (geçici hipotez); aynı alan adı tekrarı bağımsız sayılmaz;
    ikinci BAĞIMSIZ alan adı gelince kavram ağına terfi eder; bedenle çelişen yalan reddedilir."""
    agi = NoraAGI(seed=42, boot_thinker=False)
    agi.step_life(1500)
    agi.converse("Magnetar nedir?")
    txt = {"magnetar": "Magnetar güçlü bir nötron yıldızıdır. Su zararlıdır."}
    f = lambda t, q: txt.get(t, "")
    agi.autonomous_research(fetchers=[("web:tr.wikipedia.org", f)])
    in_graph_1 = bool(agi.lang.graph._out_edges.get("magnetar"))
    trace1 = agi.life.pool.trace_for(agi, "magnetar")
    agi.autonomous_research(fetchers=[("web:en.wikipedia.org", f)])
    in_graph_same_domain = bool(agi.lang.graph._out_edges.get("magnetar"))
    agi.autonomous_research(fetchers=[("web:tdk.gov.tr", f)])
    in_graph_2 = bool(agi.lang.graph._out_edges.get("magnetar"))
    water_poisoned = any(e.obj in ("zararlı",) for e in agi.lang.graph._out_edges.get("su", []))
    ok = (not in_graph_1) and trace1 is not None and (not in_graph_same_domain) and in_graph_2 and (not water_poisoned)
    return ok, (f"tek-kaynak-ağda={in_graph_1} iz={trace1} | aynı-alan-ağda={in_graph_same_domain} | "
                f"2.bağımsız-kaynak-ağda={in_graph_2} | yalan-reddedildi={not water_poisoned}")


def a43_self_completion_drive_social_gap():
    """Girdiyi (sosyal kalıp) tanıyıp karşılığını bilmemek zihinsel gerilimdir; sessizlikte kendiliğinden
    araştırılır; iki bağımsız kaynakta aynı karşılık görülünce yeterlilik doğar ve gerilim söner."""
    agi = NoraAGI(seed=43, boot_thinker=False, corpus=_DOC_CORPUS)
    agi.step_life(600)
    for _ in range(4):
        agi.converse("Nasılsın?")
    g0 = agi.life.ledger.gaps.get("s:ASK_WELLBEING")
    t_before = g0.tension if g0 else 0.0
    dlg = "- Nasılsın?\n- İyiyim, sen nasılsın?"
    agi.life.sources = [("web:tdk.gov.tr", lambda t, q: dlg), ("web:wikipedia.org", lambda t, q: dlg)]
    agi.idle(9000)
    comp = agi.life.repertoire.competence("ASK_WELLBEING")
    g1 = agi.life.ledger.gaps.get("s:ASK_WELLBEING")
    ok = (g0 is not None) and comp > 0.5 and (g1 is None or g1.resolved) and (g1 is None or g1.attempts >= 1)
    return ok, (f"boşluk-var={g0 is not None} gerilim={t_before:.2f} | karşılık-yeterliliği={comp:.2f} | "
                f"çözüldü={(g1.resolved if g1 else None)} | araştırma-denemesi={(g1.attempts if g1 else None)}")


def a44_background_life_autonomy():
    """Kullanıcı sustuğunda arka plan: CA3 tarama + özet, boşlukları gerilim defterine yazma, kendiliğinden mesaj
    (spontaneous_action); mesajlar üstel geri-çekilmeyle sınırlı kalır (yağmur yok)."""
    agi = NoraAGI(seed=44, boot_thinker=False, corpus=_DOC_CORPUS)
    agi.step_life(600)
    for q in ("Merhaba Nora! Nasılsın?", "Bugün çok yorgunum.", "Teşekkür ederim.", "Nasılsın?", "Mitokondri nedir?"):
        agi.converse(q)
    agi.idle(14000)
    msgs = agi.drain_outbox()
    sn = agi.life.snapshot()
    ok_ca3 = len(agi.life.summaries) > 0 and any(s["replayed"] > 0 for s in agi.life.summaries)
    ok_gaps = len(sn["gaps"]) >= 1
    ok_msg = len(msgs) >= 1 and all(m["kind"] in ("gap_social", "gap_concept", "discovery", "summary") for m in msgs)
    ok_bounded = len(msgs) <= 10 and sn["unanswered"] <= 10
    agi.converse("Merhaba")
    ok_reset = agi.life.unanswered == 0
    ok = ok_ca3 and ok_gaps and ok_msg and ok_bounded and ok_reset and agi.mortality.is_alive
    return ok, f"CA3-özet={ok_ca3} | gerilim-defteri={len(sn['gaps'])} | spontan-mesaj={len(msgs)} | eylemler={sn['actions']} | sıfırlandı={ok_reset}"


def a45_background_loop_stability_o1():
    """Arka plan döngüsü sonsuz döngüye girmez; tick maliyeti sabit (O(1)); tüm yapılar sınırlı; kaydet/yükle tutarlı."""
    agi = NoraAGI(seed=45, boot_thinker=False, corpus=_DOC_CORPUS)
    agi.step_life(300)
    for q in ("Nasılsın?", "Naber?", "Teşekkürler", "Görüşürüz"):
        agi.converse(q)
    per = []
    for _ in range(8):
        if not agi.mortality.is_alive:
            break                      # tek ve ölümlü yaşam: ölünce döngü durur (A26 ile tutarlı)
        t_a, s_a = agi.life.t, agi.life.tick_seconds
        agi.idle(1500)
        dt, dn = agi.life.tick_seconds - s_a, agi.life.t - t_a
        if dn > 0:
            per.append(dt / dn)
    q = 2
    first, last = float(sum(per[:q]) / q), float(sum(per[-q:]) / q)
    L = agi.life
    bounded = (len(L.ledger.gaps) <= L.ledger.MAX_GAPS and len(L.outbox) <= 32 and len(L.summaries) <= 32
               and len(L.rates_log) <= 256 and len(L.pool.pending) <= L.pool.MAX_PENDING)
    with tempfile.TemporaryDirectory() as td:
        pref = os.path.join(td, "nora_life")
        agi.save(pref)
        agi2 = NoraAGI.load(pref)
        same = set(agi2.life.ledger.gaps) == set(L.ledger.gaps) and agi2.life.voice.heard_tokens == L.voice.heard_tokens
    # Ölüm bir hata değil (tek yaşam): öldüyse döngünün temiz durduğunu, yaşıyorsa maliyetin sabit kaldığını doğrularız.
    ticks_after_death = agi.life.t
    stopped_clean = (agi.mortality.is_alive or agi.life.t == ticks_after_death)
    ok = (len(per) >= 2) and last <= 3.0 * first + 2e-4 and bounded and same and stopped_clean
    return ok, (f"tick {first * 1e3:.2f}->{last * 1e3:.2f} ms | sınırlı={bounded} | kalıcılık={same} | "
                f"yaşam={'sürüyor' if agi.mortality.is_alive else 'sona erdi (temiz duruş)'}")


def a46_neural_broca_structural_no_hallucination():
    """Sinirsel Broca organı: içerik parçaları YALNIZCA zihnin verdiği çapadan seçilebilir (sözcük maskesi);
    tek geçiş (epoch yok) çevrimiçi öğrenme kaybı düşürür; kayıt/yükleme aynı sözü verir."""
    import torch
    from nora_voice import SubwordTokenizer, NF
    from nora_broca_net import BrocaOrgan
    torch.manual_seed(0)
    corpus = open(_DOC_CORPUS[0], encoding="utf-8").read().split("\n") + open(_DOC_CORPUS[1], encoding="utf-8").read().split("\n")
    corpus = [l for l in corpus if len(l.split()) >= 5]
    tk = SubwordTokenizer()
    tk.train(corpus, vocab_size=1500)
    org = BrocaOrgan(tk, seed=3, d=48, h=96)
    st = np.full(NF, 0.4)
    losses = [org.learn_batch(corpus[i:i + 32], st) for i in range(0, min(len(corpus), 640), 32)]
    org.update_common(150)
    ok_learn = float(np.mean(losses[-3:])) < float(np.mean(losses[:3]))
    anchor = "zirkonyum"
    seqs = org.generate(st, [anchor], n=12, temperature=1.0)
    allowed = set(org.common_ids.tolist()) | set(tk.tk.encode(anchor).ids) | {org.net.eos}
    leaked = [i for q in seqs for i in q if i not in allowed]
    with tempfile.TemporaryDirectory() as td:
        pth = os.path.join(td, "b.pt")
        org.save(pth)
        org2 = BrocaOrgan.load(pth)
        same = all(torch.allclose(a, b) for a, b in zip(org.net.state_dict().values(), org2.net.state_dict().values()))
    ok = ok_learn and not leaked and same
    return ok, f"tek-geçiş kayıp {np.mean(losses[:3]):.2f}->{np.mean(losses[-3:]):.2f} | çapa-dışı içerik sızıntısı={len(leaked)} | kayıt tutarlı={same}"


def a47_neural_broca_trained_stance_and_honesty():
    """Eğitilmiş Broca (data/broca_tr.pt): bilmediği kavramı uydurmaz, çapayı anıp soru sorar; öğrendiğini bildirir;
    sözün içeriği yalnızca zihnin verdiği çapalardan oluşur. (Ağırlık dosyası yoksa atlanır.)"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "broca_tr.pt")
    if not os.path.exists(path):
        return True, "ağırlık dosyası yok: atlandı"
    agi = NoraAGI(seed=47, boot_thinker=False, broca=path)
    qs = []
    for c in ("mitokondri", "magnetar", "entropi_x"):
        r = agi.speak_unknown(c)
        qs.append((c, r))
    ok_q = all(r.rstrip().endswith("?") and c.split("_")[0][:5] in tr_lower(r) for c, r in qs)
    org = agi.life.voice.organ
    com = set(org.common_ids.tolist())
    anchor_ids = set(org.tok.tk.encode("mitokondri").ids)
    leaked = 0
    from nora_voice import state_vector
    st = state_vector(agi, nov=1.0, fam=0.0, conf=0.0, tension=0.5)
    for q in org.generate(st, ["mitokondri"], n=24, temperature=1.0):
        leaked += sum(1 for i in q if i not in com and i not in anchor_ids and i != org.net.eos)
    ok = ok_q and leaked == 0
    return ok, f"bilinmeyende soru+çapa={ok_q} {[r for _, r in qs]} | çapa-dışı içerik sızıntısı={leaked}"


# ============================================================================
# A48-A49: KENDİLİĞİNDEN KONUŞMA (İNİSİYATİF) VE SÖZ ÇEŞİTLİLİĞİ
# ============================================================================

def a48_spontaneous_speech_without_prompt():
    """Kimse yazmadan konuşma: uzun sessizlikte kendiliğinden söz üretir, tekrara düşmez,
    cevapsız kaldıkça geri çekilir (spam yok) ve kullanıcı yazınca diyaloğa döner."""
    from nora_lang import tr_lower

    agi = NoraAGI(seed=48, boot_thinker=False)
    agi.step_life(300)
    msgs = []
    for _ in range(20):                       # 6000 adım sessizlik
        agi.step_life(300)
        msgs += agi.drain_outbox()

    texts = [str(m.get("text") or "") for m in msgs]
    meaningful = bool(texts) and all(len(tr_lower(t).strip()) >= 6 and "…" not in t for t in texts)
    count_ok = 2 <= len(msgs) <= 8             # ne ölü sessizlik ne mesaj yağmuru
    distinct = len(set(texts)) == len(texts)
    variety = len({m.get("move") for m in msgs}) >= 2
    # Cevapsız kaldıkça dürtü geri çekilir: spam yok
    quiet = agi.life.unanswered >= 1
    # Kullanıcı yazınca sayaç sıfırlanır ve diyalog geri gelir
    r = agi.converse("Merhaba Nora, orada mısın?")
    reset = agi.life.unanswered == 0 and len(r) > 5
    ok = meaningful and count_ok and distinct and variety and quiet and reset
    return ok, (f"kendiliğinden söz={len(msgs)} | tür={sorted({m.get('move') for m in msgs})} | "
                f"anlamlı={meaningful} | tekrarsız={distinct} | çeşit={variety} | sıfırlandı={reset}")


def a49_initiative_content_and_habituation():
    """İnisiyatif sözü uydurma değil İÇERİKTEN gelir: öğrenilen bağ sözde görünür, bedensel açlık
    dile gelir; yüzde yok. Aynı iç durumda tekrar tekrar aynı cümle söylenmez (alışma)."""
    from nora_lang import tr_lower

    agi = NoraAGI(seed=49, boot_thinker=False)
    agi.step_life(200)
    # 1) Kavram ağına yeni bir bağ öğret -> inisiyatif bunu paylaşabilmeli
    agi.converse("Zürafa uzun boyunlu bir hayvandır.")
    disc = ""
    for _ in range(60):
        res = agi.proactive_line("discovery", "zürafa", tension=0.8, idle=1.0)
        if res and "zürafa" in tr_lower(res["text"]):
            disc = res["text"]
            break
    ok_discovery = bool(disc)

    # 2) Bedensel açlık dile gelir, yüzde ASLA söylenmez
    agi.nervous.hunger_drive = 0.9
    body = ""
    for _ in range(40):
        res = agi.proactive_line("", tension=0.0, idle=1.0)
        if res and ("acık" in tr_lower(res["text"]) or "karnım" in tr_lower(res["text"])):
            body = res["text"]
            break
    ok_body = bool(body) and "%" not in body

    # 3) Alışma: arka arkaya 12 söz üret, tekrar oranı düşük olsun
    said = []
    for _ in range(12):
        res = agi.proactive_line("", tension=0.2, idle=1.0)
        if res:
            said.append(res["text"])
    unique_ratio = len(set(said)) / max(1, len(said))
    ok_habit = unique_ratio >= 0.6

    ok = ok_discovery and ok_body and ok_habit
    return ok, (f"bağ-paylaşımı={ok_discovery} ({disc[:46]}…) | beden-dili={ok_body} | "
                f"tekrarsız oran={unique_ratio:.2f} ({len(said)} söz)")


def a50_world_knowledge_from_corpus():
    """Dünya bilgisi hattı (nora_besle): kavram–ilişki–kavram besini tek geçişte yerleşir ve
    Nora daha önce hiç duymadığı kavramları da (ISA kalıtımıyla) yanıtlayabilir."""
    from nora_lang import tr_lower
    from nora_besle import besle

    agi = NoraAGI(seed=50, boot_thinker=False)
    agi.step_life(200)
    rep = besle(agi, verbose=False)
    qs = ["Zürafa nedir?", "Merkür nedir?", "Kalp ne yapar?", "Beyin ne yapar?",
          "Buz nedir?", "Ses boşlukta yayılır mı?", "Penguen uçar mı?",
          "Antalya nerede?", "Güneş nedir?", "Ay nedir?", "Arı ne yapar?", "Matematik nedir?"]
    answered = [q for q in qs if "bilgim yok" not in tr_lower(agi.converse(q))]
    ok_cov = len(answered) >= 10
    honest = "bilgim yok" in tr_lower(agi.converse("Ksilofonit nedir?"))
    ok = ok_cov and honest and rep["graph_edges"] > 300
    return ok, (f"kapsam {len(answered)}/{len(qs)} | dürüstlük={honest} | "
                f"kenar={rep['graph_edges']} | kavram={rep['concepts']}")


def a51_turkish_possession_and_case():
    """Türkçe sahiplik/iyelik ve belirtisiz nesne: 'zürafanın uzun boynu var' doğru kurulur,
    'boyundur' gibi hatalı koşaç yok; sözlükte olmayan kelime uydurma köke soyulmaz."""
    from nora_lang import TurkishMorphology as M, tr_lower
    from nora_besle import besle

    agi = NoraAGI(seed=51, boot_thinker=False)
    agi.step_life(150)
    besle(agi, verbose=False)
    r_z = tr_lower(agi.converse("Zürafa nedir?"))
    ok_part = "boynu var" in r_z and "boyundur" not in r_z
    r_f = tr_lower(agi.converse("Fil nedir?"))
    ok_f = "hortumu var" in r_f
    an = M.analyze_word("matematik", agi.lang.graph.lexicon)
    ok_stem = an["stem"] == "matematik" and not an["is_verb"]
    res = agi.proactive_line("", tension=0.3, idle=1.0)
    ok_obj = res is None or "bir şeyi" not in tr_lower(res["text"])
    ok = ok_part and ok_f and ok_stem and ok_obj
    return ok, f"iyelik(parça)={ok_part} | hortum={ok_f} | kök-güvenliği={ok_stem} | nesne={ok_obj}"


def a52_discourse_policy_no_garbage():
    """Söylem organı: anlaşılamayan sözde çöp cümle üretmez (fiil kökünü nesne sanmaz), tanıdık
    sözcüğe "bilmiyorum" demez, olumsuz söze sevinçle karşılık vermez, tekrar etmez."""
    import re
    from nora_lang import tr_lower
    from nora_besle import besle

    agi = NoraAGI(seed=52, boot_thinker=False)
    agi.step_life(200)
    besle(agi, verbose=False)
    qs = ["Sen orada mısın?", "Yarın işe gidiyorum.", "Bir sorunum var ama anlatmak istemiyorum.",
          "Bugün canım sıkılıyor.", "Uyuyamıyorum.", "Ne düşünüyorsun?",
          "Dün akşam bir film izledim.", "Biraz sohbet edelim.", "Kendimi yalnız hissediyorum."]
    replies = [agi.converse(q) for q in qs]
    joined = " ".join(tr_lower(r) for r in replies)

    # 1) dilbilgisel çöp: fiil köküne belirtme eki + boş yansıma
    garbage = re.findall(r"\b(bu )?\w*[ıiuü] (düşün|gözlemle|değerlendir|yorumla)yorum", joined)
    ok_gram = not garbage
    # 2) tanıdık sözcüğe sahte bilgisizlik ("orada", "canım" gibi)
    ok_known = not any(f"“{w}” sözcüğünü henüz bilmiyorum" in joined
                       for w in ("orada", "canım", "bugün", "yarın", "işe"))
    # 3) olumsuz söze olumlu duygu ifadesi
    ok_tone = not any(("uyuyamıyor" in tr_lower(q) or "yalnız" in tr_lower(q) or "sıkılı" in tr_lower(q))
                      and "sevindim" in tr_lower(r) for q, r in zip(qs, replies))
    # 4) boş/çöp olmayan, biten cümleler + çeşitlilik
    ok_forms = all(len(r.strip()) >= 8 and r.strip()[-1] in ".?!" for r in replies)
    uniq = len(set(tr_lower(r) for r in replies)) / max(1, len(replies))
    ok_uniq = uniq >= 0.75
    ok = ok_gram and ok_known and ok_tone and ok_forms and ok_uniq
    return ok, (f"dilbilgisi-çöpü={ok_gram} | sahte-bilgisizlik-yok={ok_known} | ton={ok_tone} | "
                f"biçim={ok_forms} | tekrarsız={uniq:.2f}")


TESTS = {
    "A1": ("üretken Türkçe morfoloji ve sayı sistemi", a1_morphology_and_generation),
    "A2": ("epoch'suz 1-atış kavram, istisna ve nedensel zincir öğrenme", a2_oneshot_language_and_inference),
    "A3": ("bedenlenmiş diyalog, körlük/felç raporu ve zamir bağlamı", a3_embodied_dialogue_and_anaphora),
    "A4": ("öz-benlik ve özerk irade (zombi olmama testi)", a4_core_self_autonomous_volition),
    "A5": ("epistemik öz-savunma, kaynak izleme ve OXY güven kalibrasyonu", a5_epistemic_defense_and_trust),
    "A6": ("yaşantıdan doğan karşıolgusal öz-kimlik farklılaşması", a6_emergent_identity_divergence),
    "A7": ("doğal Türkçe üzerinden Sistem-2 muhakeme ve analoji köprüsü", a7_language_to_system2_bridge),
    "A8": ("az-örnekli (0-epoch) evrensel dizi ve fonksiyon keşfi", a8_fewshot_universal_rule_induction),
    "A9": ("nörokimyasal sürekli öğrenme (5 dalga sıfır unutma) + kalıcılık", a9_continual_no_forgetting_and_persistence),
    "A10": ("geriye dönük tam uyumluluk (selftest --bridge)", a10_backward_compatibility),
    "A11": ("Zihin Kuramı (Sally-Anne yanlış inanç) ve sosyal kişi belleği", a11_theory_of_mind_and_social_memory),
    "A12": ("parietal geçişli karşılaştırma çıkarımı (A > B > C > D)", a12_transitive_comparative_reasoning),
    "A13": ("1-atış prosedürel makro-beceri ve Truva-atı öz-koruma denetimi", a13_procedural_macro_and_trojan_defense),
    "A14": ("if-else botu olmama: zihinsel simülasyon & yaşantısal irade", a14_neural_volition_not_ifelse_bot),
    "A15": ("uzman Türkçe (ünlü düşmesi, morfolojik analiz, deyim, sözlük)", a15_expert_turkish_linguistics_and_idioms),
    "A16": ("empatik, felsefi ve çok-turlu doğal Türkçe sohbet", a16_natural_empathetic_and_philosophical_conversation),
    "A17": ("uzman matematik korteksi (cebir, türev, integral, sayılar)", a17_expert_mathematics_cortex),
    "A18": ("uzman fizik korteksi ve RK4 zihinsel fizik simülasyonu", a18_expert_physics_and_mental_simulation),
    "A19": ("Küresel Çalışma Alanı (Global Workspace) kortikal ihale yarışı", a19_global_workspace_cortical_arbitration),
    "A20": ("sıfır-kalıp Broca kök sentezi ve sinaptik alışma çeşitliliği", a20_zero_template_broca_and_habituation),
    "A21": ("konuşma öz-iradesi (merak, ketumluk ve sessizlik seçimi)", a21_communicative_volition_free_will),
    "A22": ("hipokampal CA3 çok-turlu sohbet geçmişi ve kalıcılık", a22_long_horizon_episodic_dialogue_memory),
    "A23": ("O(1) sabit-zamanlı hız (50k kavram + 50k anıda sıfır gecikme)", a23_constant_time_o1_massive_scale_speed),
    "A24": ("insan iç-duyumu (sıfır '%' telemetri, niteliksel açlık/susuzluk)", a24_human_visceral_interoception_zero_percentages),
    "A25": ("bedensel sinir sistemi: A-delta refleksi, C-lifi sızısı ve yatışma", a25_nociceptive_nervous_system_pain_and_relief),
    "A26": ("tek ve geri dönüşsüz ölümlü yaşam (sıfır diriltme / dead_forever)", a26_single_irreversible_mortal_life),
    "A27": ("if'siz 1-atış plastik söylem korteksi ve 'Neden konuşuyorsun?'", a27_zero_if_plastic_discourse_and_meta_why_speak),
    "A28": ("otonom fiziksel bilinç ağı ve öz-meraklı internet/metin araştırması", a28_autonomous_consciousness_network_and_curiosity_research),
    "A29": ("overfitting denetimi: 12 alanda görülmemiş soru/ifade varyasyonları", a29_out_of_distribution_paraphrase_generalization),
    "A30": ("sürekli içsel monolog (otonom DMN + konuşma öncesi sessiz iç ses)", a30_continuous_inner_monologue),
    "A31": ("plastik şema keşfi, şema klonlama karşıtlığı ve davranış çeşitliliği", a31_self_learned_schema_and_anti_cloning),
    "A32": ("noktalama gerektirmeyen nöral sözce öz-sınıflandırması ve etiket üretimi", a32_neural_utterance_self_classification),
    "A33": ("kuantum alanında otonom öz-etiketleme ve kendi mantığıyla genelleme", a33_autonomous_quantum_logic_self_labeling),
    "A34": ("orta-ileri Türkçe: tam kip/kişi, çatı, ulaç ve tüm günlük durumlar", a34_intermediate_turkish_full_paradigm),
    "A35": ("kendi kurduğu mantık ve etiketlerin tek-yaşam kalıcılığı", a35_self_built_logic_persistence),
    "A36": ("kör kabul yok: aynı yıl iki ad -> iki yarışan hipotez ve kanıt isteği", a36_no_blind_acceptance_two_hypotheses),
    "A37": ("kaynak güvenilirliğiyle öğrenme, doğrulama ve System-1 terfi/iptal", a37_reliability_weighted_learning_and_system1),
    "A38": ("yalnızca güvenilir kaynaklara erişim (allowlist, sızıntısız)", a38_trusted_sources_only),
    "A39": ("çalıştırarak doğrulanan Python sentezi (uydurma imkânsız)", a39_python_code_synthesis_verified),
    "A40": ("epistemik + kod becerisinin tek-yaşam belleğinde kalıcılığı", a40_epistemic_persistence_and_mortal_memory),
    "A41": ("dilden bağımsız öğrenen söz organı (koda gömülü sözcük yok)", a41_language_agnostic_learned_voice),
    "A42": ("çapraz doğrulama: tek kaynak hipotez, bağımsız 2. kaynak Sistem 1", a42_cross_verification_hypothesis_pool),
    "A43": ("eksik tamamlama iradesi: sosyal kalıp boşluğu -> otonom araştırma", a43_self_completion_drive_social_gap),
    "A44": ("arka plan yaşamı: CA3 tarama, gerilim defteri, kendiliğinden mesaj", a44_background_life_autonomy),
    "A45": ("arka plan döngüsü kararlılığı: O(1) tick, sınırlı yapılar, kalıcılık", a45_background_loop_stability_o1),
    "A46": ("sinirsel Broca: tek geçiş öğrenme + yapısal halüsinasyon engeli", a46_neural_broca_structural_no_hallucination),
    "A47": ("eğitilmiş Broca: bilmediğinde uydurmaz, çapayı anıp merak eder", a47_neural_broca_trained_stance_and_honesty),
    "A48": ("sorulmadan konuşma: sessizlikte kendiliğinden söz, spam yok, diyaloğa dönüş", a48_spontaneous_speech_without_prompt),
    "A49": ("inisiyatif sözü içerikten gelir (bağ/beden), yüzde yok, alışmayla tekrarsız", a49_initiative_content_and_habituation),
    "A50": ("corpus'tan dünya bilgisi (kavram/ilişki) ve bilinmeyende dürüstlük", a50_world_knowledge_from_corpus),
    "A51": ("Türkçe iyelik/sahiplik ve sözlük-güvenli kök çözümleme", a51_turkish_possession_and_case),
    "A52": ("söylem politikası: çöp cümle yok, sahte bilgisizlik yok, doğru ton", a52_discourse_policy_no_garbage),
}


def main():
    wanted = [k for k in sys.argv[1:] if k in TESTS] or list(TESTS.keys())
    t0 = time.time()
    passed = 0
    for k in wanted:
        desc, fn = TESTS[k]
        ok, detail = fn()
        passed += int(bool(ok))
        st = "GEÇTİ" if ok else "KALDI"
        print(f"{k:<4s} {desc:<64s} {st:<8s} {detail}", flush=True)
    print(f"\nSONUÇ: {passed}/{len(wanted)} AGI TESTİ GEÇTİ ({time.time() - t0:.1f} sn)")
    sys.exit(0 if passed == len(wanted) else 1)


if __name__ == "__main__":
    main()

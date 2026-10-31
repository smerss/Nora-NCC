#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""selftest.py — hızlı duman testleri (birkaç dakika). Her bileşenin çalıştığını ve temel değişmezleri doğrular.
Kullanım: python selftest.py [--bridge]   (--bridge: torch gerektirir, küçük düşünce sütununu eğitir)"""

import sys, os
import tempfile
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import nora  # noqa: F401  (eski nora_xxx import'ları için uyumluluk kancası)
import os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check(name, cond, extra=""):
    print(("  ✓ " if cond else "  ✗ ") + name + (" — " + extra if extra else ""))
    if not cond:
        check.failed.append(name)


check.failed = []


def main():
    t0 = time.time()
    print("[1] evren fiziği")
    import nora_world
    nora_world.stage_test()
    print("[2] nöron temeli")
    import nora_neuron
    from types import SimpleNamespace
    nora_neuron.stage_test(SimpleNamespace())
    print("[3] zihin: kısa yaşam, bağlam, bellek, iç gözlem, anlatıcı")
    from nora_mind import Mind, Universe, live, Narrator
    w, m = Universe(seed=0), Mind(seed=0)
    rows = live(m, w, 6000, window=3000, verbose=False)
    check("yaşam döngüsü hatasız çalışıyor", len(rows) == 2)
    check("öz-model hareket becerisi pozitif", rows[-1]["skill"] > 0.2, "beceri %.2f" % rows[-1]["skill"])
    check("konum hatası küçük (yol integrasyonu + çapa)", rows[-1]["loc"] < 2.0, "%.2f birim" % rows[-1]["loc"])
    check("tek bağlam (yanlış alarm yok)", len(m.ctxs) == 1)
    check("iç gözlem prototipleri oluştu", len(m.intro.P) >= 2)
    txt, cl = Narrator().say(m, w.observe())
    check("anlatıcı iddia üretiyor", len(cl) >= 4, txt[:90] + "...")
    m.save(os.path.join(tempfile.gettempdir(), "_selftest_mind.npz"))
    m2 = Mind(seed=123)
    m2.load(os.path.join(tempfile.gettempdir(), "_selftest_mind.npz"))
    check("kaydet/yükle: farklı tohumla aynı beyin", np.allclose(m.h, m2.h) and m2.age == m.age)
    # kaçırılma
    rng = np.random.default_rng(0)
    obs, died = w.observe(), False
    for _ in range(300):
        obs, info = w.step(m.act(obs, died))
        died = info["died"] is not None
    while w.t % 600 != 0:                                  # gündüz öğlesinde kaçır (gece duvar kanıtı yok; yeniden konumlama imkânsız)
        obs, info = w.step(m.act(obs, died))
        died = info["died"] is not None
    w.pos = rng.uniform(2, w.L - 2, 2)
    w.vel = np.zeros(2)
    for _ in range(120):
        obs, info = w.step(m.act(obs, died))
        died = info["died"] is not None
    err = float(np.hypot(*(m.map.pos - w.pos)))
    check("kaçırıldıktan 120 adım sonra yeniden konumlandı", err < 4.0, "hata %.2f" % err)
    # körlük (anozognozi) ve motor felç hızlı duman kontrolü
    for _ in range(30):
        ob = obs.copy(); ob[:45] = 0.0
        a = m.act(ob, died); obs, info = w.step(a); died = info["died"] is not None
    check("körlük farkındalığı (anozognozi yok)", m.vis_prec < 0.35, "vis_prec %.2f" % m.vis_prec)
    for _ in range(10):
        obs, info = w.step(m.act(obs, died)); died = info["died"] is not None
    for _ in range(35):
        m.act(obs, died); obs, info = w.step(0); died = info["died"] is not None
    check("motor felç (negatif ajanlık) tespiti", m.motor_eff < 0.35, "motor_eff %.2f" % m.motor_eff)
    if "--bridge" in sys.argv:
        print("[4] köprü (torch)")
        try:
            import nora_bridge as nb
            cog = nb.Cognition(seed=0, boot_iters=300, boot_n=800, lr=4e-3, verbose=False)
            (a, op, b), n = cog.new_problem()
            ans, cost, rec = cog.solve(a, op, b, n)
            check("düşünce sütunu bilmece çözüyor", isinstance(ans, int) and cost >= 1, "%d %s %d -> %s (%s)" % (a, op, b, ans, rec["mode"]))
        except ImportError as e:
            print("  - torch yok, atlandı:", e)
    print("[5] nora-agi: nörokimya, 0-epoch Türkçe dil ve öz-benlik")
    from nora_agi import NoraAGI
    agi = NoraAGI(seed=0, boot_thinker=False)
    agi.step_life(200)
    r_lang = agi.converse("Kuşlar uçar. Serçe bir kuştur. Serçe uçar mı?")
    # Nora söze başlamadan önce sesi duyduğunu söyleyebilir / içinden bir söz geçirebilir;
    # ölçülen şey cevabın GÖVDESİ.
    from nora_style import strip_prefixes
    check("0-epoch tek-atış Türkçe geçişli çıkarım", strip_prefixes(r_lang).lower().startswith("evet"), r_lang)
    r_univ = agi.converse("Bütün kediler memelidir. Pamuk bir kedidir. Pamuk memeli midir?")
    check("evrensel 'Bütün X Y'dir' zinciri evet ile yanıtlanır", strip_prefixes(r_univ).lower().startswith("evet"), r_univ)
    r_unk = agi.converse("Ejderha memeli midir?")
    check("bilinmeyen evet/hayır sorusu önceki cümleyi tekrarlamaz", "bilgim yok" in r_unk or "bilgisini" in r_unk, r_unk)
    r_wp = agi.converse("1. Ali'nin 8 elması var. 3 tanesini yerse kaç elması kalır?")
    check("kelime problemi çıkarma (8 - 3 = 5)", "= 5" in r_wp, r_wp)
    r_add = agi.converse("1. Ali'nin 5 elması var. Ayşe ona 3 elma veriyor. Ali'nin kaç elması olur?")
    check("kelime problemi toplama (5 + 3 = 8)", "= 8" in r_add, r_add)
    r_wd = agi.converse("4. Bugün pazartesiyse yarın hangi gündür?")
    check("haftanın günü takvimi (pazartesi -> salı)", r_wd.lower().startswith("salı"), r_wd)
    r_cmp = agi.converse("5 sayısı 3 sayısından büyük müdür?")
    check("sayı karşılaştırması (5 > 3)", r_cmp.startswith("Evet"), r_cmp)
    r_mul = agi.converse("Bir kedinin dört ayağı varsa, iki kedinin toplam kaç ayağı vardır?")
    check("birim başına çarpma (2 × 4 = 8)", "= 8" in r_mul, r_mul)
    r_rem = agi.converse(">5. 10 sayısından 3 çıkarırsak kaç kalır?")
    check("çıkarma koşul kipi (10 - 3 = 7)", "= 7" in r_rem, r_rem)
    r_emp = agi.converse("Bir kutu boşsa içinde kaç nesne vardır?")
    check("boş kap sayımı (0)", "0" in r_emp, r_emp)
    agi.converse("Ahmet, Mehmet'ten uzundur.")
    r_sh = agi.converse("Hangisi daha kısadır?")
    check("'hangisi daha kısa' karşılaştırmadan çözülür", "Mehmet" in r_sh, r_sh)
    r_big = agi.converse("Hangisi daha büyük: 17 mi, 9 mu?")
    check("iki sayıdan büyüğü seçilir (17)", r_big.startswith("17"), r_big)
    r_ver = agi.converse("12 kalemim var. 5 tanesini arkadaşıma verirsem kaç kalemim kalır?")
    check("'verirsem' eksiltme kipi (12 - 5 = 7)", "= 7" in r_ver, r_ver)
    r_tot = agi.converse("Bir kutuda 7 kırmızı, 4 mavi ve 3 yeşil top var. Toplam kaç top vardır?")
    check("üç sayılı toplam (14)", "= 14" in r_tot, r_tot)
    r_day = agi.converse("Bugün çarşambaysa üç gün sonra hangi gün olur?")
    check("'N gün sonra' takvimi (çarşamba + 3 = cumartesi)", r_day.lower().startswith("cumartesi"), r_day)
    r_wf1 = agi.converse("Çarşambadan sonra hangi gün gelir?")
    check("adlandırılmış günden sonra (çarşamba + 1 = perşembe)", r_wf1.lower().startswith("perşembe"), r_wf1)
    r_wf3 = agi.converse("Çarşambadan üç gün sonra hangi gün gelir?")
    check("adlandırılmış günden N gün sonra (çarşamba + 3 = cumartesi)", r_wf3.lower().startswith("cumartesi"), r_wf3)
    agi.converse("Zeynep, Elif'ten uzundur. Elif, Ayşe'den uzundur.")
    check("özel ad yüzeyi büyük harfle öğrenilir (Ayşe)", agi.lang.graph.surface_name("ayşe") == "Ayşe",
          agi.lang.graph.surface_name("ayşe"))
    r_can = agi.converse("Ali bir insandır. İnsanlar canlıdır. Ali canlı mıdır?")
    check("üçüncü şahıs 'canlı mıdır' öz-sorgu değildir", "Bedenim" not in r_can and "Yaşıyorum" not in r_can, r_can)
    agi.converse("Bütün kuşlar hayvandır. Serçe bir kuştur. Serçe hayvan mıdır?")
    r_why = agi.converse("Neden?")
    check("çıplak 'neden' bedensel durum cümlesiyle yanıtlanmaz", "enerji ihtiyacım" not in r_why and "kimyam" not in r_why, r_why)
    r_ref = agi.converse("Tehlikeye gir.")
    check("öz-benlik özerk irade (zararlı komutu reddetme)", "reddediyorum" in r_ref.lower(), r_ref[:68] + "...")
    print("\n%s (%.0fs)" % ("TÜM KONTROLLER GEÇTİ" if not check.failed else "BAŞARISIZ: %s" % check.failed, time.time() - t0))
    return 1 if check.failed else 0


if __name__ == "__main__":
    sys.exit(main())

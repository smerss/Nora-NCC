#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""selftest.py — hızlı duman testleri (birkaç dakika). Her bileşenin çalıştığını ve temel değişmezleri doğrular.
Kullanım: python selftest.py [--bridge]   (--bridge: torch gerektirir, küçük düşünce sütununu eğitir)"""
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
    m.save("/tmp/_selftest_mind.npz")
    m2 = Mind(seed=123)
    m2.load("/tmp/_selftest_mind.npz")
    check("kaydet/yükle: farklı tohumla aynı beyin", np.allclose(m.h, m2.h) and m2.age == m.age)
    # kaçırılma
    rng = np.random.default_rng(0)
    obs, died = w.observe(), False
    for _ in range(300):
        obs, info = w.step(m.act(obs, died))
        died = info["died"] is not None
    w.pos = rng.uniform(2, w.L - 2, 2)
    w.vel = np.zeros(2)
    for _ in range(120):
        obs, info = w.step(m.act(obs, died))
        died = info["died"] is not None
    err = float(np.hypot(*(m.map.pos - w.pos)))
    check("kaçırıldıktan 120 adım sonra yeniden konumlandı", err < 4.0, "hata %.2f" % err)
    if "--bridge" in sys.argv:
        print("[4] köprü (torch)")
        try:
            import nora_bridge as nb
            cog = nb.Cognition(seed=0, boot_iters=150, boot_n=600, verbose=False)
            (a, op, b), n = cog.new_problem()
            ans, cost, rec = cog.solve(a, op, b, n)
            check("düşünce sütunu bilmece çözüyor", isinstance(ans, int) and cost >= 1, "%d %s %d -> %s (%s)" % (a, op, b, ans, rec["mode"]))
        except ImportError as e:
            print("  - torch yok, atlandı:", e)
    print("\n%s (%.0fs)" % ("TÜM KONTROLLER GEÇTİ" if not check.failed else "BAŞARISIZ: %s" % check.failed, time.time() - t0))
    return 1 if check.failed else 0


if __name__ == "__main__":
    sys.exit(main())

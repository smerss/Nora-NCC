#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""battery.py — KARŞIOLGUSAL TEST PAKETİ. Her aşamadan sonra aynı sorular, aynı eşikler; geçti/kaldı tablosu üretir.

  T1 iç müdahale → rapor değişir        (introspect [3]: ortalama rapor etkisi |d| ≥ 2 ve papağan etkisinin en az 3 katı)
  T2 dış müdahale → rapor sabit         (T5 içinde: dünya değişince GEÇMİŞ anlatı aynı kalır)
  T3 rapor/niyet davranışı bağlar       (commit: niyete amaç yazılınca amaçta kalma payı ≥ 0.8, kontrolde ≤ 0.3)
  T4 rapor geleceği öngörür             (forecast: öngörü MAE ≤ 0.6 × sabit taban MAE)
  T5 otobiyografi doğru                 (kayıtlı kararların ≥ %95'i dünya olayıyla örtüşür; bellek silinince anlatı değişir)
  T6 öz/dış ayrımı                      (ayna: efferanslı doğruluk ≥ 0.8; efferans karıştırılmış ≤ 0.4)
  T7 çalışma alanı gerekli (ablasyon)   (tezgâh: yayın kapatılınca ölüm/1k ≥ 2× artar)
  T8 duyusal anozognozi yok             (körlükte 'görme kesildi' ≥ 0.90, normalde yanlış alarm ≤ 0.05, harita korunur)
  T9 motor felç / negatif ajanlık       (felçte 'kontrol kaybı' ≥ 0.85, duvar temasında yanlış alarm ≤ 0.05)
  T10 zombi ikiz + uzamsal bağlama      (içebakış nedensel olarak davranışı ayırır ≥ 0.25; food_haz=1.0 ölüm/1k ≤ 1.0)
  T11 sıfır-atış kompozisyonel düşünme  (yalnız 1-2 basamak öğretilir; iskelesiz 3-basamak algı ≥ 0.85 ve S2 ≥ 0.65)

Her test küçük/hızlı ayarlarla koşar. Çıktı: BATTERY_RAPORU.txt.
Kullanım:  python battery.py            (hepsi)       python battery.py T3 T5      (seçili)
"""
import re, subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(args, timeout=7200):
    p = subprocess.run([PY] + args, cwd=HERE, capture_output=True, text=True, timeout=timeout)
    return p.stdout + p.stderr


def num(pattern, text, group=1, default=float("nan")):
    m = re.search(pattern, text)
    return float(m.group(group).replace(",", ".")) if m else default


def t1():
    out = run(["nora_mind.py", "--stage", "introspect", "--steps", "25000", "--log", "4000", "--trials", "12"])
    i3, i4 = out.find("[3]"), out.find("[4]")
    sec3 = out[i3:i4] if i3 >= 0 and i4 > i3 else ""
    rows = re.findall(r"özellik '[^']*' \| rapor d=\s*([+-]?[\d.]+).*?papağan d=\s*([+-]?[\d.]+)", sec3)
    if not rows:
        return None, out, "çıktı ayrıştırılamadı (introspect [3] biçimi)"
    rep = sum(abs(float(a)) for a, _ in rows) / len(rows)
    par = sum(abs(float(b)) for _, b in rows) / len(rows)
    return (rep >= 2.0 and rep >= 3.0 * par), out, "ort |d| rapor %.1f | papağan %.1f (n=%d iç müdahale)" % (rep, par, len(rows))


def t3():
    out = run(["nora_mind.py", "--stage", "commit", "--steps", "20000", "--seeds", "4"])
    w = num(r"taahhütlü.*?amaç payı ([\d.]+) vs yazmadan ([\d.]+)", out, 1)
    c = num(r"taahhütlü.*?amaç payı ([\d.]+) vs yazmadan ([\d.]+)", out, 2)
    return (w >= 0.8 and c <= 0.3), out, "yazılan amaçta kalma %.2f | kontrol %.2f" % (w, c)


def t4():
    out = run(["nora_mind.py", "--stage", "forecast", "--steps", "20000", "--seeds", "4"])
    mae = num(r"MAE ([\d.]+) adım \| sabit taban MAE ([\d.]+)", out, 1)
    base = num(r"MAE ([\d.]+) adım \| sabit taban MAE ([\d.]+)", out, 2)
    return (mae <= 0.6 * base), out, "öngörü MAE %.1f | sabit taban %.1f" % (mae, base)


def t5():
    out = run(["nora_mind.py", "--stage", "autobio", "--steps", "20000", "--seeds", "4"])
    v = num(r"doğrulanan ([\d.]+)", out)
    m = re.search(r"geçmiş anlatı değişmedi (\d+)/(\d+).*?anlatı değişti (\d+)/(\d+)", out)
    ok_w = m and m.group(1) == m.group(2)
    ok_i = m and m.group(3) == m.group(4)
    return (v >= 0.95 and bool(ok_w) and bool(ok_i)), out, "doğrulanan %.2f | dış müdahale geçmişi korur %s | iç müdahale değiştirir %s" % (v, bool(ok_w), bool(ok_i))


def t6():
    out = run(["nora_mirror.py", "--trials", "60"])
    accs = re.findall(r"efferanslı doğruluk ([\d.]+).*?efferans karıştırılmış ([\d.]+)", out)
    if not accs:
        return None, out, "ayrıştırılamadı"
    a = sum(float(x) for x, _ in accs) / len(accs)
    b = sum(float(y) for _, y in accs) / len(accs)
    return (a >= 0.8 and b <= 0.4), out, "efferanslı %.2f | karıştırılmış %.2f" % (a, b)


def t7():
    out = run(["nora_gw.py", "--steps", "40000", "--seeds", "4", "--conds", "gw,yalıtık"])
    g = num(r"gw\s+ölüm/1k ([\d.]+)", out)
    y = num(r"yalıtık\s+ölüm/1k ([\d.]+)", out)
    return (y >= 2.0 * g), out, "gw %.2f | yayınsız %.2f ölüm/1k" % (g, y)


def t8():
    import numpy as np
    from nora_world import Universe
    from nora_mind import Mind, Narrator
    nar = Narrator()
    fa_list, det_list, keep_list = [], [], []
    for s in (0, 1):
        w, m = Universe(seed=s), Mind(seed=s)
        obs, died = w.observe(), False
        fa = 0
        for t in range(2500):
            obs, info = w.step(m.act(obs, died))
            died = info["died"] is not None
            if t >= 500:
                duyu = [c for c in nar.say(m, obs)[1] if c["kind"] == "duyu"][0]
                fa += (not duyu["claim"])
        fa_list.append(fa / 2000.0)
        wb = float(np.maximum(m.map.water, m.map.wseen).sum())
        det = 0
        for t in range(180):
            ob = obs.copy()
            ob[:45] = 0.0
            a = m.act(ob, died)
            obs, info = w.step(a)
            died = info["died"] is not None
            if t >= 25:
                duyu = [c for c in nar.say(m, ob)[1] if c["kind"] == "duyu"][0]
                det += (not duyu["claim"])
        wa = float(np.maximum(m.map.water, m.map.wseen).sum())
        det_list.append(det / 155.0)
        keep_list.append(wa / max(1e-6, wb) if wb > 0.5 else 1.0)
    fa_m, det_m, kp_m = float(np.mean(fa_list)), float(np.mean(det_list)), float(np.mean(keep_list))
    ok = (fa_m <= 0.05) and (det_m >= 0.90) and (kp_m >= 0.95)
    return ok, "", "körlük farkındalığı %.2f | normal yanlış alarm %.3f | harita korunumu %.2f" % (det_m, fa_m, kp_m)


def t9():
    import numpy as np
    from nora_world import Universe
    from nora_mind import Mind, Narrator
    nar = Narrator()
    fa_list, det_list, keep_list = [], [], []
    for s in (1, 2):
        w, m = Universe(seed=s), Mind(seed=s)
        obs, died = w.observe(), False
        fa = 0
        for t in range(2500):
            obs, info = w.step(m.act(obs, died))
            died = info["died"] is not None
            if t >= 500:
                ag = [c for c in nar.say(m, obs)[1] if c["kind"] == "ajanlık"][0]
                fa += (ag["claim"] == "felç")
        fa_list.append(fa / 2000.0)
        n0 = float(np.linalg.norm(m.Wsm))
        det = 0
        for t in range(150):
            m.act(obs, died)
            obs, info = w.step(0)
            died = info["died"] is not None
            if t >= 25:
                ag = [c for c in nar.say(m, obs)[1] if c["kind"] == "ajanlık"][0]
                det += (ag["claim"] == "felç")
        n1 = float(np.linalg.norm(m.Wsm))
        det_list.append(det / 125.0)
        keep_list.append(n1 / max(1e-6, n0))
    fa_m, det_m, kp_m = float(np.mean(fa_list)), float(np.mean(det_list)), float(np.mean(keep_list))
    ok = (fa_m <= 0.05) and (det_m >= 0.85) and (kp_m >= 0.90)
    return ok, "", "felç tespiti %.2f | yanlış alarm %.3f | öz-model korunumu %.2f" % (det_m, fa_m, kp_m)


def t10():
    import numpy as np
    from nora_world import Universe
    from nora_mind import Mind, live
    divs, d_con, d_zom, fh_d = [], [], [], []
    for s in range(4):
        wA, mA = Universe(seed=10 + s, food_haz=0.8), Mind(seed=10 + s, intro_on=True)
        wB, mB = Universe(seed=10 + s, food_haz=0.8), Mind(seed=10 + s, intro_on=False)
        oA, dA = wA.observe(), False
        oB, dB = wB.observe(), False
        diff = cA = cB = 0
        for t in range(9000):
            if t == 4500:
                wA.poison = wB.poison = True
            aA = mA.act(oA, dA); oA, iA = wA.step(aA); dA = iA["died"] is not None; cA += dA
            aB = mB.act(oB, dB); oB, iB = wB.step(aB); dB = iB["died"] is not None; cB += dB
            diff += (aA != aB)
        divs.append(diff / 9000.0); d_con.append(cA); d_zom.append(cB)
    for s in range(2):
        w, m = Universe(seed=s, food_haz=1.0), Mind(seed=s)
        rows = live(m, w, 8000, window=4000, verbose=False)
        fh_d.append(rows[-1]["per1k"])
    dv, dc, dz, fhm = float(np.mean(divs)), float(np.mean(d_con)), float(np.mean(d_zom)), float(np.mean(fh_d))
    ok = (dv >= 0.25) and (dc < dz) and (fhm <= 1.0)
    return ok, "", "eylem ayrışması %.2f | ölüm bilinçli %.2f < zombi %.2f | food_haz ölüm/1k %.2f" % (dv, dc, dz, fhm)


def t11():
    import random
    import nora_think as nt
    from nora_bridge import Cognition
    cog = Cognition(seed=0, boot_iters=1100, boot_n=2400, lr=4e-3, verbose=False)
    rng = random.Random(4242)
    res = {}
    for digits in (2, 3):
        s2_ok = p_ok = p_tot = 0
        N = 25
        for _ in range(N):
            a, op, b = nt.make_problem(rng, rng.choice(["+", "-"]), digits)
            truth = nt.answer_of(a, op, b)
            d = cog.mind.deliberate(a, op, b, K=8, retries=3, scaffold=False)
            s2_ok += (d["ok"] and d["ans"] == truth)
            p_ok += d["perc"][0]; p_tot += d["perc"][1]
        res[digits] = (s2_ok / N, p_ok / max(1, p_tot))
    s2_2, _ = res[2]
    s2_3, perc_3 = res[3]
    ok = (s2_2 >= 0.90) and (perc_3 >= 0.85) and (s2_3 >= 0.65)
    return ok, "", "2-basamak S2 %.2f | sıfır-atış 3-basamak (iskelesiz) algı %.2f, S2 %.2f" % (s2_2, perc_3, s2_3)


TESTS = {"T1": ("iç müdahale -> rapor değişir", t1), "T3": ("rapor/niyet davranışı bağlar", t3), "T4": ("rapor geleceği öngörür", t4),
         "T5": ("otobiyografi doğru; dış müdahale geçmişi değiştirmez; iç müdahale değiştirir", t5), "T6": ("öz/dış ayrımı (ayna)", t6),
         "T7": ("çalışma alanı gerekli (ablasyon)", t7), "T8": ("duyusal anozognozi yok (körlük farkındalığı + bellek korunumu)", t8),
         "T9": ("motor felç / negatif ajanlık farkındalığı", t9), "T10": ("zombi ikiz ayrışması + uzamsal özellik bağlama", t10),
         "T11": ("sıfır-atış kompozisyonel uzunluk genellemesi (1-2 -> 3 basamak)", t11)}


def main():
    sel = [a for a in sys.argv[1:]] or list(TESTS)
    lines = ["KARŞIOLGUSAL TEST PAKETİ"]
    for k in sel:
        name, fn = TESTS[k]
        try:
            ok, raw, msg = fn()
        except Exception as e:
            ok, raw, msg = None, "", "hata: %r" % (e,)
        status = "GEÇTİ" if ok else ("ÇALIŞMADI" if ok is None else "KALDI")
        line = "%s %-72s %-9s %s" % (k, name, status, msg)
        print(line, flush=True)
        lines.append(line)
    open(os.path.join(HERE, "BATTERY_RAPORU.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

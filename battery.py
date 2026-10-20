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

Her test küçük/hızlı ayarlarla koşar (yaklaşık 20-40 dk, 4 çekirdek). Çıktı: BATTERY_RAPORU.txt.
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


TESTS = {"T1": ("iç müdahale -> rapor değişir", t1), "T3": ("rapor/niyet davranışı bağlar", t3), "T4": ("rapor geleceği öngörür", t4),
         "T5": ("otobiyografi doğru; dış müdahale geçmişi değiştirmez; iç müdahale değiştirir", t5), "T6": ("öz/dış ayrımı (ayna)", t6),
         "T7": ("çalışma alanı gerekli (ablasyon)", t7)}


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

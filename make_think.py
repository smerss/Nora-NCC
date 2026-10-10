#!/usr/bin/env python3
"""Düşünce-izli aritmetik korpusu: cevap ÖNCESİ iz sürmeyi öğretir.
Format: 12 + 34 = [12 + 34 = 46] 46   ve   What is 12 plus 34? [12 + 34 = 46] It is 46.
Köşeliler geniş alfabede var (kod sembolleri). Sayılar 0-99 (öğrenilebilir aralık)."""
import random
import sys

# --reverse: izdeki sonuç TERS yazılır (elde sağdan sola taşınır; Lee vd. 2023), son cevap düz.
REV = "--reverse" in sys.argv
rv = (lambda r: str(r)[::-1]) if REV else str

random.seed(7)
ops = ["+", "-", "*"]
words = ["plus", "minus", "times"]
out = []
for i in range(60000):
    a, b = random.randint(0, 99), random.randint(0, 99)
    k = i % 3
    r = a + b if k == 0 else (a - b if k == 1 else a * b)
    if r < 0:
        continue
    if i % 2 == 0:
        out.append("%d %s %d = [%d %s %d = %s] %d" % (a, ops[k], b, a, ops[k], b, rv(r), r))
    else:
        out.append("What is %d %s %d? [%d %s %d = %s] It is %d." % (a, words[k], b, a, ops[k], b, rv(r), r))
t = "\n".join(out)
open("D:\\NoRa_AGI\\data\\text\\think_math%s.txt" % ("_rev" if REV else ""), "w", encoding="utf-8").write(t)
print("think: %d örnek, %.1fM char" % (len(out), len(t) / 1e6))

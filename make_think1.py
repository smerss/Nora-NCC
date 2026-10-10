#!/usr/bin/env python3
"""Tek-basamak ağırlıklı müfredat: önce P(rakam|'=') kütlesini tavan yaptır.
%80 tek-basamak (0-9), %20 iki-basamak (öğrenilmişi koruma). İz formatı aynı:
A OP B = [A OP B = R] R  ve  What is A OPw B? [A OP B = R] It is R."""
import random
import sys

# --reverse: izdeki sonuç TERS yazılır (elde sağdan sola taşınır; Lee vd. 2023), son cevap düz.
REV = "--reverse" in sys.argv
rv = (lambda r: str(r)[::-1]) if REV else str

random.seed(11)
ops = ["+", "-", "*"]
words = ["plus", "minus", "times"]
out = []
for i in range(60000):
    if i % 5 == 4:  # %20 iki-basamak
        a, b = random.randint(0, 99), random.randint(0, 99)
    else:
        a, b = random.randint(0, 9), random.randint(0, 9)
    k = i % 3
    r = a + b if k == 0 else (a - b if k == 1 else a * b)
    if r < 0:
        continue
    if i % 2 == 0:
        out.append("%d %s %d = [%d %s %d = %s] %d" % (a, ops[k], b, a, ops[k], b, rv(r), r))
    else:
        out.append("What is %d %s %d? [%d %s %d = %s] It is %d." % (a, words[k], b, a, ops[k], b, rv(r), r))
t = "\n".join(out)
open("D:\\NoRa_AGI\\data\\text\\think_1digit%s.txt" % ("_rev" if REV else ""), "w", encoding="utf-8").write(t)
n1 = sum(1 for x in out if len(x.split("=")[0].strip()) <= 5)
print("mufredat: %d ornek, %.1fM char" % (len(out), len(t) / 1e6))

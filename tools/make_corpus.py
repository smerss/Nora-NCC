import sys, os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

#!/usr/bin/env python3
"""Korpus kurucu: kod (yerel sympy) + matematik (sentetik). Tümü D'de."""
import glob
import random

TEXT = "D:\\NoRa_AGI\\data\\text\\"

files = sorted(glob.glob("D:\\NoRa_AGI\\.venv\\Lib\\site-packages\\sympy\\**\\*.py", recursive=True))[:400]
code = []
for f in files:
    try:
        code.append(open(f, encoding="utf-8", errors="ignore").read())
    except Exception:
        pass
code = "\n".join(code)[:8000000]
open(TEXT + "code_py.txt", "w", encoding="utf-8").write(code)
print("kod: %.1fM char, %d dosya" % (len(code) / 1e6, len(files)))

random.seed(0)
ops = ["+", "-", "*"]
words = ["plus", "minus", "times"]
out = []
for i in range(120000):
    a, b = random.randint(0, 999), random.randint(0, 999)
    k = i % 3
    r = a + b if k == 0 else (a - b if k == 1 else a * b)
    if i % 2 == 0:
        out.append("%d %s %d = %d" % (a, ops[k], b, r))
    else:
        out.append("What is %d %s %d? It is %d." % (a, words[k], b, r))
math = "\n".join(out)
open(TEXT + "math_synth.txt", "w", encoding="utf-8").write(math)
print("mat: %.1fM char" % (len(math) / 1e6))

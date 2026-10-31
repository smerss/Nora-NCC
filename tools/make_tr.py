import sys, os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

out, cur, n = [], [], 0
for ln in open("D:\\NoRa_AGI\\data\\text\\turkish_ud.conllu", encoding="utf-8"):
    ln = ln.strip()
    if not ln:
        if cur:
            out.append(" ".join(cur))
            cur = []
    elif ln[0].isdigit() and "-" not in ln.split("\t")[0]:
        f = ln.split("\t")
        cur.append(f[1])
        n += 1
if cur:
    out.append(" ".join(cur))
t = "\n".join(out)
open("D:\\NoRa_AGI\\data\\text\\turkish.txt", "w", encoding="utf-8").write(t)
print("cumle: %d kelime: %d karakter: %d" % (len(out), n, len(t)))
print("ornek:", t[:160])

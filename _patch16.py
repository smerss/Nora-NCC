import pathlib

# ---------------------------------------------------------------- 1) bölünen yan cümleye özneyi taşı
p = pathlib.Path("nora_lang.py")
s = p.read_text(encoding="utf-8")
old = '''                if _ok:
                    _expanded.extend([parts[0].strip(), parts[1].strip()])
                    continue'''
new = '''                if _ok:
                    _expanded.extend([parts[0].strip(), parts[1].strip()])
                    continue
                # Sağ yarı öznesiz kaldıysa ('... ve insana keyif verir') sol yarının öznesi devredilir.
                _subj = getattr(_l, "subj", "") or ""
                if _subj and not getattr(_r, "subj", ""):
                    _cand = f"{_subj} {parts[1].strip()}"
                    try:
                        _c2 = self.wernicke.parse(_cand)
                    except Exception:
                        _c2 = None
                    if _c2 is not None and _c2.kind == "teach_fact" and getattr(_c2, "obj", ""):
                        _expanded.extend([parts[0].strip(), _cand])
                        continue'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
print("lang ok")

# ---------------------------------------------------------------- 2) duygu sözcüğü dünya bilgisi olarak öğretilsin
p = pathlib.Path("nora_besle.py")
s = p.read_text(encoding="utf-8")
old = '''    ("kedi", "ISA", "evcil hayvan"), ("köpek", "ISA", "evcil hayvan"),'''
assert old in s
s = s.replace(old, old + '''
    ("kedi", "CAUSES", "huzur"), ("köpek", "CAUSES", "keyif"),''', 1)
p.write_text(s, encoding="utf-8")
print("besle ok")

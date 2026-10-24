import pathlib

# ---------------------------------------------------------------- 1) boşluk parçası + noktalama
p = pathlib.Path("nora_speculation.py")
s = p.read_text(encoding="utf-8")
old = '''            for txt in (lack, question):
                if txt and all(tr_lower(txt) != tr_lower(pp) for pp in parts):
                    parts.append(txt)
                    gap_parts.append(txt)'''
new = '''            for txt in (lack, question):
                if not txt:
                    continue
                gap_parts.append(txt)
                if all(tr_lower(txt) != tr_lower(pp) for pp in parts):
                    parts.append(txt)'''
assert old in s
s = s.replace(old, new, 1)

# 2) kesin soru sözcüğünde tür/özellik yanıt değildir (ilgisizse susar)
old = '''            for e, _p in hits:
                if e.rel == "CAN_DO":'''
new = '''            _strict = q.get("wh") in ("neden", "niye", "niçin", "nasıl")
            for e, _p in hits:
                if _strict and e.rel in ("ISA", "HAS_PROP") and q_tokens and not (self._tokens(e.obj) & q_tokens):
                    continue
                if e.rel == "CAN_DO":'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
print("spec ok")

# ---------------------------------------------------------------- 2) nora_agi: çift nokta
p = pathlib.Path("nora_agi.py")
s = p.read_text(encoding="utf-8")
old = '''                    bids.append(CorticalBid("hippocampal_concept", 0.95, 0.94, _joined + ".", f"wh:{fr.subj}"))'''
new = '''                    _join_txt = _joined if _joined.endswith((".", "?", "!")) else _joined + "."
                    bids.append(CorticalBid("hippocampal_concept", 0.95, 0.94, _join_txt, f"wh:{fr.subj}"))'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
print("agi ok")

# ---------------------------------------------------------------- 3) öğretim: USED_FOR fiil adı tam
p = pathlib.Path("nora_besle.py")
s = p.read_text(encoding="utf-8")
if '("uzay", "USED_FOR", "teleskopla gözlem")' in s:
    s = s.replace('("uzay", "USED_FOR", "teleskopla gözlem")', '("uzay", "USED_FOR", "teleskopla gözlemlemek")', 1)
    p.write_text(s, encoding="utf-8")
    print("besle ok")
else:
    print("uzay kaydı yok (atlandı)")

import pathlib

p = pathlib.Path("nora_speculation.py")
s = p.read_text(encoding="utf-8")

# 1) last_gap üzerine yazan satır kaldırıldı
old = '''        self.last_gap = getattr(self, "last_gap", "")
        self.last_trace = (f"parçala'''
new = '''        self.last_trace = (f"parçala'''
assert old in s
s = s.replace(old, new, 1)

# 2) yüklem adı bilinmeyen sözcükse YÜZEY biçim korunur ('sıra ol')
old = '''        if verb_idx > 0:
            for lem, case, _p, idx in nouns:
                if case != "NOM" or idx in used_idx:
                    continue
                if idx == verb_idx - 1:
                    if verb and (verb.startswith(lem) or lem.startswith(verb)):
                        continue                 # 'mır mırlar' gibi kök yinelemesi
                    pred_noun = lem
                    used_idx.add(idx)'''
new = '''        if verb_idx > 0:
            for lem, case, _p, idx in nouns:
                if case != "NOM" or idx in used_idx:
                    continue
                if idx == verb_idx - 1:
                    if verb and (verb.startswith(lem) or lem.startswith(verb)):
                        continue                 # 'mır mırlar' gibi kök yinelemesi
                    # Ağda karşılığı olmayan sözcük kırpılmış köküyle değil, duyulduğu gibi saklanır.
                    surf = tokens[idx] if idx < len(tokens) else lem
                    pred_noun = lem if g.query(lem, None) else surf
                    used_idx.add(idx)'''
assert old in s
s = s.replace(old, new, 1)

# 3) parça->işlev: özne tekil lemma, kişi 3SG; fiilin kendi nesnesi korunur ('nefes al')
old = '''                        try:
                            clause = SYN.realize_verb(VerbSpec(
                                verb_root=verb, subj=NP(agent),
                                obj=NP(patient, case="ACC") if patient else None,
                                ins=NP(part, case="INS"), tense="AOR", person=person))'''
new = '''                        try:
                            _own = [w for w in q["verb_phrase"].split() if w != verb and w not in _STOP]
                            _obj = patient or (" ".join(_own) if _own else "")
                            clause = SYN.realize_verb(VerbSpec(
                                verb_root=verb, subj=NP(agent),
                                obj=NP(_obj, case="NOM") if _obj else None,
                                ins=NP(part, case="INS"), tense="AOR", person="3SG"))'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
print("ok")

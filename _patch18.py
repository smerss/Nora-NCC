import pathlib

# ---------------------------------------------------------------- 1) inisiyatif: mastar ve zıtlık kurulumu
p = pathlib.Path("nora_initiative.py")
s = p.read_text(encoding="utf-8")
old = '''        words = obj.split()
        if not words:
            return ""
        last = words[-1]'''
new = '''        words = obj.split()
        if not words:
            return ""
        last = words[-1]
        if rel == "OPPOSITE":
            # 'gecenin gündüzün zıttı olduğunu fark ettim' — karşıtlık, zıt kavramın kendisiyle kurulur.
            try:
                opp = SYN.realize_np(NP(obj, case="GEN")) + " " + SYN.realize_np(NP("zıt", poss=3))
                return SYN.nominalize_dik("ol", poss=3, case="ACC", subj=NP(subj, case="GEN"), advs=(opp,))
            except Exception:
                return ""
        if last.endswith(("mak", "mek")) and last[:-3] in TurkishMorphology.KNOWN_VERB_ROOTS:
            last = last[:-3]
            words = words[:-1] + [last]'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
print("initiative ok")

# ---------------------------------------------------------------- 2) anlatı: çelişen bağda DAHA ÖZEL olan kazanır
p = pathlib.Path("nora_agi.py")
s = p.read_text(encoding="utf-8")
old = '''            if any(e.obj in u or u in e.obj for u in used_objs if u != e.obj):
                continue
            txt = broca.realize_prop(Proposition(subj=cand, rel=e.rel, obj=e.obj))'''
new = '''            _conf = next((i for i, u in enumerate(used_objs) if u != e.obj and (e.obj in u or u in e.obj)), None)
            if _conf is not None:
                # Aynı bilgiyi iki kez söyleme; ama hangisi DAHA ÖZELSE o kalsın ('evcil' < 'evcil hayvan').
                if len(e.obj) > len(used_objs[_conf]):
                    clauses.pop(_conf)
                    edges.pop(_conf)
                    used_objs.pop(_conf)
                else:
                    continue
            txt = broca.realize_prop(Proposition(subj=cand, rel=e.rel, obj=e.obj))'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
print("agi ok")

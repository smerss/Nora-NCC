import pathlib

# ---------------------------------------------------------------- 1) yüklem denetimi: yalnızca birleşik cümleyi reddet
p = pathlib.Path("nora_lang.py")
s = p.read_text(encoding="utf-8")
old = '''        words = obj.split()
        if len(words) > 4:
            return False
        if any(w in {"ve", "ama", "ancak", "fakat", "çünkü", "ki"} for w in words):
            return False
        for w in words:
            an = self.morph.analyze_word(w, self.graph.lexicon)
            if an.get("is_verb") and an.get("tense") in ("PRES", "PAST", "AOR", "EVID", "FUT") \\
                    and str(an.get("person", "")).startswith("3"):
                return False
        return True'''
new = '''        words = obj.split()
        if len(words) > 6:
            return False
        if any(w in {"ve", "ama", "ancak", "fakat", "çünkü", "ki"} for w in words):
            return False
        # Bütün bir yan cümle ad yerine geçmişse (çekimli yüklem eki taşıyorsa) kavram ağına yazılmaz.
        _finite_tails = ("dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür",
                         "mış", "miş", "muş", "müş", "yor", "yım", "yim", "yız", "yiz",
                         "sın", "sin", "sun", "sün")
        return not any(w.endswith(_finite_tails) for w in words)'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
print("lang ok")

# ---------------------------------------------------------------- 2) anlatı: doğrudan bağ önce, miras özellik atlanır
p = pathlib.Path("nora_agi.py")
s = p.read_text(encoding="utf-8")
old = '''        self._narrated.append(cand)
        hits = g.query(cand, None)
        best: dict[str, tuple] = {}
        for e, path in hits:
            cur = best.get(e.rel)
            if cur is None or (e.weight, -len(path or ())) > (cur[0].weight, -len(cur[1] or ())):
                best[e.rel] = (e, path)
        order = ("HAS_PROP", "CAN_DO", "USED_FOR", "ISA", "HAS_PART", "CAUSES", "LOCATED_AT", "OPPOSITE")
        clauses: list[str] = []
        edges: list = []
        used_objs: list[str] = []
        for rel in order:
            if rel in best and len(clauses) < 2:
                e, _path = best[rel]
                # Yineleme ayıklama: 'evcildir' ile 'evcil hayvandır' aynı bilgiyi iki kez söylemez.
                if any(e.obj in u or u in e.obj for u in used_objs if u != e.obj):
                    continue
                txt = broca.realize_prop(Proposition(subj=cand, rel=e.rel, obj=e.obj))
                if txt and txt not in clauses:
                    clauses.append(txt)
                    edges.append(e)
                    used_objs.append(e.obj)'''
new = '''        self._narrated.append(cand)
        hits = g.query(cand, None)
        order = ("HAS_PROP", "CAN_DO", "USED_FOR", "ISA", "HAS_PART", "CAUSES", "LOCATED_AT", "OPPOSITE")
        # DOĞRUDAN bağ, kalıtımla gelen bağdan önce gelir: 'köpek bir evcil hayvandır' > 'köpek bir canlıdır'.
        pool = [(order.index(e.rel), 0 if not path else 1, -float(e.weight), i, e, path)
                for i, (e, path) in enumerate(hits) if e.rel in order]
        pool.sort()
        # Üst kavramdan miras alınan özellik ayırt edici değildir ('köpek evcildir' çünkü evcil hayvan evcildir).
        _parents = {pe.obj for pe, _p in hits if pe.rel == "ISA"}
        inherited: set[str] = set()
        for par in _parents:
            try:
                inherited |= {e2.obj for e2, _p in g.query(par, ("HAS_PROP",))}
            except Exception:
                pass
        clauses: list[str] = []
        edges: list = []
        used_objs: list[str] = []
        seen_rel: dict[str, int] = {}
        for _oi, _inh, _w, _i, e, _path in pool:
            if len(clauses) >= 2:
                break
            seen_rel[e.rel] = seen_rel.get(e.rel, 0) + 1
            if e.rel != "HAS_PROP" and seen_rel[e.rel] > 1:
                continue
            if e.rel == "HAS_PROP" and seen_rel[e.rel] > 2:
                continue
            if e.rel == "HAS_PROP" and e.obj in inherited:
                continue
            if _path and e.rel == "ISA":
                continue                      # kalıtımla gelen tür, doğrudan tür varken söylenmez
            if any(e.obj in u or u in e.obj for u in used_objs if u != e.obj):
                continue
            txt = broca.realize_prop(Proposition(subj=cand, rel=e.rel, obj=e.obj))
            if txt and txt not in clauses:
                clauses.append(txt)
                edges.append(e)
                used_objs.append(e.obj)'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
print("agi ok")

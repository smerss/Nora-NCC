import pathlib

p = pathlib.Path("nora_speculation.py")
s = p.read_text(encoding="utf-8")

# 1) yüklem adı: hâl eki koşulu kalkar, yüzey biçim yeğlenir ('sıra ol')
old = '''            for lem, case, _p, idx in nouns:
                if case != "NOM" or idx in used_idx:
                    continue
                if idx == verb_idx - 1:'''
new = '''            for lem, case, _p, idx in nouns:
                if idx in used_idx:
                    continue
                if idx == verb_idx - 1:'''
assert old in s
s = s.replace(old, new, 1)

# 2) NEDEN: eylem belirliyse yalnız o eyleme bağlı neden sayılır
old = '''        if "CAUSES" in rels:
            _srcs = [q["pred_noun"], q["verb_phrase"], agent, patient]
            if q["verb"] and q["patient"]:
                _srcs.append(f"{q['patient']} {q['verb']}")
            for _vp in list(_srcs):
                if _vp and " " in _vp:
                    _srcs.append(_vp + ("mak" if _vp[-1] in "aıou" else "mek"))
            for c in dict.fromkeys(x for x in _srcs if x):'''
new = '''        if "CAUSES" in rels:
            _specific = [q["pred_noun"], q["verb_phrase"]]
            if q["verb"] and q["patient"]:
                _specific.append(f"{q['patient']} {q['verb']}")
            for _vp in list(_specific):
                if _vp and " " in _vp:
                    _specific.append(_vp + ("mak" if _vp[-1] in "aıou" else "mek"))
            _generic = [agent, patient]
            _light = q["verb"] in ("", "ol", "et", "yap", "bulun", "bul")
            _srcs = _specific + (_generic if _light else [])
            for c in dict.fromkeys(x for x in _srcs if x):'''
assert old in s
s = s.replace(old, new, 1)

# 3) bilinen bağlar: soruyla ilgisiz karşıtlık/olumsuzluk gürültüsü söylenmez
old = '''        order = {"HAS_PROP": 0, "CAN_DO": 1, "USED_FOR": 2, "CAUSES": 3, "HAS_PART": 4, "ISA": 5}
        for c in q["concepts"]:
            hits = [(e, p) for e, p in g.query(c, None) if not p]'''
new = '''        order = {"HAS_PROP": 0, "CAN_DO": 1, "USED_FOR": 2, "CAUSES": 3, "HAS_PART": 4, "ISA": 5}
        if q.get("wh") in ("neden", "niye", "niçin"):
            order = {"CAUSES": 0, "USED_FOR": 1, "CAN_DO": 2, "HAS_PROP": 3, "HAS_PART": 4, "ISA": 5}
        _asked = tr_lower(q.get("raw", ""))
        for c in q["concepts"]:
            hits = [(e, p) for e, p in g.query(c, None) if not p]
            hits = [(e, p) for e, p in hits
                    if e.rel in order or (e.rel == "OPPOSITE" and any(w in _asked for w in ("zıt", "karşıt", "ters")))]'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
print("ok")

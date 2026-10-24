import pathlib

# ---------------------------------------------------------------- 1) spekülasyon: eksik+soru parçası dışa açılır
p = pathlib.Path("nora_speculation.py")
s = p.read_text(encoding="utf-8")
old = '''        if not hypo:
            question = self._curiosity_question(agi, q)
            if question and all(tr_lower(question) != tr_lower(p) for p in parts):
                parts.append(question)'''
new = '''        gap_parts: list[str] = []
        if not hypo:
            lack = self._lack_clause(agi, q)
            question = self._curiosity_question(agi, q)
            for txt in (lack, question):
                if txt and all(tr_lower(txt) != tr_lower(pp) for pp in parts):
                    parts.append(txt)
                    gap_parts.append(txt)
        self.last_gap = " ".join(gap_parts)'''
assert old in s
s = s.replace(old, new, 1)
old = '''        self.last_trace = (f"parçala'''
new = '''        self.last_gap = getattr(self, "last_gap", "")
        self.last_trace = (f"parçala'''
assert old in s
s = s.replace(old, new, 1)
old = '''    def __init__(self, seed: int = 0):'''
new = '''    def __init__(self, seed: int = 0):
        self.last_gap = ""'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
print("spec ok")

# ---------------------------------------------------------------- 2) nora_agi: bağlama
p = pathlib.Path("nora_agi.py")
s = p.read_text(encoding="utf-8")

old = '''        self.initiative = InitiativeCortex(seed=seed + 173)'''
new = '''        self.initiative = InitiativeCortex(seed=seed + 173)
        self.speculation = SpeculativeCortex(seed=seed + 211)'''
assert old in s
s = s.replace(old, new, 1)

old = '''from nora_initiative import InitiativeCortex'''
new = '''from nora_initiative import InitiativeCortex
from nora_speculation import SpeculativeCortex'''
assert old in s
s = s.replace(old, new, 1)

# 2a) soruya bağlam: spekülasyon adayı
old = '''                hits = graph.query(fr.subj, rels)
                if not hits and fr.rel != "ALL" and not _strict:'''
new = '''                _spec, _gap = "", ""
                if _strict:
                    try:
                        _spec = self.speculation.deliberate(self, raw, fr)
                        _gap = getattr(self.speculation, "last_gap", "")
                    except Exception:
                        _spec, _gap = "", ""
                hits = graph.query(fr.subj, rels)
                if not hits and fr.rel != "ALL" and not _strict:'''
assert old in s
s = s.replace(old, new, 1)

# 2b) bilinen bağlar soruyu KARŞILAMIYORSA eksiği söyle
old = '''                    _joined = tr_capitalize("; ".join(_clean))
                    bids.append(CorticalBid("hippocampal_concept", 0.95, 0.94, _joined + ".", f"wh:{fr.subj}"))'''
new = '''                    _joined = tr_capitalize("; ".join(_clean))
                    if _gap and not self._wh_covered(hits, _wh, raw):
                        _joined = (_joined.rstrip(".") + ". " + tr_capitalize(_gap)).strip()
                    bids.append(CorticalBid("hippocampal_concept", 0.95, 0.94, _joined + ".", f"wh:{fr.subj}"))'''
assert old in s
s = s.replace(old, new, 1)

# 2c) hiç bağ yoksa: şablon yerine DÜŞÜNME
old = '''                        if _strict:
                            # Yöntem/neden sorusunda uydurma merak cümlesi kurma: soruyu aynen geri verip
                            # öğrenmeye açık ol (entelektüel dürüstlük).
                            _miss = f"Bunu henüz bilmiyorum: “{raw.strip()}” — bana bir cümleyle öğretir misin?"
                            bids.append(CorticalBid("hippocampal_concept", 0.72, 0.75, _miss,
                                                    f"wh_unknown_strict:{fr.subj}"))'''
new = '''                        if _spec:
                            # Zihin soruyla bağ kurdu: bilinen parçalar + sınanmış varsayım + tam eksik + kendi sorusu.
                            bids.append(CorticalBid("speculative_cortex", 0.86, 0.88, _spec,
                                                    f"wh_spec:{fr.subj}"))
                        elif _strict:
                            # Hiç dayanak yoksa uydurma: soruyu geri verip öğrenmeye açık ol.
                            _miss = f"Bunu henüz bilmiyorum: “{raw.strip()}” — bana bir cümleyle öğretir misin?"
                            bids.append(CorticalBid("hippocampal_concept", 0.72, 0.75, _miss,
                                                    f"wh_unknown_strict:{fr.subj}"))'''
assert old in s
s = s.replace(old, new, 1)

# 2d) yardımcı: soru sözcüğü karşılandı mı
old = '''    def _answer_preference(self, raw, speaker="insan"):'''
new = '''    def _wh_covered(self, hits, wh: str, raw: str) -> bool:
        """Bilinen bağlar sorunun İSTEDİĞİ şeyi karşılıyor mu? ('nasıl' yöntem, 'neden' sebep ister;
        'kedi huzura yol açar' mırlamayı açıklamaz.)"""
        want = {"nasıl": ("USED_FOR", "CAN_DO"), "neden": ("CAUSES", "USED_FOR"),
                "niye": ("CAUSES", "USED_FOR"), "niçin": ("CAUSES", "USED_FOR")}.get(wh or "", ())
        if not want:
            return True
        try:
            q = self.speculation._decompose(self, raw, None)
            focus = self.speculation._tokens(q.get("verb_phrase", "")) | self.speculation._tokens(q.get("pred_noun", ""))
        except Exception:
            focus = set()
        if not focus:
            return True
        for e, _p in hits:
            if e.rel in want and e.obj and (self.speculation._tokens(e.obj) & focus):
                return True
        return False

    def _answer_preference(self, raw, speaker="insan"):'''
assert old in s
s = s.replace(old, new, 1)

# 2e) konu belleği: her turda gerçekten bildiği konu hatırlanır
old = '''        self._remember_focus(raw)'''
new = '''        self._remember_focus(raw)'''
assert old in s
p.write_text(s, encoding="utf-8")
print("agi ok")

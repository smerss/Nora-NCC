import pathlib

# ---------------------------------------------------------------- 1) ÖĞRETİM: "ve" ile birleşik yüklem cümlesi bölünür
p = pathlib.Path("nora_lang.py")
s = p.read_text(encoding="utf-8")

old = '''        clauses = re.split(r"\\s+(?:ama|ancak|fakat)\\s+|;\\s*|\\.\\s+", text.strip())
        learned: list[RelationEdge] = []'''
new = '''        clauses = re.split(r"\\s+(?:ama|ancak|fakat)\\s+|;\\s*|\\.\\s+", text.strip())
        # "X ... ve Y ..." birleşik YÜKLEM cümlesi iki ayrı iddiadır: her iki yarı da kendi başına
        # yüklem taşıyorsa bölünür. Böylece bütün bir yan cümle, ilişki nesnesi olarak kaydedilmez
        # ('sadıktır ve insana keyif ver' gibi). 'Köpek ve kedi evcildir' ise bölünmez (özne listesi).
        _expanded: list[str] = []
        for cl in clauses:
            parts = re.split(r"\\s+ve\\s+", cl.strip())
            if len(parts) == 2 and all(parts):
                try:
                    _l = self.wernicke.parse(parts[0].strip())
                    _r = self.wernicke.parse(parts[1].strip())
                    _ok = (_l.kind == "teach_fact" and _r.kind == "teach_fact"
                           and getattr(_l, "obj", "") and getattr(_r, "obj", ""))
                except Exception:
                    _ok = False
                if _ok:
                    _expanded.extend([parts[0].strip(), parts[1].strip()])
                    continue
            _expanded.append(cl)
        clauses = _expanded
        learned: list[RelationEdge] = []'''
assert old in s
s = s.replace(old, new, 1)

# parantez içi yüklem denetimi BÜTÜN ilişkilerde geçerli olsun (CAN_DO dahil)
old2 = '''                if fr.rel in ("ISA", "HAS_PROP", "HAS_PART") and not self._teachable_predicate(fr.obj):
                    continue'''
new2 = '''                if not self._teachable_predicate(fr.obj):
                    continue'''
assert old2 in s
s = s.replace(old2, new2, 1)
p.write_text(s, encoding="utf-8")
print("lang ok")

# ---------------------------------------------------------------- 2) İNİSİYATİF: keşif paylaşımı Broca'ya
p = pathlib.Path("nora_initiative.py")
s = p.read_text(encoding="utf-8")

start = s.index("    def _c_share_discovery(self, agi, ctx: dict) -> str | None:")
end = s.index("    def _c_recall(self, agi, ctx: dict) -> str | None:")
body = '''    def _c_share_discovery(self, agi, ctx: dict) -> str | None:
        """Az önce kurulan bağı paylaşır. Cümle kalıbı YOK: öğrenilen önerme, sıfat-fiil (-DIğInI)
        ile bir zihin fiiline nesne olur ('köpeğin insana keyif verdiğini fark ettim')."""
        edge = ctx.get("edge")
        if edge is None:
            return None
        subj, rel, obj = edge
        subj, obj = (subj or "").strip(), (obj or "").strip()
        if not subj or not obj:
            return None
        clause = self._proposition_complement(subj, rel, obj)
        if not clause:
            return None
        verbs = ("fark et", "öğren", "anla", "gör")
        verb = verbs[self.py_rng.randrange(len(verbs))]
        try:
            if verb == "gör":
                out = SYN.realize_verb(VerbSpec(verb_root=verb, obj=clause, tense="PRES", person="1SG"))
            else:
                out = SYN.realize_verb(VerbSpec(verb_root=verb, obj=clause, tense="PAST", person="1SG"))
        except Exception:
            return None
        return _cap(out) if out else None

    def _proposition_complement(self, subj: str, rel: str, obj: str) -> str:
        """Bir (özne, ilişki, nesne) üçlüsünü '-DIğInI' yan cümleciğine çevirir (tamamen sentaks)."""
        try:
            if rel in ("ISA", "HAS_PROP", "HAS_PART", "LOCATED_AT", "OPPOSITE", "CAUSES") or " " not in obj:
                pre = ("bir",) if (rel == "ISA" and not obj.startswith("bir ")) else ()
                return SYN.nominalize_dik("ol", poss=1, case="ACC", subj=NP(subj, case="GEN"),
                                          advs=pre + (obj,))
            words = obj.split()
            root_word = words[-1]
            info = agi_morph(root_word)
            verb = info.get("stem") if info.get("is_verb") else ""
            if not verb:
                return SYN.nominalize_dik("ol", poss=1, case="ACC", subj=NP(subj, case="GEN"), advs=(obj,))
            prev = tuple(w for w in words[:-1] if w)
            return SYN.nominalize_dik(verb, poss=1, case="ACC", subj=NP(subj, case="GEN"), advs=prev)
        except Exception:
            return ""

'''
s = s[:start] + body + s[end:]

# morfoloji erişimi: modül düzeyinde tembel yardımcı
anchor = "MOVES = ("
helper = '''_MORPH = None


def agi_morph(word: str) -> dict:
    """Kelimenin çözümlemesi (sözdizimi motorunun morfoloji bilgisi); motor bir kez kurulur."""
    global _MORPH
    if _MORPH is None:
        from nora_lang import TurkishMorphology
        _MORPH = TurkishMorphology()
    return _MORPH.analyze_word(word)


'''
assert anchor in s
s = s.replace(anchor, helper + anchor, 1)
p.write_text(s, encoding="utf-8")
print("initiative ok")

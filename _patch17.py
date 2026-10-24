import pathlib

p = pathlib.Path("nora_initiative.py")
s = p.read_text(encoding="utf-8")

start = s.index("    def _proposition_complement(self, subj: str, rel: str, obj: str) -> str:")
end = s.index("    def _c_recall(self, agi, ctx: dict) -> str | None:")
body = '''    def _proposition_complement(self, subj: str, rel: str, obj: str) -> str:
        """Bir (özne, ilişki, nesne) üçlüsünü '-DIğInI' yan cümleciğine çevirir (tamamen sentaks).
        'köpeğin insana keyif verdiğini', 'kedinin tatlı olduğunu'."""
        from nora_lang import TurkishMorphology

        def _copular() -> str:
            pre = ("bir",) if (rel == "ISA" and not obj.startswith("bir ")) else ()
            return SYN.nominalize_dik("ol", poss=3, case="ACC", subj=NP(subj, case="GEN"),
                                      advs=pre + (obj,))

        words = obj.split()
        if not words:
            return ""
        last = words[-1]
        # Nesne çekimli bir fiille bitiyorsa: 'insana keyif ver' -> 'köpeğin insana keyif verdiğini'
        if rel in ("CAN_DO", "USED_FOR", "CAUSES") and last in TurkishMorphology.KNOWN_VERB_ROOTS:
            prev = tuple(w for w in words[:-1] if w)
            return SYN.nominalize_dik(last, poss=3, case="ACC", subj=NP(subj, case="GEN"), advs=prev)
        if rel in ("CAN_DO", "USED_FOR"):
            return ""                      # fiil tanınmadıysa bozuk cümle kurmaktansa susar
        try:
            return _copular()
        except Exception:
            return ""

'''
s = s[:start] + body + s[end:]
p.write_text(s, encoding="utf-8")
print("initiative ok")

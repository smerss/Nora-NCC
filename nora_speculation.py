#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_speculation.py — Bilinmeyen karşısında DÜŞÜNME korteksi (spekülasyon / hipotez).

Bir çocuk, kendisine hiç öğretilmemiş bir şey sorulduğunda susmaz; elindeki parçalarla bağ kurar.
Bu organ o düşünmeyi kurar. HAZIR CÜMLE YOKTUR: her cümle nora_broca'da çıplak köklerden
çekimlenir, topolojiyi nora_monologue.PlasticSchemaCortex seçer.

Adımlar
-------
1. SORUYU PARÇALA: soru sözcüğü, eyleyici, nesne, yer, ana fiil (ve gerekiyorsa ad + yardımcı fiil:
   'sıra ol', 'nefes al').
2. NE BİLİYORUM: sorudaki kavramların ağdaki GERÇEK bağlarından cümleler.
3. BAĞ KUR: (a) araç/ölçüt, (b) neden, (c) benzetme (kardeş kavramdan aktarma).
4. Eksik kalan bağı TAM OLARAK söyler ve onu sorar.
"""
from __future__ import annotations

import random
import re

from nora_broca import NP, TurkishSyntaxEngine as SYN, VerbSpec
from nora_lang import Proposition, tr_capitalize, tr_lower

_STOP = {"bir", "bu", "şu", "o", "her", "çok", "daha", "en", "ve", "da", "de", "ki", "mi", "mı",
         "mu", "mü", "için", "ile", "gibi", "kadar", "ama", "ancak", "fakat", "çünkü", "zaman",
         "biz", "siz", "onlar", "bana", "sana", "beni", "seni", "bunu", "çünkü",
         "nasıl", "neden", "niye", "nerede", "nereye", "nereden", "kim", "ne", "hangi", "kaç",
         "bazı", "bütün", "tüm", "birkaç", "hiç", "hep", "kendi", "şey", "şeyler",
         "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz", "on"}
_QUESTION_WORDS = ("nasıl", "neden", "niye", "nerede", "nereye", "nereden", "kim", "ne", "hangi", "kaç")
_AGENT_HEADS = {"insan", "kişi", "uzman", "öğrenci", "çocuk", "bilim insanı"}
_WH_REL = {"nasıl": ("USED_FOR", "CAN_DO"), "neden": ("CAUSES",), "niye": ("CAUSES",),
           "nerede": ("LOCATED_AT",), "kim": ("ISA",), "ne": ("ISA", "HAS_PROP")}


def _cap(txt: str) -> str:
    txt = (txt or "").strip()
    if not txt:
        return ""
    out = tr_capitalize(txt)
    return out if out.endswith((".", "?", "!")) else out + "."


class SpeculativeCortex:
    """Bilinmeyen soruya karşı düşünme: bilgi kırıntısı -> hipotez -> eksik -> kendi sorusu."""

    def __init__(self, seed: int = 0):
        self.last_gap = ""
        self.last_agent = ""
        self.steps = 0
        self.rng = random.Random(seed)
        self.last_trace: str = ""

    # ------------------------------------------------------------------ LEMMA
    @staticmethod
    def _lemma(agi, word: str) -> tuple[str, str, bool, bool]:
        """(lemma, hâl, çoğul, fiil_mi) — ağ bilgisi ölçüt: 'uzayı' -> ('uzay', 'ACC')."""
        g = agi.lang.graph
        morph = agi.lang.morph
        w = tr_lower(word)
        try:
            an = morph.analyze_word(w, g.lexicon)
        except Exception:
            an = {}
        if an.get("is_verb") and an.get("stem") in morph.KNOWN_VERB_ROOTS:
            return an["stem"], "NOM", False, True
        stem = an.get("stem") or w
        case = str(an.get("case", "NOM")).upper()
        plural = bool(an.get("plural"))
        # Yüzey biçim ağda BAĞ taşıyorsa ve kökü boşsa, sözcük olduğu gibi alınır ('gece' -> 'geç'+DAT değil).
        if stem != w and w in g._out_edges and not g.query(stem, None):
            return w, "NOM", plural, False
        # Biçimbilim kökü yüzeyden farklıysa ve ağda karşılığı varsa o kazanır ('köpekler' -> 'köpek').
        if stem != w and (stem in g.lexicon or g.query(stem, None)):
            return stem, case, plural, False
        # Ekli yüzey biçim, ağda BAĞI OLAN kısa kökten sonra gelir ('uzayı' < 'uzay').
        for cut in (1, 2, 3):
            cand = w[:-cut]
            if len(cand) >= 3 and g.query(cand, None) and not g.query(w, None):
                return cand, ("ACC" if cut == 1 and w[-1] in "ıiuü" else case), plural, False
        if stem in g.lexicon or g.query(stem, None):
            return stem, case, plural, False
        for cut in (1, 2, 3):
            cand = w[:-cut]
            if len(cand) >= 3 and (cand in g.lexicon or g.query(cand, None)):
                return cand, case if cut == 1 else "NOM", plural, False
        return stem, case, plural, False

    # ------------------------------------------------------------------ 1. PARÇALA
    def _decompose(self, agi, raw: str, fr=None) -> dict:
        g = agi.lang.graph
        low = tr_lower(raw)
        wh = ""
        for w in _QUESTION_WORDS:
            if re.search(rf"\b{w}\b", low):
                wh = w
                break
        tokens = [t for t in re.findall(r"[\wçğıöşü]+", low) if t not in _STOP]
        nouns: list[list] = []          # [lemma, case, plural, index]
        verb, verb_idx = "", -1
        converb = ""
        raw_lemma: dict[str, str] = {}
        for i, t in enumerate(tokens):
            if t.endswith("ken") and len(t) > 5:
                converb = t                # yürürken / uçarken: ulaç olduğu gibi saklanır
                continue
            lem, case, plural, is_v = self._lemma(agi, t)
            if is_v:
                if lem != "ol" or not verb:
                    verb, verb_idx = lem, i
                continue
            if lem in _STOP:
                continue
            raw_lemma[t] = lem
            nouns.append((lem, case, plural, i))
        if not verb:
            # 3. tekil geniş zaman gibi görünen, ad olmayan son sözcük fiil sayılır ('esner' -> 'esne').
            for _i, t in reversed(list(enumerate(tokens))):
                if len(t) < 4 or not t.endswith('r') or (t in g.lexicon and g.query(t, None)):
                    continue
                if t.endswith(('lar', 'ler')) and (t[:-3] in g.lexicon or g.query(t[:-3], None)):
                    continue
                try:
                    _an = agi.lang.morph.analyze_word(t, g.lexicon)
                except Exception:
                    _an = {}
                if _an.get('copula') or _an.get('plural'):
                    continue
                roots = []
                if t.endswith(('ar', 'er', 'ır', 'ir', 'ur', 'ür')):
                    roots.append(t[:-2])          # kal-ır, yap-ar
                cand1 = t[:-1]
                if cand1 and cand1[-1] in 'aeıioöuü':
                    roots.append(cand1)           # mırla-r, esne-r
                def _is_root(c: str) -> bool:
                    if not c or len(c) < 2:
                        return False
                    return (c in agi.lang.morph.KNOWN_VERB_ROOTS or g.query(c, None)
                            or c + "mak" in g.lexicon or c + "mek" in g.lexicon)
                known = next((c for c in roots if _is_root(c)), None)
                if known:
                    cand = known
                elif roots:
                    _r0 = roots[0]
                    _cluster = len(_r0) >= 2 and _r0[-1] not in 'aeıioöuü' and _r0[-2] not in 'aeıioöuü'
                    # Ünsüz kümesiyle biten gövde ('esn', 'sall', 'mırl') Türkçede kök değildir:
                    # bu durumda ünlüyle biten aday köktür ('esne', 'salla', 'mırla').
                    if _cluster and cand1 and cand1[-1] in 'aeıioöuü':
                        cand = cand1
                    else:
                        cand = _r0
                else:
                    cand = '' 
                if cand and len(cand) >= 2:
                    verb, verb_idx = cand, _i
                    nouns = [n for n in nouns if not (verb.startswith(n[0]) and len(n[0]) >= 2)]
                if verb:
                    break
        if not verb:
            # Sözlükte olmayan ama fiil gibi görünen sözcük ('mırlar' -> 'mırla'): gövde tahmini.
            for _i, t in reversed(list(enumerate(tokens))):
                if len(t) >= 5 and t.endswith("r"):
                    if t.endswith(("lar", "ler")) and (t[:-3] in g.lexicon or g.query(t[:-3], None)):
                        continue                     # 'ağaçlar' bir addır, fiil değil
                    cand = t[:-1]
                    if cand.endswith(("la", "le", "da", "de", "ta", "te")):
                        verb, verb_idx = cand, _i
                        nouns = [n for n in nouns if not (verb.startswith(n[0]) and len(n[0]) >= 2)]
                        break

        if verb_idx >= 0:
            nouns = [n for n in nouns if n[3] != verb_idx]
        agent, patient, place, pred_noun = "", "", "", ""
        # Koşaçlı sözcük ('mavidir', 'tuzludur') YÜKLEMDİR; nesne sanılmaz.
        for t in reversed(tokens):
            lem_t, case_t, _pl_t, _v_t = self._lemma(agi, t)
            try:
                _an = agi.lang.morph.analyze_word(t, g.lexicon)
            except Exception:
                _an = {}
            if _an.get("copula") and lem_t not in _STOP:
                pred_noun = lem_t
                nouns = [n for n in nouns if n[0] != lem_t]
                break
        agent_plural = False
        used_idx: set[int] = set()
        # 'bilim insanları' -> eyleyici 'bilim insanı'
        for j in range(len(nouns) - 1):
            if nouns[j + 1][0] in _AGENT_HEADS:
                agent = f"{nouns[j][0]} {nouns[j + 1][0]}ı"
                agent_plural = nouns[j + 1][2]
                used_idx |= {nouns[j][3], nouns[j + 1][3]}
                break
        if not agent:
            for lem, case, plural, idx in nouns:
                if case == "NOM" and idx != verb_idx:
                    agent, agent_plural = lem, plural
                    used_idx.add(idx)
                    break
        for lem, case, _p, idx in nouns:
            if idx in used_idx:
                continue
            if case == "LOC" and not place:
                place = lem
                used_idx.add(idx)
                continue
            if case == "ACC" and not patient:
                patient = lem
                used_idx.add(idx)
            elif case == "LOC" and not place:
                place = lem
                used_idx.add(idx)
        # Ad + yardımcı fiil ('sıra ol', 'nefes al'): fiilden hemen önceki ad yüklem adıdır.
        if verb_idx > 0:
            for lem, case, _p, idx in nouns:
                if idx in used_idx:
                    continue
                if idx >= verb_idx:
                    continue
                if idx == verb_idx - 1 and not pred_noun:
                    if verb and (verb.startswith(lem) or lem.startswith(verb)):
                        continue                 # 'mır mırlar' gibi kök yinelemesi
                    # Ağda karşılığı olmayan sözcük kırpılmış köküyle değil, duyulduğu gibi saklanır.
                    surf = tokens[idx] if idx < len(tokens) else lem
                    pred_noun = lem if g.query(lem, None) else surf
                    used_idx.add(idx)
        if not patient:
            for lem, case, _p, idx in nouns:
                if idx not in used_idx and case == "NOM":
                    patient = lem
                    used_idx.add(idx)
                    break
        verb_phrase = (pred_noun + " " + verb).strip()
        concepts = [c for c in (agent, patient, place, pred_noun) if c]
        return {"raw": raw, "converb": converb, "wh": wh or str(getattr(fr, "wh", "") or "ne"),
                "agent": agent, "agent_plural": agent_plural, "patient": patient, "place": place,
                "verb": verb, "pred_noun": pred_noun, "verb_phrase": verb_phrase,
                "concepts": list(dict.fromkeys(concepts)), "tokens": tokens}

    # ------------------------------------------------------------------ 2. NE BİLİYORUM
    def _known_clauses(self, agi, q: dict, limit: int = 2, background: bool = False) -> list[str]:
        g = agi.lang.graph
        broca = agi.lang.broca
        out: list[str] = []
        order = {"HAS_PROP": 0, "CAN_DO": 1, "USED_FOR": 2, "CAUSES": 3, "HAS_PART": 4, "ISA": 5}
        if background:
            order = {"CAN_DO": 0, "HAS_PART": 1, "HAS_PROP": 2, "USED_FOR": 3, "CAUSES": 4, "ISA": 6}
        if q.get("wh") in ("neden", "niye", "niçin"):
            order = {"CAUSES": 0, "USED_FOR": 1, "CAN_DO": 2, "HAS_PROP": 3, "HAS_PART": 4, "ISA": 5}
        _asked = tr_lower(q.get("raw", ""))
        # 'neden/nasıl' sorusunda yalnız SORULAN varlığın bağları sayılır; yan kavramlar (yer, araç) gürültüdür.
        _concepts = ([q["agent"]] if (q.get("wh") in ("neden", "niye", "niçin", "nasıl") and q.get("agent"))
                     else q["concepts"]) if not background else q["concepts"]
        for c in _concepts:
            hits = [(e, p) for e, p in g.query(c, None) if not p]
            hits = [(e, p) for e, p in hits
                    if e.rel in order or (e.rel == "OPPOSITE" and any(w in _asked for w in ("zıt", "karşıt", "ters")))]
            hits.sort(key=lambda h: (order.get(h[0].rel, 9), -float(h[0].weight)))
            q_tokens = self._tokens(q.get("verb_phrase", "")) | self._tokens(q.get("patient", ""))
            _strict = (not background) and q.get("wh") in ("neden", "niye", "niçin", "nasıl")
            for e, _p in hits:
                if _strict and e.rel in ("ISA", "HAS_PROP") and q_tokens and not (self._tokens(e.obj) & q_tokens):
                    continue
                if e.rel == "CAN_DO":
                    if " " in (e.obj or "") and e.obj.split()[-1] in ("verir", "ver"):
                        continue
                    # Konuyla ilgisiz beceri cümlesi bilgi değil gürültüdür.
                    if q_tokens and not (self._tokens(e.obj) & q_tokens):
                        continue
                try:
                    txt = broca.realize_prop(Proposition(subj=c, rel=e.rel, obj=e.obj))
                except Exception:
                    continue
                if txt and tr_lower(txt) not in [tr_lower(t) for t in out]:
                    out.append(txt)
                    break
            if len(out) >= limit:
                break
        return out

    # ------------------------------------------------------------------ 3. BAĞ KUR
    @staticmethod
    def _verb_in(txt: str) -> str:
        """İfadenin sonundaki fiil kökü ('suda nefes almak' -> 'al')."""
        words = re.findall(r"[\wçğıöşü]+", tr_lower(txt))
        if not words:
            return ""
        last = words[-1]
        if last.endswith(("mak", "mek")):
            return last[:-3]
        return last

    @staticmethod
    def _tokens2(txt: str) -> set[str]:
        """Odak sözcükleri: 2 harfli fiiller de sayılır ('uç', 'at', 'ol')."""
        return {w[:4] for w in re.findall(r"[\wçğıöşü]+", tr_lower(txt)) if len(w) >= 2}

    @staticmethod
    def _tokens(txt: str) -> set[str]:
        # Ekler kırpılır: 'uzayı' ~ 'uzay', 'gözlemlemek' ~ 'gözlem'.
        return {w[:4] for w in re.findall(r"[\wçğıöşü]+", tr_lower(txt)) if len(w) >= 3}

    def _hypothesis(self, agi, q: dict) -> tuple[str, str]:
        """(cümle, iz). Yalnızca ağdaki GERÇEK bir bağdan türetir; uydurmaz."""
        g = agi.lang.graph
        rels = _WH_REL.get(q["wh"], ("ISA", "HAS_PROP"))
        agent, patient, verb = q["agent"], q["patient"], q["verb"]
        person = "3PL" if q["agent_plural"] else "3SG"
        pat_tokens = self._tokens(patient) | self._tokens(q["verb"])

        # (a) ARAÇ: sorulan NESNEYLE örtüşen USED_FOR bağının öznesi ölçüttür.
        if "USED_FOR" in rels and patient and verb:
            for src, edges in list(g._out_edges.items()):
                if src in (agent, patient, *q["concepts"]):
                    continue
                for e in edges:
                    if e.rel != "USED_FOR" or not e.obj:
                        continue
                    if not (self._tokens(e.obj) & self._tokens(patient)):
                        continue
                    try:
                        clause = SYN.realize_verb(VerbSpec(
                            verb_root=verb, subj=NP(agent) if agent and not q["agent_plural"] else None,
                            obj=NP(patient, case="ACC"),
                            ins=NP(src, case="INS"), tense="ABIL_AOR", person=person))
                        if clause:
                            return clause, f"araç:{src}->{e.obj}"
                    except Exception:
                        continue

        # (b) NEDEN: kavramın kendi CAUSES bağı varsa bu tahmin değil, bilgidir.
        if "CAUSES" in rels:
            _specific = [q["pred_noun"], q["verb_phrase"]]
            if q["verb"] and q["patient"]:
                _specific.append(f"{q['patient']} {q['verb']}")
            for _vp in list(_specific):
                if _vp and " " in _vp:
                    _specific.append(_vp + ("mak" if _vp[-1] in "aıou" else "mek"))
            _generic = [agent] if not q["verb"] else [agent, patient]
            _light = q["verb"] in ("", "ol", "et", "yap", "bulun", "bul")
            _srcs = _specific + (_generic if _light else [])
            for c in dict.fromkeys(x for x in _srcs if x):
                if not c:
                    continue
                for e, p in g.query(c, ("CAUSES",)):
                    if p or not e.obj:
                        continue
                    try:
                        return agi.lang.broca.realize_prop(Proposition(subj=c, rel="CAUSES", obj=e.obj)), \
                            f"neden:{c}->{e.obj}"
                    except Exception:
                        continue

        # (b1) PARÇA -> İŞLEV: organın işlevi, o organa sahip varlığa aktarılır.
        if agent and verb:
            for e_part, _pp in g.query(agent, ("HAS_PART",)):
                part = e_part.obj
                if not part or " " in part:
                    continue
                for rel in ("CAN_DO", "USED_FOR"):
                    for e2, p2 in g.query(part, (rel,)):
                        if p2 or not e2.obj:
                            continue
                        if not (self._tokens(e2.obj) & pat_tokens) and self._verb_in(e2.obj) != verb:
                            continue
                        try:
                            _own = [w for w in q["verb_phrase"].split() if w != verb and w not in _STOP]
                            _obj = patient or (" ".join(_own) if _own else "")
                            clause = SYN.realize_verb(VerbSpec(
                                verb_root=verb, subj=NP(agent),
                                obj=NP(_obj, case="NOM") if _obj else None,
                                ins=NP(part, case="INS", poss=3, plural=True), tense="AOR", person="3SG"))
                            if clause:
                                return clause, f"parça:{agent}.{part}->{e2.obj}"
                        except Exception:
                            continue

        # (b2) KALITIM: üst kavramda olan genel kural, alt kavram için de geçerlidir.
        if agent:
            for rel in rels:
                for e, path in g.query(agent, (rel,)):
                    if not path or not e.obj:
                        continue
                    if not (self._tokens(e.obj) & pat_tokens):
                        continue
                    try:
                        clause = agi.lang.broca.realize_prop(Proposition(subj=agent, rel=rel, obj=e.obj))
                        if clause:
                            parent = path[0].obj
                            return clause, f"kalıtım:{agent}<{parent}.{rel}={e.obj}"
                    except Exception:
                        continue

        # (c) BENZETME: aynı üst kavramın altındaki kardeş, o bağı taşıyorsa aktarılır.
        if agent and " " not in agent:
            parents = [e.obj for e, p in g.query(agent, ("ISA",)) if not p]
            for par in parents:
                sibs = [src for src, edges in g._out_edges.items()
                        if src != agent and any(e.rel == "ISA" and e.obj == par for e in edges)]
                for sib in sibs[:12]:
                    for rel in rels:
                        for e, p in g.query(sib, (rel,)):
                            if p or not e.obj or rel == "ISA":
                                continue
                            if not (self._tokens(e.obj) & pat_tokens):
                                continue
                            try:
                                clause = SYN.realize_verb(VerbSpec(
                                    verb_root=verb or "yap", subj=NP(agent),
                                    obj=NP(patient, case="ACC") if patient else None,
                                    tense="ABIL_AOR", person=person))
                                return clause, f"benzetme:{agent}~{sib}.{rel}={e.obj}"
                            except Exception:
                                continue
        return "", ""

    def _part_function(self, agi, q: dict) -> str:
        """Soruyu tam yanıtlamasa da kurulabilen bağ: 'kuş kanadıyla uçar' (kanat uçmak içindir)."""
        g = agi.lang.graph
        agent = q["agent"]
        if not agent or " " in agent:
            return ""
        conv = self._tokens(q.get("converb", "") or "")
        for e_part, _p in g.query(agent, ("HAS_PART",)):
            part = e_part.obj
            if not part or " " in part:
                continue
            for rel in ("CAN_DO", "USED_FOR"):
                for e2, p2 in g.query(part, (rel,)):
                    if p2 or not e2.obj:
                        continue
                    obj = e2.obj
                    last = obj.split()[-1]
                    if last.endswith(("mak", "mek")) and last[:-3] in agi.lang.morph.KNOWN_VERB_ROOTS:
                        root = last[:-3]
                    elif last in agi.lang.morph.KNOWN_VERB_ROOTS:
                        root = last
                    else:
                        continue
                    if conv and not (self._tokens(root) & conv):
                        continue
                    try:
                        out = SYN.realize_verb(VerbSpec(verb_root=root, subj=NP(agent),
                                                        ins=NP(part, case="INS", poss=3, plural=True),
                                                        tense="AOR", person="3SG"))
                        return _cap(out).rstrip(".") if out else ""
                    except Exception:
                        continue
        return ""

    # ------------------------------------------------------------------ 4. EKSİK + SORU
    def _complement(self, agi, q: dict) -> str:
        """'-DIğInI' yan cümleciği: 'karıncanın sıra ol' + 'duğunu'."""
        try:
            advs = []
            if q.get("converb"):
                advs.append(q["converb"])            # 'uçarken'
            if q.get("place"):
                advs.append(SYN.realize_np(NP(q["place"], case="LOC")))   # 'karanlıkta'
            if q["wh"] in ("nasıl", "neden", "nerede"):
                advs.append(q["wh"])                 # 'nasıl' en sonda: 'karanlıkta nasıl uçtuğunu'
            advs = tuple(a for a in advs if a)
            if q["pred_noun"] and q["verb"]:
                return SYN.nominalize_dik(q["verb"], poss=3, case="ACC",
                                          subj=NP(q["agent"], case="GEN") if q["agent"] else None,
                                          advs=advs + (q["pred_noun"],))
            if not q["verb"] and (q["pred_noun"] or q["patient"]):
                return SYN.nominalize_dik("ol", poss=3, case="ACC",
                                          subj=NP(q["agent"], case="GEN") if q["agent"] else None,
                                          advs=advs, obj=NP(q["pred_noun"] or q["patient"], case="NOM"))
            if q["verb"]:
                return SYN.nominalize_dik(q["verb"], poss=3, case="ACC",
                                          subj=NP(q["agent"], case="GEN") if q["agent"] else None,
                                          advs=advs,
                                          obj=NP(q["patient"], case="ACC") if q["patient"] else None)
        except Exception:
            return ""
        return ""

    def _lack_clause(self, agi, q: dict) -> str:
        comp = self._complement(agi, q)
        try:
            if comp:
                return _cap(SYN.realize_verb(VerbSpec(verb_root="bil", obj=comp, neg=True,
                                                      tense="PRES", person="1SG")))
            concept = q["agent"] or q["patient"]
            if concept:
                (g := agi.lang.graph)
                if not g.query(concept, None) and concept not in g.lexicon:
                    return _cap(SYN.realize_verb(VerbSpec(verb_root="tanı", obj=NP(concept, case="ACC"),
                                                          neg=True, tense="PRES", person="1SG")))
                noun = "sebep" if q.get("wh") in ("neden", "niye", "niçin") else "yol"
                return _cap(SYN.realize_verb(VerbSpec(verb_root="bil", obj=NP(noun, poss=3, case="ACC"),
                                                      neg=True, tense="PRES", person="1SG")))
        except Exception:
            pass
        return ""

    def _curiosity_question(self, agi, q: dict) -> str:
        comp = self._complement(agi, q)
        try:
            if comp:
                return tr_capitalize(SYN.realize_verb(VerbSpec(
                    verb_root="anlat", dat=NP("ben", case="DAT"), obj=comp,
                    question_yn=True, tense="AOR", person="2SG"))) + "?"
            concept = q["agent"] or q["patient"]
            if concept:
                return tr_capitalize(SYN.realize_verb(VerbSpec(
                    verb_root="anlat", dat=NP("ben", case="DAT"), obj=NP(concept, case="ACC"),
                    question_yn=True, tense="AOR", person="2SG"))) + "?"
        except Exception:
            pass
        return ""

    # ------------------------------------------------------------------ 5. BİRLEŞTİR
    def deliberate(self, agi, raw: str, fr=None) -> str:
        q = self._decompose(agi, raw, fr)
        self.last_agent = q.get("agent", "")
        if not q["concepts"] and not q["verb"]:
            return ""
        known = [_cap(t) for t in self._known_clauses(agi, q, limit=2)]
        hypo, trace = self._hypothesis(agi, q)
        if not known:
            known = [_cap(t) for t in self._known_clauses(agi, q, limit=1, background=True)]

        parts: list[str] = []
        seen: set[str] = set()
        for txt in known:
            key = tr_lower(txt)
            if key not in seen:
                parts.append(txt)
                seen.add(key)
        if hypo:
            hedged = hypo
            if trace == "" or trace.startswith(("parça:", "kalıtım:", "benzetme:")):
                # Doğrudan öğrenilmemiş, çıkarımla kurulmuş bağ: alçakgönüllü sunulur.
                markers = ("sanırım", "galiba", "bence", "tahminimce")
                marker = markers[int(self.steps) % len(markers)] if hasattr(self, "steps") else markers[0]
                try:
                    self.steps += 1
                except Exception:
                    pass
                hedged = marker + " " + tr_lower(hypo[0]) + hypo[1:]
            try:
                hedged = agi.schema_cortex.transform_clauses(hedged, "", "", "S1_NEDEN_SONUC")
            except Exception:
                hedged = hedged
            if tr_lower(hedged) not in seen:
                parts.append(hedged if hedged.endswith((".", "?", "!")) else hedged + ".")
        else:
            part_fn = self._part_function(agi, q)
            if part_fn:
                parts.append(part_fn if part_fn.endswith((".", "?", "!")) else part_fn + ".")
            lack = self._lack_clause(agi, q)
            if lack and tr_lower(lack) not in seen:
                parts.append(lack)
        gap_parts: list[str] = []
        if not hypo:
            lack = self._lack_clause(agi, q)
            question = self._curiosity_question(agi, q)
            for txt in (lack, question):
                if not txt:
                    continue
                gap_parts.append(txt)
                if all(tr_lower(txt) != tr_lower(pp) for pp in parts):
                    parts.append(txt)
        self.last_gap = " ".join(gap_parts)
        self.last_trace = (f"parçala: eyleyici={q['agent']} nesne={q['patient']} fiil={q['verb_phrase']} "
                           f"wh={q['wh']} | bilgi={len(known)} | {trace or 'hipotez yok'}")
        return " ".join(parts)

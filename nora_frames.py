#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_frames.py — CÜMLE YAPISI KORTEKSİ (yapısal dilbilgisi + valans + anlam soruları).

Neden var: Nora tek tek sözcük bilse de cümlenin İSKELETİNİ bilmiyordu. "Selam nasıl
denilir?" sorusunu duyunca selamlaşma yoluna giriyor ("Selam, huzurluyum") ya da soruyu
düzyazı gibi alıntılıyordu. Oysa insan, cümleyi *yapı* olarak okur:

    [AD: selam]  [SORU BELİRTECİ: nasıl]  [FİİL: de·PASSIVE·AOR·3SG]

ve "nasıl + <söyleme fiili> edilgen" yapısının GENEL anlamını bilir: "X nasıl söylenir?"
Bu yapı hangi sözcükle doldurulursa doldurulsun aynı soruyu sorar. Bu modül:

1. Tokens → morpholojik ROLLER (özne, nesne, dolaylı tümleç, zarf, soru, yüklem).
2. Valans öğrenimi: her fiilin hangi hâl ekli nesneyle kullanıldığını GÖZLEMDEN sayar
   ('kapıyı açtım' → aç: ACC; 'eve gittim' → git: DAT). Kalıp yok, istatistik var.
3. Yapı öğrenimi: cümlelerin iskelet imzaları sayılır ("AD:ACC + FİİL:PAST:1SG" gibi).
4. Anlam-sorusu yapıları: soru belirteci + yüklem sınıfı → niyet (NASIL_SÖYLENİR,
   NASIL_YAPILIR, NE_DEMEK, NEDİR). Yüklem sınıfı sözlükten gelir (söyleme fiilleri,
   yapma fiilleri), cümle kalıbı koddan değil.
5. SORULAN SÖZCÜK (mention) ayrımı: "Merhaba!" sözcüğü KULLANIR, "Merhaba nasıl
   denilir?" sözcükten BAHSEDER. Bahsedilen sözcük bilinmiyorsa dürüst boşluk + araştırma.
"""
from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from nora_lang import tr_capitalize, tr_lower

# ---------------------------------------------------------------- SÖZCÜK SINIFLARI
# Bunlar CÜMLE değil, sözcük bilgisidir: hangi fiil "söyleme", hangisi "yapma" fiilidir.
SAY_VERBS = {"de", "söyle", "yaz", "oku", "seslen", "adlandır", "an", "telaffuz et"}
DO_VERBS = {"yap", "et", "aç", "kapa", "sür", "kullan", "pişir", "kur", "tamir et", "çalıştır",
            "başlat", "bitir", "temizle", "yıka", "kes", "boya", "çiz", "dik", "bağla", "çöz"}
MEAN_VERBS = {"de", "demek", "anlam", "anlat", "ifade et", "karşıla"}

WH_WORDS = {"nasıl", "nasıldır", "ne", "nedir", "neden", "niçin", "niye", "kim", "kime",
            "kimi", "kimin", "nerede", "nereye", "nereden", "kaç", "hangi", "ne zaman"}
Q_PARTICLES = {"mi", "mı", "mu", "mü", "miyim", "mısın", "musun", "müsün",
               "miyiz", "mıyız", "musunuz", "misiniz"}

FINITE_TENSES = {"PAST", "PRES", "FUT", "AOR", "EVID", "PROG", "IMP", "COND", "OPT", "NEC",
                 "PRES_HIK", "FUT_HIK", "AOR_HIK", "EVID_HIK", "NEC_HIK", "PROG_HIK"}

# Yalın hâl (NOM) alan ADLAR özne olabilir; ama bunlar zarf/ilgeç gibi davranır.
# Şahıs adılları: sorulan KAVRAM olamazlar ('Sence bilinç nedir?' -> 'sen' değil 'bilinç').
# Yalın hâlleri hedef seçilmez; '-e/-i' almış hâlleri ('Bana ne dedin?') seçilebilir.
PRONOUN_STEMS = {"ben", "sen", "o", "biz", "siz", "onlar", "kendi", "nere", "ne", "kim"}

NON_SUBJECT = {"kendi", "kendimi", "kendini", "kendimizi", "kendinizi", "kendilerini",
               "kendime", "kendine", "çok", "az", "biraz", "pek", "daha", "en", "hiç", "hep", "yine", "şimdi",
               "bugün", "dün", "yarın", "sabah", "akşam", "gece", "öğlen", "sonra", "önce",
               "burada", "orada", "şurada", "birlikte", "yalnız", "tek", "her", "bazı",
               "seninle", "benimle", "onunla", "bizimle", "sizinle", "lütfen", "artık"}


@dataclass
class Token:
    word: str
    role: str = ""            # WH | PARTICLE | VERB | NOUN | ADJ | ADV | PRON | CONJ | OTHER
    lemma: str = ""
    case: str = ""
    tense: str = ""
    person: str = ""
    voice: str = ""
    neg: bool = False
    modal: str = ""
    poss: int = 0
    plural: bool = False
    copula: bool = False
    info: dict = field(default_factory=dict)

    @property
    def tag(self) -> str:
        """İskelet imzası için tek etiket: içerik sözcükleri rol+hâl, yüklem zaman+kişi."""
        if self.role == "VERB":
            return f"FİİL:{self.tense or '?'}:{self.person or '?'}"
        if self.role == "WH":
            return f"SORU:{self.word}"
        if self.role == "PARTICLE":
            return "SORU-EKİ"
        if self.role == "NOUN":
            return f"AD:{self.case or 'NOM'}"
        if self.role == "ADJ":
            return "SIFAT"
        if self.role == "ADV":
            return "BELİRTEÇ"
        return self.role or "DİĞER"


@dataclass
class Clause:
    text: str
    tokens: list[Token]
    verb: str = ""
    verb_tense: str = ""
    person: str = ""
    voice: str = ""
    neg: bool = False
    modal: str = ""
    wh: str = ""
    subject: str = ""
    obj: str = ""
    obj_case: str = ""
    dative: str = ""
    locative: str = ""
    ablative: str = ""
    is_question: bool = False
    target: str = ""          # hakkında konuşulan/sorulan sözcük (mention)
    copula: bool = False
    copula_stem: str = ""

    @property
    def signature(self) -> tuple[str, ...]:
        """İçerik sözcükleri yerine ROLLERİYLE cümlenin iskeleti."""
        return tuple(t.tag for t in self.tokens)

    def roles_tr(self) -> str:
        names = {"NOUN": "ad", "VERB": "yüklem", "WH": "soru", "ADV": "belirteç",
                 "ADJ": "sıfat", "PARTICLE": "soru eki", "PRON": "adıl", "CONJ": "bağlaç"}
        out = []
        for t in self.tokens:
            n = names.get(t.role, t.role.lower() or "söz")
            extra = ""
            if t.role == "NOUN" and t.case and t.case != "NOM":
                extra = {"ACC": "(belirtme)", "DAT": "(yönelme)", "LOC": "(bulunma)",
                         "ABL": "(ayrılma)", "GEN": "(tamlayan)", "INS": "(araç)"}.get(t.case, "")
            if t.role == "VERB" and t.tense:
                extra = f"({t.tense.lower()},{t.person})"
            out.append(f"{t.word} → {n}{extra}")
        return " | ".join(out)


class ClauseParser:
    """Sözcükleri biçimbilimden ROLLERE çevirir; hiçbir sözcük dizisi koda gömülmez."""

    def __init__(self, morph, lexicon=None):
        self.morph = morph
        self.lexicon = lexicon
        try:
            from nora_knowledge import (COMMON_ADJECTIVES, COMMON_NOUNS, CORE_TURKISH_VERBS,
                                        PROPER_NOUNS, STATE_ADJECTIVES, WORD_CLASSES)
            self.vocab = (COMMON_NOUNS | COMMON_ADJECTIVES | STATE_ADJECTIVES | PROPER_NOUNS
                          | CORE_TURKISH_VERBS | set(WORD_CLASSES))
        except Exception:
            self.vocab = set()

    def _an(self, w: str) -> dict:
        """Biçimbilim + sözlük sezgisi.

        Nora'nın kavram ağı küçükken kök soyma savrulabilir ('zemheri' -> 'zemher'+ACC,
        'turşu' -> 'turş'+ACC). Hal eki yalnızca soyma SÖZLÜKTE doğrulanıyorsa kabul edilir.
        """
        try:
            an = dict(self.morph.analyze_word(w, self.lexicon) or {})
        except Exception:
            return {}
        if an.get("case") in ("ACC", "DAT", "LOC", "ABL", "GEN", "INS") and not an.get("is_verb"):
            stem = an.get("stem") or ""
            known = stem in self.vocab or bool(self.lexicon and stem in self.lexicon)
            if not known and not an.get("vowel_drop") and not an.get("poss"):
                an["case"] = "NOM"
                an["stem"] = w
        return an

    @staticmethod
    def _as_wh(w: str) -> str:
        """Soru belirtecinin çekimli/koşaçlı biçimini köke indirger: 'kaçtır'->'kaç',
        'nedir'->'ne', 'kimdir'->'kim'. Yalnız soru sözcükleri için (tahmin yok)."""
        _CANON = {"nedir": "ne", "nasıldır": "nasıl", "kimdir": "kim",
                  "nedendir": "neden", "niçindir": "niçin", "kaçtır": "kaç"}
        if w in _CANON:
            return _CANON[w]
        if w in WH_WORDS:
            return w
        for base in ("nasıl", "neden", "niçin", "niye", "kim", "nere", "kaç", "hangi", "ne"):
            if not w.startswith(base):
                continue
            rest = w[len(base):]
            if rest and (rest in ("dir", "dır", "dur", "dür", "tir", "tır", "tur", "tür")
                         or rest[:1] in ("y", "d", "t", "n", "s")
                         or rest in ("si", "sı", "ye", "ya", "de", "da", "den", "dan",
                                     "nin", "nın", "yi", "yı", "i", "ı", "e", "a")):
                return "ne" if base == "ne" else base
        return ""

    def parse(self, text: str) -> Clause:
        raw_words = [w for w in re.split(r"[\s,;:.!?()\[\]\"“”]+", tr_lower(text)) if w]
        cl = Clause(text=text, tokens=[])
        cl.is_question = ("?" in text) or any(w in Q_PARTICLES for w in raw_words)
        verb_idx = -1
        # 1) Yüklemi bul: en sondaki bitmiş fiil; yoksa koşaç.
        for i, w in enumerate(raw_words):
            an = self._an(w)
            if an.get("is_verb") and str(an.get("tense") or "") in FINITE_TENSES:
                verb_idx = i
        cop_idx = -1
        for i, w in enumerate(raw_words):
            an = self._an(w)
            if an.get("copula"):
                cop_idx = i
                cl.copula = True
                cl.copula_stem = an.get("stem") or w
                if verb_idx < 0 or i >= verb_idx:
                    pass
        cl.verb_tense = ""
        # 2) Her sözcüğe rol ver.
        for i, w in enumerate(raw_words):
            an = self._an(w)
            t = Token(word=w, info=an, lemma=an.get("stem") or w,
                      case=an.get("case") or "", tense=an.get("tense") or "",
                      person=an.get("person") or "", voice=an.get("voice") or "",
                      neg=bool(an.get("neg")), modal=an.get("modal") or "",
                      poss=int(an.get("poss") or 0), plural=bool(an.get("plural")),
                      copula=bool(an.get("copula")))
            _wh = self._as_wh(w)
            if _wh:
                t.role = "WH"
                cl.wh = cl.wh or _wh
            elif w in Q_PARTICLES or w.startswith(("mı", "mi", "mu", "mü")) and len(w) <= 7:
                t.role = "PARTICLE"
            elif i == verb_idx:
                t.role = "VERB"
                cl.verb = t.lemma
                cl.verb_tense = t.tense
                cl.person = t.person
                cl.voice = t.voice
                cl.neg = t.neg
                cl.modal = t.modal
            elif w in NON_SUBJECT:
                t.role = "ADV"
            elif an.get("copula") or (cl.copula and i == cop_idx and i != verb_idx):
                t.role = "ADJ" if not an.get("case") else "NOUN"
            elif an.get("is_verb"):
                t.role = "ADV" if (an.get("converb") or an.get("participle")) else "VERB"
            elif an.get("case") == "NOM" and not an.get("poss") and len(w) > 1:
                t.role = "NOUN"
            elif an.get("case") in ("ACC", "DAT", "LOC", "ABL", "GEN", "INS"):
                t.role = "NOUN"
            else:
                t.role = "ADJ"
            cl.tokens.append(t)
        # 3) Özne / nesne / tümleçler (yüklemden ÖNCEKİ adlar).
        limit = verb_idx if verb_idx >= 0 else len(cl.tokens)
        for t in cl.tokens[:limit]:
            if t.role != "NOUN" or t.word in WH_WORDS or t.lemma in WH_WORDS:
                continue
            if t.case in ("ACC",) and not cl.obj:
                cl.obj, cl.obj_case = t.lemma, "ACC"
            elif t.case == "DAT" and not cl.dative:
                cl.dative = t.lemma
            elif t.case == "LOC" and not cl.locative:
                cl.locative = t.lemma
            elif t.case == "ABL" and not cl.ablative:
                cl.ablative = t.lemma
            elif t.case == "NOM" and not cl.subject:
                cl.subject = t.lemma
        if not cl.obj:
            for t in cl.tokens[:limit]:
                if t.role == "NOUN" and t.case == "NOM" and t.lemma != cl.subject \
                        and t.lemma not in WH_WORDS:
                    cl.obj, cl.obj_case = t.lemma, "NOM"
                    break
        if cl.copula and not cl.subject:
            for t in cl.tokens[:limit] or cl.tokens:
                if t.role == "NOUN" and t.case == "NOM":
                    cl.subject = t.lemma
                    break
        # 4) BAHSEDİLEN SÖZCÜK: soru belirtecinin hemen solundaki içerik sözcüğü;
        #    yoksa nesne; yoksa koşaçlı yüklemin öznesi.
        cl.target = self._target_of(cl)
        return cl

    def _known_stem(self, lem: str) -> bool:
        """Sözcük Nora'nın sözlüğünde mi? (Bileşik ad kurarken tek başına bilinen sözcüğü bölmemek için.)"""
        if not lem:
            return False
        if getattr(self, "lexicon", None) and lem in self.lexicon:
            return True
        return lem in getattr(self, "vocab", set())

    def _np_at(self, toks: list, idx: int) -> str:
        """`idx`'ten başlayan BAŞ-SONDALI ad öbeğini döndürür: 'kara delik', 'yapay zekâ'.

        Yalnız BAŞ (son) sözcük bilinmiyorsa birleştirilir: 'bilinç' gibi bilinen bir ada
        dokunulmaz, çünkü hakkında zaten bilgi var. Böylece 'Kara delik nedir?' sorusu
        'kar' (kar yağışı) diye yanlış çözülmez; Nora bilmediği TERİMİ sorar.
        """
        def _name(t) -> str:
            """Kökü sözlükle doğrulanmamışsa duyulan biçim kullanılır ('pusul' değil 'pusula')."""
            if (t.info or {}).get("unverified_stem"):
                return t.word
            return t.lemma

        if idx + 1 >= len(toks):
            return _name(toks[idx])
        a, b = toks[idx], toks[idx + 1]
        if not (a.role in ("NOUN", "ADJ") and b.role in ("NOUN", "ADJ")):
            return _name(a)
        if a.case not in ("", "NOM") or b.case not in ("", "NOM") or a.poss or b.poss:
            return _name(a)
        if len(a.lemma) < 2 or len(b.lemma) < 2 or self._known_stem(b.lemma):
            return _name(a)
        nxt = toks[idx + 2].role if idx + 2 < len(toks) else "END"
        if nxt not in ("WH", "END", "VERB", "ADJ", "PARTICLE"):
            return _name(a)
        return f"{_name(a)} {_name(b)}"

    def _target_of(self, cl: Clause) -> str:
        toks = cl.tokens
        # Sorulan kavram genellikle cümlenin BAŞINDAKİ addır ('Elma gerçekte nerededir?', 'Kapı nasıl
        # açılır?'). Soru belirtecinden geriye bakmak, araya giren zarfları ('gerçekte') hedef sanıyordu.
        for i, t in enumerate(toks):
            if t.role == "WH":
                break
            if t.role in ("NOUN", "ADJ") and t.word not in NON_SUBJECT \
                    and t.word not in WH_WORDS and t.lemma not in WH_WORDS \
                    and not (t.lemma in PRONOUN_STEMS and t.case in ("", "NOM")):
                return self._np_at(toks, i)
        wh_i = next((i for i, t in enumerate(toks) if t.role == "WH"), -1)
        if wh_i > 0:
            for j in range(wh_i - 1, -1, -1):
                t = toks[j]
                if t.role in ("NOUN", "ADJ") and t.word not in NON_SUBJECT \
                        and t.word not in WH_WORDS and t.lemma not in WH_WORDS \
                        and not (t.lemma in PRONOUN_STEMS and t.case in ("", "NOM")):
                    return self._np_at(toks, j)
        if cl.obj:
            return cl.obj
        if cl.subject:
            return cl.subject
        return ""

    # ------------------------------------------------------------ YAPI ÖĞRENİMİ
    def content_signature(self, cl: Clause) -> tuple[str, ...]:
        """İçerik sözcüklerini 'yuva' sayan iskelet: 'AD:ACC + FİİL:PAST:1SG'."""
        return cl.signature


class FrameMemory:
    """Gözlemden öğrenilen yapı ve valans belleği (kalıcı, kaydedilebilir)."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed + 23)
        self.verb_counts: Counter = Counter()
        self.valency: dict[str, Counter] = defaultdict(Counter)      # fiil -> {hâl: sayı}
        self.frames: Counter = Counter()                             # iskelet -> sayı
        self.frame_examples: dict[tuple, str] = {}
        self.verb_objects: dict[str, Counter] = defaultdict(Counter)  # fiil -> nesne sözcükleri
        self.learned_count = 0

    # --- gözlem
    def observe(self, cl: Clause) -> None:
        if cl.verb:
            self.verb_counts[cl.verb] += 1
            if cl.obj:
                self.valency[cl.verb][cl.obj_case or "NOM"] += 1
                self.verb_objects[cl.verb][cl.obj] += 1
            if cl.dative:
                self.valency[cl.verb]["DAT"] += 1
        sig = cl.signature
        if sig:
            self.frames[sig] += 1
            self.frame_examples.setdefault(sig, cl.text)
        self.learned_count += 1
        if len(self.frames) > 20000:
            self.frames = Counter(dict(self.frames.most_common(12000)))
            self.frame_examples = {k: v for k, v in self.frame_examples.items() if k in self.frames}

    # --- sorgular
    def object_case(self, verb: str, default: str = "ACC") -> str:
        c = self.valency.get(verb) or Counter()
        return c.most_common(1)[0][0] if c else default

    def knows_verb(self, verb: str) -> bool:
        return self.verb_counts.get(verb, 0) > 0

    def most_common_frame(self, n: int = 3) -> list[tuple[tuple, int]]:
        return self.frames.most_common(n)

    def valency_report(self, top: int = 8) -> str:
        items = sorted(self.valency.items(), key=lambda kv: -sum(kv[1].values()))[:top]
        return ", ".join(f"{v}:{'/'.join(c.keys())}" for v, c in items)

    def snapshot(self) -> dict:
        return {"verbs": dict(self.verb_counts.most_common(200)),
                "valency": {k: dict(v) for k, v in list(self.valency.items())[:200]},
                "frames": [("|".join(k), n) for k, n in self.frames.most_common(200)]}

    def load_snapshot(self, d: dict | None) -> None:
        if not d:
            return
        self.verb_counts = Counter(d.get("verbs") or {})
        self.valency = defaultdict(Counter, {k: Counter(v) for k, v in (d.get("valency") or {}).items()})
        self.frames = Counter({tuple(str(k).split("|")): n for k, n in (d.get("frames") or [])})


@dataclass
class MetaQuestion:
    kind: str          # HOW_SAY | HOW_DO | MEANING | DEFINITION | UNKNOWN_Q
    target: str        # sorulan sözcük
    verb: str
    clause: Clause
    wh_phrase: str = ""  # UNKNOWN_Q: sorunun neyi sorduğu ('ne zaman', 'ne işe yarar')


class ConstructionCortex:
    """Cümleyi yapı olarak okuyup 'ne soruluyor?' sorusunu genelleyen korteks."""

    def __init__(self, morph, lexicon=None, seed: int = 0):
        self.morph = morph
        self.lexicon = lexicon
        self.parser = ClauseParser(morph, lexicon)
        self.memory = FrameMemory(seed)
        self.rng = random.Random(seed + 41)
        self.meta_counts: Counter = Counter()
        self.pending: list[str] = []      # anlamı bilinmeyen, öğrenilmeyi bekleyen sözcükler

    # ---------------------------------------------------------------- YARDIMCI
    def known_word(self, w: str, agi=None) -> bool:
        """Sözcük Nora'nın dünyasında var mı: sözlük, ad/sıfat hazinesi, kavram ağı, anı."""
        if not w:
            return False
        low = tr_lower(w)
        try:
            from nora_knowledge import (COMMON_ADJECTIVES, COMMON_NOUNS, CORE_TURKISH_VERBS,
                                        PROPER_NOUNS, STATE_ADJECTIVES, WORD_CLASSES)
            if low in (COMMON_NOUNS | COMMON_ADJECTIVES | STATE_ADJECTIVES | PROPER_NOUNS):
                return True
            if low in CORE_TURKISH_VERBS or low in WORD_CLASSES:
                return True
        except Exception:
            pass
        if self.lexicon and low in self.lexicon:
            return True
        if agi is not None:
            try:
                graph = agi.lang.graph
                if low in getattr(graph, "nodes", {}) or graph.query(low, None):
                    return True
                if low in getattr(graph, "_out_edges", {}):
                    return True
            except Exception:
                pass
        return False

    def usage_quote(self, w: str, agi) -> str:
        """Nora o sözcüğü GERÇEKTEN nerede duymuş/kullanmış: en taze anıdan alıntı."""
        low = tr_lower(w)
        try:
            for ep in reversed(list(agi.broca_organ.episodic.episodes)[-300:]):
                for txt, who in ((ep.user_text, "sens"),):
                    t = tr_lower(txt or "")
                    if not low or len(t.split()) < 2:
                        continue
                    if not re.search(rf"(^|\W){re.escape(low)}", t):
                        continue
                    # Sözcüğün KENDİSİ hakkında sorulmuş bir cümle kullanım örneği değildir
                    # ('Selam nasıl denilir?' -> selamın nasıl kullanıldığını göstermez).
                    if self.meta_question(t) is not None:
                        continue
                    return (txt or "").strip(), who
        except Exception:
            pass
        return "", ""

    # ---------------------------------------------------------------- YAPI OKUMA
    def parse(self, text: str) -> Clause:
        return self.parser.parse(text)

    def observe(self, text: str) -> Clause:
        cl = self.parse(text)
        self.memory.observe(cl)
        return cl

    # ---------------------------------------------------------------- SORU TÜRÜ
    def meta_question(self, text: str) -> MetaQuestion | None:
        """'X nasıl denilir / ne demek / nedir' yapılarını GENEL olarak tanır.

        Karar sözcük dizisine değil, YAPIYA dayanır: soru belirteci + yüklem sınıfı +
        yüklemin çatısı. Yeni bir sözcükle aynı yapı gelirse aynı soru türü çıkar.
        """
        # Soru, turun SON cümlesindedir: çok cümleli bir turda yapıyı son cümleye göre oku.
        # ('Zirkon parlak bir taştır. ... Zirkon ne işe yarar?' -> soru son cümledir.)
        # UCUZ ÖN SÜZGEÇ (O(1) yanıt süresi): sözcük sorusu olmayan turlar biçimbilime hiç girmez.
        _low0 = tr_lower(text or "")
        if any(c in _low0 for c in ("sence", "bence", "ne düşün", "fikrin", "için ne")):
            return None
        # SOHBET ÜSTÜ sorular sözcük sorusu değildir: 'Az önce ne konuştuk?', 'ne demiştim?'
        if any(c in _low0 for c in ("konuş", "söyledi", "demiş", "dediğim", "hatırla", "bahset",
                                   "sorduğum", "sorumu", "anladın", "demek istedi")):
            return None
        if not re.search(r"\b(ne|nedir|nasıl|neden|niçin|niye|kim|kimdir|nere\w*|kaç|hangi|"
                         r"ne zaman|ne kadar|ne işe|ne yapar)\b|demek|anlam", _low0):
            return None
        # ÇOK CÜMLELİ tur: cevap önceki cümlelerden ÇIKARIM ister (A>B>C). Sözcük korteksi karışmaz.
        _parts = [p for p in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if p.strip()]
        if len(_parts) > 1:
            return None
        focus = self._question_sentence(text)
        if focus is None:
            return None
        if len(focus.split()) > 9:
            return None
        # MATEMATİK/FORMÜL sorusu sözcük sorusu değildir: '= ^ ∫ √' ya da '3x', 'x2' gibi
        # simge-sayı örüntüsü varsa yapı okuması devre dışı (hesap korteksi cevaplar).
        if re.search(r"[=^∫√±]|\d\s*[a-zA-Zçğıöşü]|[a-zA-Zçğıöşü]\s*\d", focus):
            return None
        cl = self.parse(focus)
        text = focus
        low = tr_lower(text)
        # 1) SORU ve KISA olmalı: sözcük/eylem sorusu kısa bir yapıdır (2-6 sözcük).
        #    Uzun cümleler (anlatı, hesap, çok tümleçli sorular) bu korteksin işi değil.
        if not (cl.is_question or cl.wh):
            return None
        if not (2 <= len(cl.tokens) <= 6):
            return None
        verb = cl.verb or ""
        passive = "PASSIVE" in (cl.voice or "")
        lemma = self._verb_lemma(verb)
        wh = cl.wh
        target = cl.target
        # Tırnak içindeki söz konudur: sözcükse hedef o; ÖBEK ise (deyim, tamlama) bu korteks karışmaz.
        quoted = [q.strip() for q in re.findall(r"[\"\'‘’“”]([^\"\'‘’“”]{1,40})[\"\'‘’“”]", text)]
        if quoted:
            if any(" " in q for q in quoted):
                return None
            target = quoted[0]
        if not target:
            return None
        # GÖRÜŞ/DEĞERLENDİRME sorusu sözcük dersi değildir ('Sence özgür irade nedir?').
        if any(c in low for c in ("sence", "bence", "ne düşün", "fikrin", "için ne")):
            return None
        # Bahsedilen şey cümlenin BAŞINDAKİ ad olmalı; önünde niteleyici varsa bu bir
        # ÇOK SÖZCÜKLÜ terimdir ('özgür irade', 'kara delik') ve sözcük dersi değildir.
        words = [t.word for t in cl.tokens]
        if target in words:
            idx = words.index(target)
            if idx > 0 and words[idx - 1] not in ("bir", "bu", "şu", "o", "her"):
                return None
        # 2) SÖZ EDİMİ / NESNE adı olmalı: bahsedilen şey bir tek sözcük (ad/sıfat) olur.
        try:
            from nora_knowledge import WORD_CLASSES
        except Exception:
            WORD_CLASSES = {}
        _tgt_an = {}
        try:
            _tgt_an = self.morph.analyze_word(target, self.lexicon) or {}
        except Exception:
            _tgt_an = {}
        if _tgt_an.get("is_verb") and target not in WORD_CLASSES and target not in self.vocab:
            return None
        # 3) ANLAM: yalnız açık 'ne demek / anlam' kalıbı (tahmin yok)
        if "demek" in low or "anlam" in low:
            return MetaQuestion("MEANING", target, lemma, cl)
        # 4) TANIM: yalnız cümle 'nedir/kimdir' ile BİTİYORSA ('X nedir?')
        _last = tr_lower(cl.tokens[-1].word) if cl.tokens else ""
        if _last in ("nedir", "kimdir", "ne", "kim"):
            return MetaQuestion("DEFINITION", target, lemma, cl)
        # 5) NASIL + SÖYLEME FİİLİ (edilgen ya da geniş zaman): 'X nasıl denilir?'
        if wh == "nasıl" and lemma in SAY_VERBS:
            return MetaQuestion("HOW_SAY", target, lemma, cl)
        # 5b) SÖZ EDİMİ öbeği: 'teşekkür edilir', 'selam verilir', 'özür dilenir'
        if wh == "nasıl" and target in WORD_CLASSES and lemma in ("et", "ver", "dile", "ol", "yap"):
            return MetaQuestion("HOW_SAY", target, lemma, cl)
        # 6) NASIL + YAPMA FİİLİ (edilgen ya da ettirgen): 'X nasıl açılır?'
        if wh == "nasıl" and (lemma in DO_VERBS or passive):
            return MetaQuestion("HOW_DO", target, lemma, cl)
        # 7) DİĞER SORU TÜRLERİ: 'Zemheri ne zaman gelir?', 'Nabız ne işe yarar?'
        #    Soru BİLİNMEYEN bir kavramın üzerine kuruluysa (herhangi bir wh) boşluk dürüstçe
        #    adlandırılır. Bilinen kavramda susulur (o zaman ana bilgi hattı yanıtlar).
        # Yalnız YÜKLEMLİ soru bir kavram hakkında sorudur: 'ne var ne yok' gibi kalıp
        # sözler (selamlaşma, argo) yüklem taşımaz ve sözcük sorusu sayılmaz.
        _CUES = ("ne zaman", "ne kadar", "ne işe yarar", "ne yapar", "kaç",
                 "nereyi", "nereye", "nerede", "nereden")
        _pred = next((k for k in _CUES if k in low), "")
        if wh and (cl.verb or cl.copula or _pred):
            return MetaQuestion("UNKNOWN_Q", target, lemma, cl, wh_phrase=_pred or wh)
        return None

    @staticmethod
    def _question_sentence(text: str) -> str | None:
        """Turun soru cümlesini seç: '?' taşıyan son cümle; yoksa son cümle."""
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if p.strip()]
        if not parts:
            return None
        for p in reversed(parts):
            if "?" in p or p.endswith("?"):
                return p
        return parts[-1]

    def _verb_lemma(self, verb: str) -> str:
        """Çatı gövdesinden yalın kök: 'sürül' -> 'sür', 'edil' -> 'et', 'denil' -> 'de'."""
        v = tr_lower(verb or "")
        if not v:
            return ""
        if v in SAY_VERBS or v in DO_VERBS or v in MEAN_VERBS:
            return v
        for cut in range(0, 4):
            cand = v[: len(v) - cut] if cut else v
            for base in (cand, self.morph._unsoften(cand) if hasattr(self.morph, "_unsoften") else cand):
                if base in self.morph.KNOWN_VERB_ROOTS:
                    return base
        return v

    # ---------------------------------------------------------------- YANIT
    def answer_meta(self, mq: MetaQuestion, agi) -> str:
        """Yapı tanındı: şimdi tek soru kaldı — sorulan SÖZCÜĞÜ biliyor muyum?"""
        self.meta_counts[mq.kind] += 1
        target = mq.target or ""
        if not target:
            return self._no_target(mq)
        known = self.known_word(target, agi)
        if not known:
            return self._gap_answer(mq, agi)
        return self._known_answer(mq, agi)

    # --- bilinmeyen sözcük: dürüst boşluk + araştırma
    def _gap_answer(self, mq: MetaQuestion, agi) -> str:
        """Sözcük BİLİNMİYOR: anlam boşluğunu adıyla söyler, sorunun cevabının da bu yüzden
        verilemediğini açıklar ve araştırmaya/öğrenmeye yönelir. Hiçbir şey uydurulmaz."""
        target = mq.target
        # Erken meta-soru yolu da aynı merak defterini kullanır; yalnız metinde
        # 'merak listeme aldım' demek davranış kanıtı değildir.
        try:
            agi.life.note_unknown_concept(agi, target)
        except Exception:
            pass
        gen = self._quote_inflect(target, "GEN")
        acc = self._quote_inflect(target, "ACC")
        # 1) Dürüst boşluk: önce 'bilgim yok', sonra tam olarak neyi bilmediğim
        head = f"{tr_capitalize(target)} hakkında bilgim yok; {gen} ne demek olduğunu bilmiyorum"
        # 2) Bu yüzden SORULAN ŞEY de yapılamaz: yapıya göre adlaştırma
        if mq.kind == "HOW_SAY":
            try:
                from nora_broca import TurkishSyntaxEngine as SYN
                nom = SYN.nominalize_dik("söylen", poss=3, case="ACC")   # söylendiğini
            except Exception:
                nom = "söylendiğini"
            extra = f"bu yüzden nasıl {nom} de bilmiyorum"
        elif mq.kind == "HOW_DO":
            try:
                from nora_broca import TurkishSyntaxEngine as SYN
                nom = SYN.nominalize_dik(self._passive_of(mq.verb), poss=3, case="ACC")
            except Exception:
                nom = "yapıldığını"
            extra = f"bu yüzden nasıl {nom} de bilmiyorum"
        elif mq.kind == "UNKNOWN_Q":
            # Sorunun NEYİ sorduğu adlandırılır: 'ne zaman gelir' -> 'ne zaman geldiğini'.
            _wp = (mq.wh_phrase or "").strip()
            _verb = mq.verb or ""
            _nom = ""
            if _verb:
                try:
                    from nora_broca import TurkishSyntaxEngine as SYN
                    _nom = SYN.nominalize_dik(self._verb_lemma(_verb), poss=3, case="ACC")
                except Exception:
                    _nom = ""
            _qmap = {"ne zaman": "ne zaman", "ne kadar": "ne kadar", "ne işe yarar": "ne işe yaradığını",
                     "ne yapar": "ne yaptığını", "kaç": "kaç olduğunu", "nerede": "nerede",
                     "nereyi": "nereyi", "nereye": "nereye", "nereden": "nereden",
                     "nere": "nerede", "neden": "neden", "niçin": "niçin", "kim": "kim",
                     "nasıl": "nasıl", "ne": "ne", "hangi": "hangi"}
            _cue = _qmap.get(_wp, _wp)
            if _nom and _cue in ("ne zaman", "ne kadar", "nasıl", "ne", "neden", "niçin", "kim",
                                 "nerede", "nereyi", "nereye", "nereden"):
                extra = f"bu yüzden {_cue} {_nom} de bilmiyorum"
            elif _nom:
                extra = f"bu yüzden {_cue} olduğunu ({_nom}) da bilmiyorum"
            else:
                extra = "bu yüzden sorunun cevabını da bilmiyorum"
        else:
            extra = "bu yüzden anlamını açıklayamam"
        # 3) Araştırma: kaynak açıksa gerçekten dene, yoksa merak listesine al (uydurma yok)
        learned_txt, reached = self._try_research(target, agi)
        if learned_txt:
            try:
                resolver = getattr(agi, "_resolve_concept_gap", None)
                if resolver is not None:
                    resolver(target)
            except Exception:
                pass
            body = f"{head}; araştırdım, artık biliyorum: {self._join_sentences(learned_txt)}"
            ask = self.rng.choice((
                "Bunu sen de böyle mi biliyorsun?",
                f"{tr_capitalize(target)} hakkında daha ne öğrenmeliyim?",
                f"Bu bilgi işine yarar mı?",
            ))
            return f"{tr_capitalize(body)} {ask}"
        else:
            try:
                # Kavram ağında bir düğüm aç ki merak hedefi GERÇEKTEN bu sözcük olsun
                # (araştırma rastgele bir konuya değil, sorulan şeye yönelsin).
                net = getattr(agi, "auto_net", None)
                if net is not None:
                    if target not in getattr(net, "assemblies", {}):
                        try:
                            net.stimulate([target], valence=0.2, resolved_relations=0)
                        except Exception:
                            pass
                    node = getattr(net, "assemblies", {}).get(target)
                    if node is not None:
                        node.uncertainty = max(float(getattr(node, "uncertainty", 0.0)), 0.85)
                q = getattr(agi.auto_net, "curiosity_queue", None)
                if q is not None and target not in q:
                    q.append(target)
                if target not in self.pending:
                    self.pending.append(target)
                pending = getattr(agi, "pending_research", None)
                if isinstance(pending, list) and target not in pending:
                    pending.append(target)
            except Exception:
                pass
            body = (f"{head}; {extra}. Merak listeme aldım, kaynak açılınca araştıracağım"
                    if not reached else
                    f"{head}; {extra}. Kaynaklara ulaştım ama doğrulanmış bilgi çıkmadı, "
                    f"merak listemde tutuyorum")
        # 4) Karşı tarafa tek somut istek: sözcüğü öğret
        ask = self.rng.choice((
            f"{tr_capitalize(target)} ne demek, bunu öğrenmem gerek.",
            f"{acc} bir cümlede duyarsam anlarım.",
            f"{tr_capitalize(target)} bir cümlede geçerse oradan çıkarırım.",
        ))
        return f"{tr_capitalize(body)}. {ask}"

    # --- bilinen sözcük: bildiğimden yanıt
    def _known_answer(self, mq: MetaQuestion, agi) -> str:
        target = mq.target
        acc = self._quote_inflect(target, "ACC")
        gen = self._quote_inflect(target, "GEN")
        quote, who = self.usage_quote(target, agi)
        morph = self.morph
        # Sözcüğün edim sınıfı (sözlükten): 'merhaba' selamlaşmadır, karşılaşınca söylenir.
        cls_name, ctx = "", ""
        try:
            from nora_knowledge import WORD_CLASS_CONTEXT, WORD_CLASSES
            cls_name = WORD_CLASSES.get(tr_lower(target), "")
            ctx = WORD_CLASS_CONTEXT.get(cls_name, "")
        except Exception:
            pass
        say_passive = self._say_passive()

        if mq.kind == "HOW_SAY":
            if ctx:
                # Edilgen cümlede sözcük ÖZNE olur: yalın hâl ('Selam karşılaşınca söylenir').
                lead = f"{tr_capitalize(target)} {ctx} {say_passive}"
                if quote:
                    lead += f"; senden duyduğum kullanım: “{quote}”"
                ask = self.rng.choice(("Ben bunu ne zaman kullanacağımı merak ediyorum.", "Yerine başka ne dersin?",
                                       "Sence bu kullanım doğru mu?"))
                return f"{tr_capitalize(lead)}. {ask}"
            if quote:
                lead = self.rng.choice((
                    f"{acc} senden şöyle duymuştum: “{quote}”",
                    f"{acc} bende şu kullanımla duruyor: “{quote}”",
                ))
                tail = self.rng.choice(("Aynı yerde kullanırım.", "Böyle söylendiğini biliyorum.",
                                        "Yanlış mı hatırlıyorum?"))
                return f"{lead} {tail}"
            return (f"{acc} tanıyorum ama söylenişine dair bir örneğim yok; "
                    f"bir cümlede kullanırsan o kullanımı öğrenirim.")

        if mq.kind == "MEANING":
            rel = self._relations_text(target, agi)
            if rel:
                return f"{tr_capitalize(target)}: {rel}"
            if cls_name:
                return (f"{tr_capitalize(target)} bir {cls_name} sözü. "
                        f"{tr_capitalize(ctx)} {say_passive}." if ctx else
                        f"{tr_capitalize(target)} bir {cls_name} sözü.")
            return (f"{acc} tanıyorum ama anlamını tek cümlede "
                    f"toparlayamıyorum; benim için bir tanım iyi olur.")

        # HOW_DO / DEFINITION
        rel = self._relations_text(target, agi)
        if rel:
            return f"{tr_capitalize(target)}: {rel}"
        if mq.kind == "HOW_DO" and mq.verb:
            try:
                from nora_broca import TurkishSyntaxEngine as SYN
                nom = SYN.nominalize_dik(self._passive_of(mq.verb), poss=3, case="ACC")
            except Exception:
                nom = ""
            if nom:
                try:
                    v_pass = morph.conjugate_verb(self._passive_of(mq.verb), "PRES", person="3SG")
                except Exception:
                    v_pass = ""
                lead = (f"{acc} kelime olarak tanıyorum ama nasıl {nom} bilmiyorum"
                        if nom else f"{acc} kelime olarak tanıyorum ama nasıl yapıldığını bilmiyorum")
                ask = self.rng.choice(("Bana tarif eder misin?", "Bir kez anlatsan öğrenirim.",
                                       "Nereden başlanır?"))
                return f"{lead}. {ask}"
        if quote:
            return f"{tr_capitalize(target)} geçen bir anım var: “{quote}”. Devamını merak ediyorum."
        return (f"{acc} kelime olarak tanıyorum ama doğrulanmış bir tanımım yok; "
                f"bir cümleyle öğretirsen kullanabilirim.")

    def _no_target(self, mq: MetaQuestion) -> str:
        return self.rng.choice((
            "Neyin böyle olduğunu sorduğunu anlamadım; hangi sözcük ya da eylem?",
            "Yapıyı anladım, nesnesini kaçırdım: neyin nasıl olduğunu soruyorsun?",
        ))

    # --- araçlar
    def _relations_text(self, concept: str, agi, limit: int = 2) -> str:
        """Kavram ağındaki komşuluktan kısa tanım: önce ISA, sonra özellikler."""
        try:
            graph = agi.lang.graph
        except Exception:
            return ""
        # query (kenar, isa_yolu) çiftleri döndürür; öncelik sırası anlamlı tanım içindir.
        prio = {"ISA": 0, "HAS_PROP": 1, "USED_FOR": 2, "CAN_DO": 3, "CAUSES": 4}
        edges = []
        try:
            for item in graph.query(concept, tuple(prio.keys())):
                e = item[0] if isinstance(item, (tuple, list)) else item
                if e is not None:
                    edges.append(e)
        except Exception:
            return ""
        edges.sort(key=lambda e: prio.get(e.rel, 9))
        if any(getattr(e, "rel", "") == "ISA" for e in edges):
            # Tanım varsa yalnız tanım: 'kuantum daldır' gibi kopuk özellikler kafa karıştırır.
            edges = [e for e in edges if e.rel == "ISA"] + [e for e in edges if e.rel != "ISA"][:0]
        out = []
        for e in edges[:limit]:
            try:
                from nora_lang import Proposition
                out.append(agi.lang.broca.realize_prop(
                    Proposition(subj=concept, rel=e.rel, obj=e.obj)))
            except Exception:
                out.append(f"{concept} {e.rel} {e.obj}")
        txt = " ".join(out).strip()
        if txt and txt[-1] not in ".!?":
            txt += "."
        return tr_capitalize(txt)

    def _quote_inflect(self, word: str, case: str) -> str:
        """Bahsedilen sözcüğü tırnaklı/kesmeli çek: 'Selam'ın', 'Bisiklet'in'.

        KÖK SÖZLÜKTE YOKSA ses olayı UYGULANMAZ: 'kuark' -> 'kuarkın' (kuarğın değil).
        Tanımadığı kökü yumuşatmak/düşürmek uydurma olur; Türkçede alıntı sözler zaten
        çoğunlukla yumuşamaz.
        """
        w = tr_lower(word)
        _vocab = getattr(self.parser, "vocab", None) or set()
        _known_root = (w in _vocab) or bool(self.lexicon and w in self.lexicon)
        if not _known_root:
            surf = self._inflect_plain(w, case)
        else:
            try:
                surf = self.morph.inflect_noun(w, case)
            except Exception:
                surf = w
        cap = tr_capitalize(w)
        if surf == w:
            return cap
        if surf.startswith(w):
            return f"{cap}'{surf[len(w):]}"
        return tr_capitalize(surf)

    def _inflect_plain(self, w: str, case: str) -> str:
        """Ses olayı UYGULAMADAN hal eki (tanımadığı kökü yumuşatıp uydurmamak için)."""
        from nora_lang import ALL_VOWELS
        if not w:
            return w
        c = (case or "NOM").upper()
        if c == "NOM":
            return w
        try:
            v4 = self.morph.vowel_4(w)
            ve = self.morph.vowel_e_a(w)
        except Exception:
            v4, ve = "ı", "a"
        _v4 = {"ı": "ı", "a": "ı", "u": "u", "ü": "ü", "i": "i", "e": "i"}.get(v4, "ı")
        _last_vowel = w[-1] in ALL_VOWELS
        if c == "ACC":
            return w + ("y" if _last_vowel else "") + _v4
        if c == "DAT":
            return w + ("y" if _last_vowel else "") + ve
        if c in ("LOC", "ABL"):
            _d = self.morph.dental_d_t(w) if hasattr(self.morph, "dental_d_t") else "d"
            _d = "d" if _d in ("d", "t") and w[-1] in ALL_VOWELS else _d
            _tail = _d + ve
            return w + (_tail if c == "LOC" else _tail + "n")
        if c == "GEN":
            return w + ("n" if _last_vowel else "") + _v4 + "n"
        if c == "INS":
            return w + ("yl" if _last_vowel else "l") + ve
        return w

    def _say_passive(self) -> str:
        """'söylenir' — söyleme fiilinin edilgen geniş zamanı (çekim motorundan)."""
        try:
            stem = self.morph.voice_stem("söyle", "PASSIVE")
            return self.morph.conjugate_verb(stem, "AOR", person="3SG")
        except Exception:
            return "söylenir"

    def _passive_of(self, verb: str) -> str:
        try:
            return self.morph.voice_stem(verb, "PASSIVE")
        except Exception:
            return verb

    def _try_research(self, target: str, agi) -> tuple[str, bool]:
        """SORULAN SÖZCÜĞÜ hedefli araştırma: kaynak açıksa gerçekten okur.

        Rastgele bir merak konusu değil, tam olarak sorulan sözcük araştırılır. İddialar
        doğrudan kavram ağına girmez: önce hipotez havuzuna yazılır, ≥2 bağımsız kaynak
        doğrulayınca terfi eder (uydurma/halüsinasyon yok).

        Dönüş: (öğrenilenlerin cümlesi, kaynağa_ulaşıldı_mı)
        """
        try:
            sources = list(getattr(getattr(agi, "life", None), "sources", []) or [])
        except Exception:
            sources = []
        if not sources:
            return "", False
        pool = getattr(agi.life, "pool", None)
        if pool is None:
            return "", False
        reached = 0
        for sid, fetch in sources[:3]:
            try:
                text = str(fetch(target, target) or "")
            except Exception:
                text = ""
            if not text.strip():
                continue
            reached += 1
            try:
                pool.ingest(agi, text, sid, target)
            except Exception:
                continue
        if not reached:
            return "", False
        try:
            promoted = pool.promote_ready(agi)
        except Exception:
            promoted = []
        learned = [(p.get("triple"), p.get("credence")) for p in promoted
                   if tr_lower(str(p.get("topic") or "")) == tr_lower(target)]
        # Tanım (ISA) varsa YALNIZ onu söyle: 'X bir Y'dir' en anlaşılır cevaptır; kopuk
        # özellikler ('kuantum daldır') tek başına tanım yerine geçmez.
        learned.sort(key=lambda t: 0 if (t[0] and t[0][1] == "ISA") else 1)
        if any(t[0] and t[0][1] == "ISA" for t in learned):
            learned = [t for t in learned if t[0] and t[0][1] == "ISA"][:1]
        if not learned:
            return "", True
        texts = []
        try:
            from nora_lang import Proposition as _P
            for (s_, r_, o_), cred in learned[:2]:
                texts.append(agi.lang.broca.realize_prop(_P(subj=s_, rel=r_, obj=o_)))
        except Exception:
            texts = [f"{s_} {r_} {o_}" for (s_, r_, o_), _c in learned[:2]]
        return self._join_sentences(". ".join(texts)), True

    @staticmethod
    def _join_sentences(text: str) -> str:
        """Parçaları cümle gibi birleştir: her parça büyük harfle, aralarında nokta."""
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|(?<=\w)\s{2,}", text) if p.strip()]
        if not parts:
            return text.strip()
        return " ".join(tr_capitalize(p if p[-1] in ".!?" else p + ".") for p in parts)


def attach(agi_cls=None) -> None:      # ileride doğrudan sınıf bağlamak için
    return None

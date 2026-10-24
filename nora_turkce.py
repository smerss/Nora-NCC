#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_turkce.py — Türkçenin EKSİKSİZ çekim/türetim çekirdeği.

Bu modül TurkishMorphology sınıfına eklenen yetenekleri barındırır; hiçbir hazır cümle yoktur,
yalnızca ek envanteri ve sesbilim kuralları vardır:

  * BİRLEŞİK ZAMANLAR  : geliyordu, gelirmişim, gelecektim, gelmeliydim, geliyorsam
  * SIFAT-FİİLLER      : gelen, geldiğim, gelecek, gelmiş, gelmez, geliş, gelme
  * YAPIM EKLERİ       : dostluk, susuz, kitapçı, Türkçe, akşamki, küçücük, evli
  * ÇATI ÇÖZÜMLEME     : yazıldı/yaptırıldı/görüşüldü -> kök + çatı + kip + kişi (sentezle doğrulanır)
  * SAYILAR            : milyar/milyon, sıra sayıları, ondalık okunuş

Her fonksiyon `cls` (sınıf) üzerinden sesbilim yardımcılarını kullanır; böylece çözümleme ve
üretim AYNI kurallardan beslenir (tek kaynak).
"""
from __future__ import annotations

import re

from nora_lang import ALL_VOWELS, soften_verb_t, tr_lower
from nora_knowledge import PROPER_NOUNS, STATE_ADJECTIVES


def _v4(cls, w: str) -> str:
    return cls.vowel_4(w)


def _ea(cls, w: str) -> str:
    return cls.vowel_e_a(w)


def _strip_inf(v: str) -> str:
    for inf in ("mek", "mak"):
        if len(v) > len(inf) + 1 and v.endswith(inf):
            return v[:-3]
    return v


# ============================================================================ BİRLEŞİK ZAMAN
COMPOUND_MAP = {
    "PRES_HIK": ("PRES", "HIK"), "AOR_HIK": ("AOR", "HIK"), "FUT_HIK": ("FUT", "HIK"),
    "NEC_HIK": ("NEC", "HIK"), "EVID_HIK": ("EVID", "HIK"), "PAST_HIK": ("PAST", "HIK"),
    "PRES_RIV": ("PRES", "RIV"), "AOR_RIV": ("AOR", "RIV"), "FUT_RIV": ("FUT", "RIV"),
    "NEC_RIV": ("NEC", "RIV"), "EVID_RIV": ("EVID", "RIV"),
    "PRES_SART": ("PRES", "SART"), "AOR_SART": ("AOR", "SART"), "FUT_SART": ("FUT", "SART"),
    "EVID_SART": ("EVID", "SART"), "NEC_SART": ("NEC", "SART"),
}


def compound_tense(cls, stem: str, base_tense: str, aux: str, neg: bool = False,
                   person: str = "3SG") -> str:
    """Temel kipin 3. tekil yüzeyi + (y)di / (y)mış / (y)se + kişi eki."""
    p = str(person or "3SG").upper()
    base = cls.conjugate_verb(stem, base_tense, neg=neg, person="3SG")
    if not base:
        return base
    v4 = _v4(cls, base)
    ea = _ea(cls, base)
    a = str(aux or "").upper()
    ybuf = "y" if base[-1] in ALL_VOWELS else ""
    if a == "HIK":
        d = cls.dental_d_t(base)
        body = f"{base}{ybuf}{d}{v4}"
        pers = {"1SG": "m", "2SG": "n", "1PL": "k", "2PL": f"n{v4}z", "3PL": f"l{ea}r"}.get(p, "")
    elif a == "RIV":
        body = f"{base}{ybuf}m{v4}ş"
        pers = {"1SG": f"{v4}m", "2SG": f"s{v4}n", "1PL": f"{v4}z", "2PL": f"s{v4}n{v4}z",
                "3PL": f"l{ea}r"}.get(p, "")
    elif a == "SART":
        body = f"{base}{ybuf}s{ea}"
        pers = {"1SG": "m", "2SG": "n", "1PL": "k", "2PL": f"n{v4}z", "3PL": f"l{ea}r"}.get(p, "")
    else:
        return base
    return body + pers


# ============================================================================ SIFAT-FİİL
def _possessive(cls, base: str, poss: int, case: str) -> str:
    if not poss:
        return base
    v4 = _v4(cls, base)
    suff = {1: f"{v4}m", 2: f"{v4}n", 3: f"{v4}", 4: f"{v4}m{v4}z", 5: f"{v4}n{v4}z"}.get(poss, "")
    out = base + suff
    case = str(case or "NOM").upper()
    if case != "NOM":
        out = cls.inflect_noun(out, case, poss3=(poss == 3))
    return out


def that_clause(cls, root: str, poss: int = 1, case: str = "ACC", neg: bool = False,
                subject: str = "") -> str:
    """'-DIğI' yan cümleciği, istenen kişiyle: geldiğim, geldiğini, geldiğimizi.

    `subject` verilirse tamlama kurulur ('köpeğin geldiğini')."""
    v = _strip_inf(tr_lower(str(root or "").strip()))
    if not v:
        return ""
    body = (v + "m" + _ea(cls, v)) if neg else v
    d = cls.dental_d_t(body)
    v4 = _v4(cls, body)
    stem_soft = f"{body}{d}{v4}ğ"
    out = _possessive_soft(cls, stem_soft, poss, case)
    if subject:
        try:
            return f"{cls.inflect_noun(tr_lower(subject), 'GEN')} {out}"
        except Exception:
            return out
    return out


def _possessive_soft(cls, stem_soft: str, poss: int, case: str) -> str:
    """'ğ' ile biten gövde + iyelik + hal. Hal eki iyelikten SONRA kaynaştırma 'n' ile gelir."""
    v4 = _v4(cls, stem_soft)
    ea = _ea(cls, stem_soft)
    if poss == 1:
        out = stem_soft + v4 + "m"
    elif poss == 2:
        out = stem_soft + v4 + "n"
    elif poss == 3:
        out = stem_soft + v4
    elif poss == 4:
        out = stem_soft + v4 + "m" + v4 + "z"
    elif poss == 5:
        out = stem_soft + v4 + "n" + v4 + "z"
    else:
        out = stem_soft
    case = str(case or "NOM").upper()
    if case == "NOM" or not poss:
        return out
    # İyelikli gövdeye hal: 3. kişi 'n' kaynaştırması, diğerlerinde doğrudan
    n_buf = "n" if poss == 3 else ""
    if case == "ACC":
        return out + n_buf + v4
    if case == "DAT":
        return out + n_buf + ea
    if case == "LOC":
        return out + n_buf + cls.dental_d_t(out + n_buf) + ea
    if case == "ABL":
        return out + n_buf + cls.dental_d_t(out + n_buf) + ea + "n"
    if case == "GEN":
        return out + n_buf + v4 + "n"
    return out


def participle_form(cls, root: str, kind: str = "an", neg: bool = False,
                    poss: int = 0, case: str = "NOM") -> str:
    v = _strip_inf(tr_lower(str(root or "").strip()))
    if not v:
        return ""
    k = str(kind or "").lower()
    v_s = soften_verb_t(v)
    buf = "y" if v_s[-1] in ALL_VOWELS else ""
    v4 = _v4(cls, v_s)
    ea = _ea(cls, v_s)
    body = (v + "m" + _ea(cls, v)) if neg else v
    if k in ("an", "en"):
        return _possessive(cls, f"{body}{buf}{ea}n", poss, case)
    if k in ("dik", "dık", "diğ", "dig"):
        d = cls.dental_d_t(body)
        if poss:
            return _possessive_soft(cls, f"{body}{d}{v4}ğ", poss, case)
        return f"{body}{d}{v4}k"
    if k in ("acak", "ecek"):
        base = f"{body}{buf}{'aca' if ea == 'a' else 'ece'}k"
        if poss:
            return _possessive_soft(cls, base[:-1] + "ğ", poss, case)
        return _possessive(cls, base, poss, case)
    if k in ("mis", "mış"):
        return _possessive(cls, f"{body}m{v4}ş", poss, case)
    if k in ("maz", "mez"):
        return _possessive(cls, f"{body}m{ea}z", poss, case)
    if k in ("is", "ış", "uş", "üş"):
        return _possessive(cls, f"{body}{buf}{v4}ş", poss, case)
    if k in ("ma", "me"):
        return _possessive(cls, f"{body}m{ea}", poss, case)
    return ""


# ============================================================================ YAPIM EKLERİ
_UP = {"ı": "ı", "a": "a", "u": "u", "ü": "ü", "i": "i", "e": "e", "o": "u", "ö": "ü"}


def derive_noun(cls, root: str, kind: str) -> str:
    r = tr_lower(str(root or "").strip())
    if not r:
        return ""
    k = str(kind or "").lower()
    if k in ("ce", "ça", "çe"):
        k = "ca"
    if k in ("cu", "cü", "çı", "çi"):
        k = "ci"
    v4 = _v4(cls, r)
    ea = _ea(cls, r)
    vowels = _UP[v4]
    buf = "y" if r[-1] in ALL_VOWELS else ""
    hard_end = r[-1] in "fstkçşhp"
    if k == "lik":
        return r + {"ı": "lık", "a": "lık", "u": "luk", "ü": "lük", "i": "lik", "e": "lik"}[v4]
    if k == "siz":
        return r + {"ı": "sız", "a": "sız", "u": "suz", "ü": "süz", "i": "siz", "e": "siz"}[v4]
    if k in ("ci", "ci_kisi"):
        ek = {"ı": "cı", "a": "cı", "u": "cu", "ü": "cü", "i": "ci", "e": "ci"}[v4]
        if hard_end:
            ek = "ç" + ek[1:]
        return r + ek
    if k == "ca":
        return r + ("ça" if hard_end and ea == "a" else ("çe" if hard_end else ("ca" if ea == "a" else "ce")))
    if k == "ki":
        if r[-1] in ALL_VOWELS:
            return r + "ki"
        return r + {"ı": "ki", "a": "ki", "u": "ku", "ü": "kü", "i": "ki", "e": "ki"}[v4]
    if k == "cik":
        ek = {"ı": "cık", "a": "cık", "u": "cuk", "ü": "cük", "i": "cik", "e": "cik"}[v4]
        return r + ("ç" + ek[1:] if hard_end else ek)
    if k == "li":
        return r + {"ı": "lı", "a": "lı", "u": "lu", "ü": "lü", "i": "li", "e": "li"}[v4]
    if k == "sel":
        return r + ("sal" if ea == "a" else "sel")
    if k == "gil":
        return r + buf + "gil"
    return ""


# ============================================================================ ÇATI ÜRETİMİ
IRREG_CAUS = {"çık": "çıkar", "iç": "içir", "düş": "düşür", "bit": "bitir", "kalk": "kaldır",
              "ak": "akıt", "kop": "kopar", "uç": "uçur", "geç": "geçir", "kaç": "kaçır",
              "piş": "pişir", "şaş": "şaşırt", "art": "artır", "az": "azalt"}


def voice_stem(cls, v: str, voice: str) -> str:
    """Çatı gövdesi: edilgen, ettirgen, dönüşlü, işteş (sesbilim kurallarıyla).
    'yaz'->'yazıl', 'oku'->'okun', 'yap'->'yaptır', 'gör'->'görüş'."""
    vo = str(voice or "").upper()
    if not vo or not v:
        return v
    v4 = _v4(cls, v)
    vowel_end = v[-1] in ALL_VOWELS
    if vo == "PASSIVE":
        if vowel_end:
            return v + "n"
        if v.endswith("l"):
            return v + v4 + "n"
        return v + v4 + "l"
    if vo == "CAUSATIVE":
        if v in IRREG_CAUS:
            return IRREG_CAUS[v]
        if vowel_end:
            return v + "t"
        if v.endswith("t"):
            return v + v4 + "r"
        return v + cls.dental_d_t(v) + v4 + "r"
    if vo == "REFLEXIVE":
        return v + ("n" if vowel_end else v4 + "n")
    if vo == "RECIPROCAL":
        return v + ("ş" if vowel_end else v4 + "ş")
    return v


# ============================================================================ BİRLEŞİK ZAMAN ÇÖZÜMLEME
_PERSON_MAP = {"": "3SG", "m": "1SG", "n": "2SG", "k": "1PL", "niz": "2PL", "nız": "2PL",
               "nuz": "2PL", "nüz": "2PL", "ler": "3PL", "lar": "3PL",
               "um": "1SG", "üm": "1SG", "ım": "1SG", "im": "1SG",
               "sun": "2SG", "sün": "2SG", "sın": "2SG", "sin": "2SG",
               "uz": "1PL", "üz": "1PL", "ız": "1PL", "iz": "1PL",
               "sunuz": "2PL", "sünüz": "2PL", "sınız": "2PL", "siniz": "2PL"}


def analyze_compound_tense(cls, w: str) -> dict:
    """Birleşik zaman (hikâye/rivayet/şart) çözümü.

    Eki sıyır, kalan GÖVDEyi taban kip çözücüsüyle çöz, sonucu sentezle doğrula.
    'geliyordum', 'gelirmiş', 'gelecektim', 'gelmeliydim', 'geliyorsa', 'yazacakmışım'."""
    low = tr_lower(w)
    if len(low) < 4:
        return {}
    plans = (
        ("HIK", ("du", "dü", "dı", "di", "tu", "tü", "tı", "ti",
                 "ydu", "ydü", "ydı", "ydi"), ("", "m", "n", "k", "niz", "nız", "nuz", "nüz", "ler", "lar")),
        ("RIV", ("muş", "müş", "mış", "miş", "ymuş", "ymüş", "ymış", "ymiş"),
         ("", "um", "üm", "ım", "im", "sun", "sün", "sın", "sin", "uz", "üz", "ız", "iz",
          "sunuz", "sünüz", "sınız", "siniz", "ler", "lar")),
        ("SART", ("sa", "se", "ysa", "yse"), ("", "m", "n", "k", "niz", "nız", "nuz", "nüz", "ler", "lar")),
    )
    for aux, eks, persons in plans:
        for ek in eks:
            for pers in persons:
                tail = ek + pers
                if not low.endswith(tail):
                    continue
                govde = low[: -len(tail)]
                if len(govde) < 2:
                    continue
                fin = cls._finite_verb(govde)
                if not fin or not fin.get("is_verb"):
                    continue
                stem = fin.get("stem", "")
                btense = fin.get("tense", "")
                bneg = bool(fin.get("neg"))
                if not stem or not btense:
                    continue
                if cls.conjugate_verb(stem, btense, neg=bneg, person="3SG") != govde:
                    continue
                return {"stem": stem, "is_verb": True, "tense": f"{btense}_{aux}",
                        "person": _PERSON_MAP.get(pers, "3SG"), "neg": bneg, "compound": aux,
                        "base_surface": govde}
    return {}


# ============================================================================ ÇATI ÇÖZÜMLEME
def analyze_voice(cls, w: str) -> dict:
    low = tr_lower(w)
    if len(low) < 4:
        return {}
    for cut in range(1, min(9, len(low) - 2) + 1):
        cand = low[:-cut]
        for base in (cand, cls._unsoften(cand)):
            if len(base) < 2 or base not in cls.KNOWN_VERB_ROOTS:
                continue
            layers = [base]
            for v1 in ("CAUSATIVE", "PASSIVE", "REFLEXIVE", "RECIPROCAL"):
                s1 = cls.voice_stem(base, v1)
                if s1 and s1 != base:
                    layers.append((v1, s1))
            # ikinci katman: 'yap+tır+ıl+dı', 'gör+üş+ül+dü'
            first = [x for x in layers if isinstance(x, tuple)]
            for v1, s1 in first:
                for v2 in ("PASSIVE", "CAUSATIVE", "RECIPROCAL"):
                    s2 = cls.voice_stem(s1, v2)
                    if s2 and s2 != s1:
                        layers.append((v1 + "+" + v2, s2))
            for item in layers:
                if isinstance(item, tuple):
                    vname, vstem = item
                else:
                    vname, vstem = "", item
                if not vstem:
                    continue
                for t in ("PRES", "PAST", "FUT", "AOR", "NEC", "EVID"):
                    for p in ("1SG", "2SG", "3SG", "1PL", "2PL", "3PL"):
                        if cls.conjugate_verb(vstem, t, person=p) == low:
                            return {"stem": base, "voice_stem": vstem, "voice": vname,
                                    "tense": t, "person": p, "is_verb": True}
    return {}


# ============================================================================ SAYILAR
BIG_SCALES = (("milyar", 10 ** 9), ("milyon", 10 ** 6), ("bin", 10 ** 3))


def int_to_words_full(cls, n: int) -> str:
    n = int(n)
    if n < 0:
        return "eksi " + int_to_words_full(cls, -n)
    if n < 10 ** 6:
        return cls.int_to_words(n)
    parts: list[str] = []
    for name, scale in BIG_SCALES:
        if n >= scale:
            head = n // scale
            n %= scale
            if not (name == "bin" and head == 1):
                parts.append(int_to_words_full(cls, head))
            parts.append(name)
    if n:
        parts.append(cls.int_to_words(n))
    return " ".join(parts)


def ordinal_tr(cls, n: int) -> str:
    words = int_to_words_full(cls, n)
    last = words.split()[-1]
    v4 = _v4(cls, last)
    if last[-1] in ALL_VOWELS:
        return words + "nc" + {"ı": "ı", "a": "ı", "u": "u", "ü": "ü", "i": "i", "e": "i"}[v4]
    ek = {"ı": "ıncı", "a": "ıncı", "u": "uncu", "ü": "üncü", "i": "inci", "e": "inci"}[v4]
    # 'dört' -> 'dördüncü' (ünsüz yumuşaması), 'yedi' -> 'yedinci'
    return cls.soften_stem(last) + ek


def decimal_tr(cls, number: float) -> str:
    txt = ("%.10f" % float(number)).rstrip("0").rstrip(".")
    if "." not in txt:
        return int_to_words_full(cls, int(float(number)))
    whole, frac = txt.split(".")
    if len(frac) == 1:
        frac_words = cls.int_to_words(int(frac))
    else:
        frac_words = int_to_words_full(cls, int(frac))
    return f"{int_to_words_full(cls, int(whole))} virgül {frac_words}"


# ============================================================================ YETERLİLİK (-(y)Abil-)
_ABIL_TENSE = (("iyor", "PRES"), ("ecek", "FUT"), ("acak", "FUT"), ("mış", "EVID"), ("miş", "EVID"),
               ("dı", "PAST"), ("di", "PAST"), ("du", "PAST"), ("dü", "PAST"),
               ("tı", "PAST"), ("ti", "PAST"), ("tu", "PAST"), ("tü", "PAST"), ("ir", "AOR"), ("ır", "AOR"))
_ABIL_PERSON = {"im": "1SG", "ım": "1SG", "um": "1SG", "üm": "1SG",
                "sin": "2SG", "sın": "2SG", "sun": "2SG", "sün": "2SG",
                "iz": "1PL", "ız": "1PL", "uz": "1PL", "üz": "1PL",
                "ler": "3PL", "lar": "3PL", "sınız": "2PL", "siniz": "2PL",
                "sunuz": "2PL", "sünüz": "2PL"}


def ability_stem(cls, root: str, neg: bool = False) -> str:
    """Yeterlilik gövdesi: 'gel' -> 'gelebil' (olumlu), 'gele' (olumsuz: geleme-)."""
    v = tr_lower(root)
    ybuf = "y" if v[-1:] in list("aeıioöuü") else ""
    if neg:
        return v + ybuf + cls.vowel_e_a(v)
    return v + ybuf + ("ebil" if cls.vowel_e_a(v) == "e" else "abil")


def ability_form(cls, root: str, tense: str = "AOR", person: str = "3SG", neg: bool = False) -> str:
    """-(y)Abil- çekimi; olumluda 'bil-' yardımcı fiili çekimlenir, olumsuzda kök +(y)A + me-."""
    v = cls._strip_inf(root) if hasattr(cls, "_strip_inf") else root
    if neg:
        return cls.conjugate_verb(ability_stem(cls, v, neg=True), tense, neg=True, person=person)
    return cls.conjugate_verb(ability_stem(cls, v), tense, person=person)


def analyze_ability(cls, w: str) -> dict | None:
    """"gelebilirim"/"okuyamadım" çözümlemesi: kök + yeterlilik + zaman + kişi."""
    m = re.match(r"^(.+?)(y?[ae])(bil|ma|me)(.*)$", w)
    if not m:
        return None
    root, lv, aux, tail = m.group(1), m.group(2), m.group(3), m.group(4)
    if len(root) < 2:
        return None
    neg = aux in ("ma", "me")
    if neg and not tail:
        return None          # 'gelme' isim-fiil/emir olabilir; yeterlilik değil
    bases = [root, cls._unsoften(root)]
    for b in list(bases):
        for v in ("a", "e", "ı", "i", "u", "ü"):
            bases.append(b + v)
    hit = next((b for b in bases if b in cls.KNOWN_VERB_ROOTS), None)
    if hit is None:
        return None
    if not neg:
        # 'bil' gövdesinden sonra gelen zaman/kişi: -ir-im, -iyor-um, -ecek-im, -di-m
        tense = "AOR"
        rest = tail
        for suf, t in _ABIL_TENSE:
            if rest.startswith(suf):
                tense, rest = t, rest[len(suf):]
                break
        person = _ABIL_PERSON.get(rest, "3SG")
        return {"stem": hit, "case": "NOM", "is_verb": True, "tense": tense,
                "person": person, "modal": "ABIL", "neg": False}
    # Olumsuz yeterlilik: kök+(y)A+me+[zaman]+kişi  ('okuyamadım', 'gelemez'→olumsuz geniş zaman)
    rest = tail
    tense, person = "AOR", "3SG"
    for suf, t in (("dı", "PAST"), ("di", "PAST"), ("du", "PAST"), ("dü", "PAST"),
                   ("tı", "PAST"), ("ti", "PAST"), ("yor", "PRES"), ("yacak", "FUT"),
                   ("yecek", "FUT"), ("yacağ", "FUT"), ("yeceğ", "FUT")):
        if rest.startswith(suf):
            tense, rest = t, rest[len(suf):]
            break
    if rest.startswith("y"):
        rest = rest[1:]
    person = _ABIL_PERSON.get(rest, "1SG" if rest in ("m",) else person)
    if rest == "m":
        person = "1SG"
    if tense == "PRES" and rest.endswith("um"):
        rest = rest[:-2]
    return {"stem": hit, "case": "NOM", "is_verb": True, "tense": tense,
            "person": person, "modal": "ABIL", "neg": True}


# ============================================================================ EMİR KİPİ
def imperative_form(cls, root: str, person: str = "2SG", neg: bool = False) -> str:
    """Emir: gel / gelin / gelsin / gelsinler; gelme / gelmeyin / gelmesin / gelmesinler."""
    v = cls._strip_inf(root) if hasattr(cls, "_strip_inf") else root
    p = person.upper()
    a = cls.vowel_e_a(v)
    v4 = cls.vowel_4(v)
    soft = cls.soften_verb_t(v) if hasattr(cls, "soften_verb_t") else v
    # Kişi ekleri kökün SON ünlüsüne göre 4'lü uyum alır: gel->gelin/gelsin, oku->okuyun/okusun
    _p = ({"a": "ın", "e": "in", "ı": "ın", "i": "in", "o": "un", "ö": "ün", "u": "un", "ü": "ün"}
          .get(v4, "ın"))
    _sin = "s" + {"a": "ı", "e": "i", "ı": "ı", "i": "i", "o": "u", "ö": "ü", "u": "u", "ü": "ü"}.get(v4, "ı") + "n"
    if not neg:
        if p == "2SG":
            return v
        if p == "2PL":
            return v + ("y" if v[-1:] in list("aeıioöuü") else "") + _p
        if p == "3SG":
            return v + _sin
        return v + _sin + ("lar" if a == "a" else "ler")
    # Olumsuzda uyum -mA ekini izler: okumayın, okumasın, görmeyin, görmesin
    _pn = "y" + ("ın" if a == "a" else "in")
    _sn = "s" + ("ın" if a == "a" else "in")
    if p == "2SG":
        return v + "m" + a
    if p == "2PL":
        return v + "m" + a + _pn
    if p == "3SG":
        return v + "m" + a + _sn
    return v + "m" + a + _sn + ("lar" if a == "a" else "ler")


def analyze_imperative(cls, w: str) -> dict | None:
    """"gelsin"/"yürüyün"/"gelmeyin"/"okusunlar" çözümlemesi (yalnızca bilinen kökle)."""
    for pat, neg in ((r"^(.+?)m([ae])y(in|ın|un|ün)$", True),
                     (r"^(.+?)m([ae])(sın|sin|sun|sün)(lar|ler)$", True),
                     (r"^(.+?)m([ae])(sın|sin|sun|sün)$", True),
                     (r"^(.+?)(sın|sin|sun|sün)(lar|ler)$", False),
                     (r"^(.+?)(sın|sin|sun|sün)$", False),
                     (r"^(.+?)(yın|yin|yun|yün)$", False),
                     (r"^(.+?)(ın|in|un|ün)$", False)):
        m = re.match(pat, w)
        if not m:
            continue
        root = m.group(1)
        bases = [root, cls._unsoften(root)]
        for b in list(bases):
            for v in ("a", "e", "ı", "i", "u", "ü"):
                bases.append(b + v)
        hit = next((b for b in bases if b in cls.KNOWN_VERB_ROOTS), None)
        if hit is None:
            continue
        person = "2PL" if pat.endswith(("yün)$", "vün)$")) or m.group(0).endswith(("yın", "yin", "yun", "yün", "ın", "in", "un", "ün")) else ""
        g = m.groups()
        if len(g) >= 2 and g[-1] in ("lar", "ler"):
            person = "3PL"
        elif "sın" in pat or "sin" in pat:
            person = "3SG"
        elif neg:
            person = "2PL"
        else:
            person = "2PL"
        return {"stem": hit, "case": "NOM", "is_verb": True, "tense": "IMP",
                "person": person, "neg": neg}
    return None


# ============================================================================ SÜREKLİLİK (-mAktA)
def progressive_form(cls, root: str, person: str = "3SG", neg: bool = False) -> str:
    """'-mAktA' süreklilik: gelmekteyim, okumaktasın, gitmektedir."""
    v = cls._strip_inf(root) if hasattr(cls, "_strip_inf") else root
    base = v + "m" + cls.vowel_e_a(v) + "kt" + cls.vowel_e_a(v)
    tail = {"1SG": "yim" if cls.vowel_e_a(v) == "e" else "yım",
            "2SG": "sin" if cls.vowel_e_a(v) == "e" else "sın",
            "3SG": "dir" if cls.vowel_e_a(v) == "e" else "dır",
            "1PL": "yiz" if cls.vowel_e_a(v) == "e" else "yız",
            "2PL": "siniz" if cls.vowel_e_a(v) == "e" else "sınız",
            "3PL": "ler" if cls.vowel_e_a(v) == "e" else "lar"}.get(person.upper(), "")
    return base + tail


def analyze_progressive(cls, w: str) -> dict | None:
    m = re.match(r"^(.+?)m([ae])kt([ae])(dir|dır|yim|yız|yım|sin|sın|siniz|sınız|ler|lar)?$", w)
    if not m:
        return None
    root, bases = m.group(1), None
    bases = [root, cls._unsoften(root)]
    for b in list(bases):
        for v in ("a", "e", "ı", "i", "u", "ü"):
            bases.append(b + v)
    hit = next((b for b in bases if b in cls.KNOWN_VERB_ROOTS), None)
    if hit is None:
        return None
    tail = m.group(4) or ""
    person = {"yim": "1SG", "yım": "1SG", "sin": "2SG", "sın": "2SG", "dir": "3SG", "dır": "3SG",
              "yiz": "1PL", "yız": "1PL", "siniz": "2PL", "sınız": "2PL", "ler": "3PL", "lar": "3PL"}.get(tail, "3SG")
    return {"stem": hit, "case": "NOM", "is_verb": True, "tense": "PROG", "person": person}


# ============================================================================ İSİM-FİİL / KOŞAÇ / -ki / -CA
def analyze_verbal_noun(cls, w: str) -> dict | None:
    """İsim-fiil çözümlemesi: gelme, okumak, geliş, gidiş (gövdeye 'm' katmadan)."""
    for suf, kind in (("mek", "MAK"), ("mak", "MAK"), ("me", "MA"), ("ma", "MA"),
                      ("iş", "IS"), ("ış", "IS"), ("uş", "US"), ("üş", "US")):
        if not w.endswith(suf) or len(w) - len(suf) < 2:
            continue
        root = w[: -len(suf)]
        bases = [root, cls._unsoften(root)]
        if root.endswith("y"):
            bases = [root[:-1], cls._unsoften(root[:-1])] + bases     # yürü+y+üş -> yürü
        for b in list(bases):
            for v in ("a", "e", "ı", "i", "u", "ü"):
                bases.append(b + v)
        hit = next((b for b in bases if b in cls.KNOWN_VERB_ROOTS), None)
        if hit is None:
            continue
        return {"stem": hit, "case": "NOM", "is_verb": True, "verbal_noun": kind, "tense": None}
    return None


def analyze_copula_dir(cls, w: str) -> dict | None:
    """Koşaç -DIr: 'öğretmendir', 'hastadır', 'soğuktur'; çekimli fiilde 'geliyordur'."""
    # 'çökertir' / 'başlatır': -tIr dizisi KOŞAÇ mı, ETTİRGEN mi? Sentez karar verir.
    if w.endswith(("tir", "tır", "tur", "tür")):
        _v = cls.analyze_voice(w) if hasattr(cls, "analyze_voice") else {}
        if _v.get("is_verb"):
            return {"stem": _v.get("voice_stem") or _v.get("stem"), "case": "NOM", "is_verb": True,
                    "tense": _v.get("tense"), "person": _v.get("person")}
    for suf in ("dir", "dır", "dur", "dür", "tir", "tır", "tur", "tür"):
        if not w.endswith(suf) or len(w) - len(suf) < 3:
            continue
        base = w[: -len(suf)]
        if base in cls.KNOWN_VERB_ROOTS:
            return None      # 'başlatır' = ettirgen FİİL (başlat+ır); çıplak köke koşaç gelmez
        fin = cls._finite_verb(base)
        if fin is not None:
            return {**fin, "copula": True}
        if base in cls.KNOWN_VERB_ROOTS:
            return {"stem": base, "case": "NOM", "is_verb": True, "tense": "AOR",
                    "person": "3SG", "copula": True}
        if base in STATE_ADJECTIVES or base in PROPER_NOUNS:
            return {"stem": base, "case": "NOM", "is_verb": False, "copula": True}
        info = cls._analyze_word_raw(base, None)
        if info.get("is_verb") and not info.get("tense") and not info.get("converb"):
            return None      # 'başlatır' = ettirgen fiil, koşaç değil
        if info.get("stem") in cls.KNOWN_VERB_ROOTS and not info.get("is_verb"):
            # 'hasta' -> 'has'+LOC değil: hal eki FİİL köküne gelmez, sözcük bütündür
            return {"stem": base, "case": "NOM", "is_verb": False, "copula": True}
        if not info.get("is_verb") and info.get("stem") and info.get("stem") != base:
            return {**info, "copula": True}
        if not info.get("is_verb"):
            return {**info, "stem": base, "copula": True}
    return None


def analyze_rel_ki(cls, w: str) -> dict | None:
    """İlgi -ki: 'evdeki', 'akşamki', 'dünkü', 'benimki'."""
    for suf in ("ki", "kü"):
        if not w.endswith(suf) or len(w) - len(suf) < 3:
            continue
        base = w[: -len(suf)]
        hard = base[:-1] + {"k": "ğ", "ç": "c", "t": "d"}.get(base[-1:], base[-1:]) if base[-1:] in "kçt" else base
        for cand in (base, hard):
            info = cls.analyze_word(cand, None)
            if info.get("stem") and info["stem"] != cand or info.get("case", "NOM") != "NOM":
                return {**info, "rel_ki": True}
            if info.get("stem") == cand and len(cand) >= 3 and not info.get("is_verb"):
                return {**info, "rel_ki": True}       # 'dün'+kü, 'akşam'+kı
        return None
    return None


def analyze_ca(cls, w: str) -> dict | None:
    """-CA eki: 'bence', 'insanca', 'hızlıca', 'Türkçe'."""
    if w.endswith(("dıkça", "dikçe", "dukça", "dükçe", "tıkça", "tikçe", "tukça", "tükçe")):
        return None          # -DIkçA bir ulaçtır (-CA türetimi değil)
    for suf in ("ca", "ce", "ça", "çe"):
        if not w.endswith(suf) or len(w) - len(suf) < 3:
            continue
        root = w[: -len(suf)]
        if cls.derive_noun(root, "ce") == w:
            return {"stem": root, "case": "NOM", "is_verb": False, "derivation": "CA"}
    return None


# ============================================================================ BAĞLA
def attach(morph_cls) -> None:
    """Yetenekleri TurkishMorphology sınıfına bağlar (tek giriş noktası)."""
    morph_cls.COMPOUND_MAP = COMPOUND_MAP
    morph_cls.compound_tense = classmethod(compound_tense)
    morph_cls.voice_stem = classmethod(voice_stem)
    morph_cls.participle_form = classmethod(participle_form)
    morph_cls.that_clause = classmethod(that_clause)
    morph_cls.derive_noun = classmethod(derive_noun)
    morph_cls.analyze_voice = classmethod(analyze_voice)
    morph_cls.analyze_compound_tense = classmethod(analyze_compound_tense)
    morph_cls.int_to_words_full = classmethod(int_to_words_full)
    morph_cls.ordinal_tr = classmethod(ordinal_tr)
    morph_cls.decimal_tr = classmethod(decimal_tr)
    morph_cls.ability_stem = classmethod(ability_stem)
    morph_cls.ability_form = classmethod(ability_form)
    morph_cls.analyze_ability = classmethod(analyze_ability)
    morph_cls.imperative_form = classmethod(imperative_form)
    morph_cls.analyze_imperative = classmethod(analyze_imperative)
    morph_cls.progressive_form = classmethod(progressive_form)
    morph_cls.analyze_progressive = classmethod(analyze_progressive)
    morph_cls.analyze_verbal_noun = classmethod(analyze_verbal_noun)
    morph_cls.analyze_copula_dir = classmethod(analyze_copula_dir)
    morph_cls.analyze_rel_ki = classmethod(analyze_rel_ki)
    morph_cls.analyze_ca = classmethod(analyze_ca)

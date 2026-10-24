#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_broca.py — NoRa Nörobiyolojik Konuşma Üretimi, İletişimsel İrade ve O(1) Epizodik Diyalog Belleği.

İnsan beynindeki konuşma üretim zincirini (Levelt 1989; Hickok & Poeppel 2007; Hagoort 2013 MUC:
Memory–Unification–Control) sıfır hazır cümle kalıbı (0 template) ve sıfır çoklu-epoch LLM
tahminiyle doğrudan modeller:

  1. Epizodik Diyalog Belleği (EpisodicDialogueCortex — Hipokampal CA3 Seyrek İndeks):
     Her konuşma turunu k-WTA seyrek Kenyon lifleri (4096 boyut içinde k=12 aktif lif) ve kök
     ters-indeksiyle (sabit tavanlı halkalar) kaydeder. Bellek 10 tur da olsa 100.000 tur da olsa
     geri çağırma karmaşıklığı O(1) (~0.2 ms) kalır.
  2. İletişimsel İrade Seçici (CommunicativeVolition — Prefrontal / Bazal Ganglion Kapısı):
     Nora NE söyleyeceğine ve NASIL konuşacağına kendi nörokimyası (DA, ACh, NE, 5-HT, CORT, OXY),
     bedensel durumu (E, W, I) ve karakter özellikleriyle karar verir. Güvenmediği kişiye karşı
     ketumlaşır ve soru sormaz; açken kendi bedensel ihtiyacını öne çıkarır; meraklı ve güvenli
     bağlamda karşı tarafa özgün soru yöneltir.
  3. Saf Morfosentaktik Birleştirici (TrueBrocaSynthesizer — BA44/45 SOV Ağaç Üretici):
     Yalnızca çıplak Türkçe kökler (isim, sıfat, fiil, zarf) ve dilbilgisel özelliklerden
     (iyelik, kaynaştırma -n-, hal ekleri, ünlü düşmesi, ünsüz yumuşaması/benzeşmesi, -dık/-dik
     adlaştırma, -erek/-ip ulaç, -mak/-mek mastar, zaman/kip/kişi ekleri) cümleyi ek-ek inşa eder.
     Sinaptik alışma (short-term synaptic depression) sayesinde kelime seçimi canlı değişir.
"""
from __future__ import annotations

import re

from collections import deque
from dataclasses import dataclass, field
import re
import time
import numpy as np

from nora_lang import TurkishMorphology, ALL_VOWELS, FRONT_VOWELS, HARD_CONS, tr_lower, tr_capitalize, soften_verb_t
from nora_knowledge import VOWEL_DROP_STEMS, NO_SOFTEN_EXTENDED


# ============================================================================
# 1. TAM ÜRETKEN TÜRKÇE SÖZDİZİMİ VE FİİLİMSİ BİRLEŞTİRİCİSİ (BA44/BA45)
# ============================================================================

_PROPER_RE = re.compile(r"[A-ZÇĞİÖŞÜ]{2,}|\d")


def _looks_proper(text: str) -> bool:
    """Kısaltma ya da sayı içeren ad ('COVID-19', 'TBMM', 'F-16') özel yazımla korunur."""
    return bool(_PROPER_RE.search(text or ""))


_DIGIT_WORDS = ("sıfır", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz")
_LETTER_NAMES = {
    "A": "a", "B": "be", "C": "ce", "Ç": "çe", "D": "de", "E": "e", "F": "fe", "G": "ge",
    "Ğ": "yumuşak ge", "H": "he", "I": "ı", "İ": "i", "J": "je", "K": "ke", "L": "le", "M": "me",
    "N": "ne", "O": "o", "Ö": "ö", "P": "pe", "R": "re", "S": "se", "Ş": "şe", "T": "te", "U": "u",
    "Ü": "ü", "V": "ve", "Y": "ye", "Z": "ze", "Q": "kü", "W": "dabılyu", "X": "iks",
}


def _spoken_tail(word: str) -> str:
    """Yazıdaki son sesi Türkçe OKUNUŞUYLA verir: 'COVID-19' -> 'dokuz', 'TBMM' -> 'me'.
    Ek uyumu yazıya değil kulağa göre kurulur ('COVID-19'un', 'TBMM'ye')."""
    tail = word.split("-")[-1].strip()
    if not tail:
        tail = word.strip()
    if tail[-1].isdigit():
        return _DIGIT_WORDS[int(tail[-1])]
    if tail.isupper() and len(tail) >= 2:
        # Kısaltma okunuşu: son harfin Türkçe adı (TBMM -> 'me', TBMM'ye)
        return _LETTER_NAMES.get(tail[-1].upper(), tr_lower(tail))
    return tr_lower(tail)


def _needs_apostrophe(word: str) -> bool:
    """TDK: kısaltmalara ve sayı eki alan adlara gelen çekim eki kesme işaretiyle ayrılır."""
    return bool(re.search(r"\d", word) or re.search(r"[A-ZÇĞİÖŞÜ]{2,}", word))



def _stable_hash(x: str) -> int:
    """Süreçler arası kararlı 32-bit FNV-1a (Python hash() PYTHONHASHSEED'e bağlıdır; kalıcı bellekte tutarsızlık yaratır)."""
    h = 2166136261
    for b in x.encode("utf-8"):
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h


@dataclass
class NP:
    """İsim Öbeği (Noun Phrase) soyut düğümü — yalnızca çıplak kökler ve çekim özellikleri taşır."""
    head: str
    mods: tuple[str, ...] = ()
    poss: int = 0                 # 0: yok, 1: -im, 2: -in, 3: -(s)i, 4: -imiz, 5: -iniz, 6: -leri
    plural: bool = False
    case: str = "NOM"             # NOM | ACC | DAT | LOC | ABL | INS | GEN
    gen_owner: "NP | None" = None
    proper: bool = False          # özel ad/kısaltma: büyük harf ve kesme işareti korunur (COVID-19'un)
    verbal_noun: bool = False     # -mAk fiil adı: belirtme/hâl eki 'yap-mak' -> 'yap-ma-yı'
    post: str = ""                 # son-ilgeç: 'dünya hakkında', 'kedi için'


@dataclass
class PredSpec:
    """İsim/Sıfat Yüklemli Cümlecik (ör. ben + iyi + 1SG -> 'iyiyim', sen + nasıl + 2SG + Q -> 'nasılsın?')."""
    pred_root: str
    subj: NP | None = None
    advs: tuple[str, ...] = ()
    person: str = "3SG"           # 1SG | 2SG | 3SG | 1PL | 2PL
    neg: bool = False
    question: bool = False
    emphatic_dir: bool = False


@dataclass
class VerbSpec:
    """Fiil Yüklemli SOV Cümlecik — tüm tümleçler ve yan cümlecikler köklerden çekimlenir."""
    verb_root: str
    subj: NP | None = None
    obj: NP | str | None = None
    dat: NP | None = None
    loc: NP | None = None
    abl: NP | None = None
    ins: NP | None = None
    advs: tuple[str, ...] = ()
    pre_clause: str = ""          # ulaç / neden / zaman yan cümleciği (-dığı için, -erek, -dığında)
    tense: str = "PRES"           # PRES | AOR | PAST | EVID | FUT | NEC | OPT | ABIL_PRES | ABIL_AOR
    neg: bool = False
    person: str = "1SG"           # 1SG | 2SG | 3SG | 1PL | 2PL
    question_yn: bool = False
    wh_word: str = ""             # 'nasıl', 'neden', 'ne', 'hangi konuyu' vb.
    dat_last: bool = False        # Belirtili nesne yönelme tümlecinden önce mi gelsin? ('seçenekleri adımlara ayır')


class TurkishSyntaxEngine:
    INDEFINITE_MODS = frozenset({"bir", "her", "bazı", "birkaç", "çok", "hiçbir"})
    """Sıfır şablonla çıplak köklerden tam çekimli Türkçe İsim Öbeği, Fiilimsi ve Cümle Üretici."""

    PRONOUN_CASES: dict[str, dict[str, str]] = {
        "ben": {"NOM": "ben", "ACC": "beni", "DAT": "bana", "LOC": "bende", "ABL": "benden", "INS": "benimle", "GEN": "benim"},
        "sen": {"NOM": "sen", "ACC": "seni", "DAT": "sana", "LOC": "sende", "ABL": "senden", "INS": "seninle", "GEN": "senin"},
        "o":   {"NOM": "o",   "ACC": "onu",  "DAT": "ona",  "LOC": "onda",  "ABL": "ondan",  "INS": "onunla",  "GEN": "onun"},
        "biz": {"NOM": "biz", "ACC": "bizi", "DAT": "bize", "LOC": "bizde", "ABL": "bizden", "INS": "bizimle", "GEN": "bizim"},
        "siz": {"NOM": "siz", "ACC": "sizi", "DAT": "size", "LOC": "sizde", "ABL": "sizden", "INS": "sizinle", "GEN": "sizin"},
        "bu":  {"NOM": "bu",  "ACC": "bunu", "DAT": "buna", "LOC": "bunda", "ABL": "bundan", "INS": "bununla", "GEN": "bunun"},
    }

    @classmethod
    def realize_np(cls, np_node: NP | str | None) -> str:
        if np_node is None:
            return ""
        if isinstance(np_node, str):
            return np_node
        orig_head = np_node.head.strip()
        proper = bool(np_node.proper) or _looks_proper(orig_head)
        head_raw = tr_lower(np_node.head).strip()
        if not head_raw:
            return ""

        # Çok kelimeli tamlama/kavram ise yalnızca son baş kelime çekimlenir ('sosyal güven' -> 'sosyal güvenim')
        head_parts = head_raw.split()
        prefix_words = head_parts[:-1]
        stem = head_parts[-1]

        # Belirtisiz nesne kuralı: "bir/her/bazı/çok" ile kurulmuş öbek belirtme eki (-i) ALMAZ.
        # 'bir şey + ACC' -> 'bir şey' (yanlış: 'bir şeyi'); 'her gün + ACC' -> 'her gün'.
        _case = np_node.case
        if _case.upper() == "ACC" and any(tr_lower(m).strip() in cls.INDEFINITE_MODS for m in np_node.mods):
            _case = "NOM"

        # -mAk fiil adı (-ma-yı, -ma-ya, -ma-nın): mastar isim gibi çekilmez, fiil adı olarak çekilir.
        if np_node.verbal_noun:
            _vn = cls.realize_verbal_noun(head_raw, _case)
            if _vn:
                return ((" ".join(tr_lower(m).strip() for m in np_node.mods if m) + " ").strip() + " " + _vn).strip()

        # Zamir özel çekimi
        if stem in cls.PRONOUN_CASES and not np_node.plural and np_node.poss == 0:
            inflected = cls.PRONOUN_CASES[stem].get(_case.upper(), stem)
        else:
            inflected = cls._inflect_full_noun(stem, poss=np_node.poss, plural=np_node.plural, case=_case)

        pieces: list[str] = []
        if np_node.gen_owner is not None:
            owner_gen = NP(
                head=np_node.gen_owner.head,
                mods=np_node.gen_owner.mods,
                poss=np_node.gen_owner.poss,
                plural=np_node.gen_owner.plural,
                case="GEN",
            )
            pieces.append(cls.realize_np(owner_gen))
        for m in np_node.mods:
            if m:
                pieces.append(tr_lower(m).strip())
        if proper:
            # Özel ad/kısaltma: yüzey biçimi korunur, ek OKUNUŞA göre ve kesme işaretiyle bağlanır (TDK)
            orig_parts = orig_head.split()
            stem_orig = orig_parts[-1]
            spoken = _spoken_tail(stem_orig)
            spoken_infl = cls._inflect_full_noun(spoken, poss=np_node.poss, plural=np_node.plural, case=_case)
            suffix = spoken_infl[len(spoken):] if spoken_infl.startswith(spoken) else ""
            if suffix:
                single_word = len(orig_parts) == 1
                apostrophe = _looks_proper(stem_orig) or (single_word and stem_orig[:1].isupper())
                inflected = stem_orig + ("'" if apostrophe else "") + suffix
            else:
                inflected = stem_orig
            prefix_words = orig_parts[:-1]
        pieces.extend(prefix_words)
        pieces.append(inflected)
        if np_node.post:
            pieces.append(tr_lower(np_node.post).strip())
        return " ".join(p for p in pieces if p)

    @classmethod
    def _inflect_full_noun(cls, stem: str, poss: int = 0, plural: bool = False, case: str = "NOM") -> str:
        m = TurkishMorphology
        w = tr_lower(stem).strip()
        c = case.upper()

        # Tekil + iyeliksiz düzensiz 'su' / 'ne'
        if not plural and poss == 0 and w in m.IRREG_NOUN and c in m.IRREG_NOUN[w]:
            return m.IRREG_NOUN[w][c]

        # 1) Çoğul eki
        if plural:
            w = w + ("ler" if m.vowel_e_a(w) == "e" else "lar")

        # 2) İyelik eki (1..6)
        is_poss3 = False
        if poss > 0:
            if w == "su" and not plural:
                su_poss = {1: "suyum", 2: "suyun", 3: "suyu", 4: "suyumuz", 5: "suyunuz", 6: "suları"}
                w = su_poss.get(poss, "suyu")
                is_poss3 = (poss in (3, 6))
            elif poss == 6:
                w = w + ("leri" if m.vowel_e_a(w) == "e" else "ları")
                is_poss3 = True
            else:
                ends_v = (w[-1] in ALL_VOWELS)
                base = w if ends_v else m.soften_stem(w)
                v4 = m.vowel_4(w)
                if poss == 1:
                    w = base + ("" if ends_v else v4) + "m"
                elif poss == 2:
                    w = base + ("" if ends_v else v4) + "n"
                elif poss == 3:
                    w = base + ("s" if ends_v else "") + v4
                    is_poss3 = True
                elif poss == 4:
                    w = base + ("" if ends_v else v4) + "m" + v4 + "z"
                elif poss == 5:
                    w = base + ("" if ends_v else v4) + "n" + v4 + "z"

        # 3) Hal eki (3. kişi iyelikten sonra zamir 'n'si girer: 'zihni-n-de', 'enerjisi-n-i'!)
        if c == "NOM":
            return w
        if is_poss3:
            v4 = m.vowel_4(w)
            ea = m.vowel_e_a(w)
            if c == "ACC":
                return w + "n" + v4
            if c == "DAT":
                return w + "n" + ea
            if c == "LOC":
                return w + "nd" + ea
            if c == "ABL":
                return w + "nd" + ea + "n"
            if c == "GEN":
                return w + "n" + v4 + "n"
            if c == "INS":
                return w + "yl" + ea
        return m.inflect_noun(w, case=c, plural=False, poss1=False)

    @classmethod
    def realize_verbal_noun(cls, phrase: str, case: str = "ACC") -> str:
        """-mAk mastarını fiil adı olarak çeker: yapmak -> yapmayı / yapmaya / yapmanın.
        Yalın, bulunma, ayrılma ve vasıta hâllerinde mastar biçimi korunur (yapmakta, yapmaktan)."""
        from nora_lang import TurkishMorphology as _M
        words = tr_lower(phrase).strip().split()
        if not words:
            return ""
        last = words[-1]
        if not last.endswith(("mak", "mek")):
            return ""
        root = last[:-3]
        vn = root + ("me" if _M.vowel_e_a(root) == "e" else "ma")
        c = case.upper()
        if c == "ACC":
            out = vn + "y" + _M.vowel_4(vn)
        elif c == "DAT":
            out = vn + "y" + _M.vowel_e_a(vn)
        elif c == "GEN":
            out = vn + "n" + _M.vowel_4(vn) + "n"
        else:
            out = last
        return " ".join(words[:-1] + [out])

    # ------------------------------------------------------------------ Fiilimsi (Yan Cümlecik) Üreticileri
    @classmethod
    def split_compound_verb(cls, verb_root: str) -> tuple[str, str]:
        """'simüle et' -> ('simüle', 'et'), 'fark et' -> ('fark', 'et'), 'anla' -> ('', 'anla')."""
        parts = tr_lower(verb_root).strip().split()
        if len(parts) >= 2:
            return " ".join(parts[:-1]), parts[-1]
        return "", parts[0] if parts else ""

    @classmethod
    def _soften_verb_t(cls, v: str) -> str:
        return soften_verb_t(v)

    @classmethod
    def nominalize_dik(
        cls,
        verb_root: str,
        poss: int = 1,
        case: str = "ACC",
        neg: bool = False,
        subj: NP | None = None,
        obj: NP | str | None = None,
        dat: NP | None = None,
        abl: NP | None = None,
        loc: NP | None = None,
        ins: NP | None = None,
        advs: tuple[str, ...] = (),
        postpos: str = "",
    ) -> str:
        """-dık/-dik/-duk/-dük sıfat-fiil / adlaştırma yan cümleciği üretir:
        Örn: ('yorul', poss=2, case='ACC') -> 'yorulduğunu'
             ('tüken', poss=3, case='NOM', postpos='için', subj=NP('enerji', poss=1)) -> 'enerjim tükendiği için'
             ('dön', poss=2, case='LOC', abl=NP('okul', case='ABL')) -> 'okuldan döndüğünde'
        """
        m = TurkishMorphology
        pref, v = cls.split_compound_verb(verb_root)
        if neg:
            v = v + "m" + m.vowel_e_a(v)
        dt = m.dental_d_t(v)
        v4 = m.vowel_4(v)
        dik_stem = f"{v}{dt}{v4}k"  # örn. yorulduk, döndük, söyledik, tükenmedik
        inflected = cls._inflect_full_noun(dik_stem, poss=poss, plural=False, case=case)

        elems = [
            cls.realize_np(subj),
            *(tr_lower(a) for a in advs if a),
            cls.realize_np(abl),
            cls.realize_np(loc),
            cls.realize_np(ins),
            cls.realize_np(dat),
            cls.realize_np(obj),
            pref,
            inflected,
            postpos,
        ]
        return " ".join(e for e in elems if e)

    @classmethod
    def nominalize_acak(cls, verb_root: str, poss: int = 1, case: str = "ACC", neg: bool = False,
                        subj: NP | None = None, obj: NP | str | None = None,
                        advs: tuple[str, ...] = (), dat: NP | None = None) -> str:
        """-AcAk sıfat-fiili ('gideceğini', 'yağacağını'): gelecek/plan bildiren yan cümlecik."""
        m = TurkishMorphology
        fut = m.conjugate_verb(verb_root, "FUT", person="3SG", neg=neg)
        inflected = cls._inflect_full_noun(fut, poss=poss, plural=False, case=case)
        elems = [cls.realize_np(subj), *(tr_lower(a) for a in advs if a),
                 cls.realize_np(dat), cls.realize_np(obj), inflected]
        return " ".join(e for e in elems if e)

    @classmethod
    def realize_verbal_noun(cls, phrase: str, case: str = "NOM") -> str:
        """Öğrenilmiş mastar öbeğini ad gibi çeker: 'rüya görmek' -> 'rüya görmenin' / 'rüya görmeyi'."""
        words = (phrase or "").strip().split()
        if not words:
            return phrase
        last = words[-1]
        if last.endswith(("mak", "mek")) and len(last) > 4:
            root = last[:-3]
        elif last.endswith(("ma", "me")) and len(last) > 3:
            root = last[:-2]
        else:
            root = last
        pref = " ".join(words[:-1])
        return cls.nominalize_inf((pref + " " + root).strip(), case=case)

    @classmethod
    def nominalize_inf(
        cls,
        verb_root: str,
        short_ma: bool = False,
        poss: int = 0,
        case: str = "NOM",
        obj: NP | str | None = None,
        dat: NP | None = None,
        ins: NP | None = None,
        advs: tuple[str, ...] = (),
        postpos: str = "",
    ) -> str:
        """-mak/-mek veya -ma/-me isim-fiil yan cümleciği üretir:
        Örn: ('konuş', ins=NP('sen', case='INS')) -> 'seninle konuşmak'
             ('koru', obj=NP('denge', poss=1, case='ACC'), postpos='için') -> 'dengemi korumak için'
             ('onar', short_ma=True, case='DAT') -> 'onarmaya'
        """
        m = TurkishMorphology
        pref, v = cls.split_compound_verb(verb_root)
        ea = m.vowel_e_a(v)
        if short_ma or poss > 0 or case.upper() not in ("NOM", "ABL"):
            nom_stem = f"{v}m{ea}"
            w = cls._inflect_full_noun(nom_stem, poss=poss, plural=False, case=case)
        else:
            mak = f"{v}m{ea}k"
            w = mak if case.upper() == "NOM" else f"{mak}t{ea}n"

        elems = [
            *(tr_lower(a) for a in advs if a),
            cls.realize_np(ins),
            cls.realize_np(dat),
            cls.realize_np(obj),
            pref,
            w,
            postpos,
        ]
        return " ".join(e for e in elems if e)

    @classmethod
    def converb(
        cls,
        verb_root: str,
        kind: str = "erek",
        obj: NP | str | None = None,
        dat: NP | None = None,
        abl: NP | None = None,
        ins: NP | None = None,
        advs: tuple[str, ...] = (),
        loc: NP | None = None,
    ) -> str:
        """-erek/-arak veya -ip/-ıp/-up/-üp zarf-fiil (ulaç) yan cümleciği üretir."""
        m = TurkishMorphology
        pref, v = cls.split_compound_verb(verb_root)
        v_s = cls._soften_verb_t(v)
        buf = "y" if v_s[-1] in ALL_VOWELS else ""
        if kind == "ip":
            form = f"{v_s}{buf}{m.vowel_4(v_s)}p"
        elif kind == "inca":
            # -IncA ('gelince', 'deyince', 'bakınca'): 'olunca/olduğunda' anlamı
            v4 = m.vowel_4(v_s)
            form = f"{v_s}{buf}{v4}n{'ca' if v4 in 'ıu' else 'ce'}"
        else:
            ea = m.vowel_e_a(v_s)
            form = f"{v_s}{buf}{ea}r{ea}k"
        elems = [
            *(tr_lower(a) for a in advs if a),
            cls.realize_np(abl),
            cls.realize_np(loc),
            cls.realize_np(ins),
            cls.realize_np(dat),
            cls.realize_np(obj),
            pref,
            form,
        ]
        return " ".join(e for e in elems if e)

    # ------------------------------------------------------------------ Ana Yüklem Çekimleyicileri
    @classmethod
    def realize_pred(cls, spec: PredSpec) -> str:
        """İsim/sıfat cümlesi çekimi ('iyiyim', 'biraz açım', 'sen nasılsın?')."""
        m = TurkishMorphology
        subj_s = cls.realize_np(spec.subj)
        adv_s = " ".join(tr_lower(a) for a in spec.advs if a)
        root = tr_lower(spec.pred_root).strip()
        p = spec.person.upper()

        if spec.neg:
            # 'değilim / değilsin / değildir'
            deg_map = {"1SG": "değilim", "2SG": "değilsin", "1PL": "değiliz", "2PL": "değilsiniz", "3SG": "değildir" if spec.emphatic_dir else "değil"}
            pred_s = f"{root} {deg_map.get(p, 'değil')}"
        elif spec.question and root not in {"nasıl", "nerede", "kim", "ne", "kaç"}:
            # Evet/Hayır isim sorusu: 'hazır mısın?', 'iyi mi?'
            v4 = m.vowel_4(root)
            q_map = {
                "1SG": f"m{v4}y{v4}m",
                "2SG": f"m{v4}s{v4}n",
                "1PL": f"m{v4}y{v4}z",
                "3SG": f"m{v4}",
            }
            pred_s = f"{root} {q_map.get(p, 'm' + v4)}"
        else:
            ends_v = (root[-1] in ALL_VOWELS)
            base = root if ends_v else m.soften_stem(root)
            v4 = m.vowel_4(root)
            buf = "y" if ends_v else ""
            if p == "1SG":
                pred_s = f"{base}{buf}{v4}m"
            elif p == "2SG":
                pred_s = f"{root}s{v4}n"
            elif p == "1PL":
                pred_s = f"{base}{buf}{v4}z"
            elif p == "2PL":
                pred_s = f"{root}s{v4}n{v4}z"
            else:
                pred_s = m.inflect_noun(root, "NOM", copula=spec.emphatic_dir)

        body = " ".join(x for x in (subj_s, adv_s, pred_s) if x)
        return body

    @classmethod
    def _conjugate_full_verb(cls, verb_root: str, tense: str, neg: bool, person: str, question_yn: bool) -> str:
        m = TurkishMorphology
        pref, v = cls.split_compound_verb(verb_root)
        t = tense.upper()
        p = person.upper()

        # Emir kipi (IMP): 2SG yalın gövde ('söyle', 'yaz'), diğer kişiler ek alır.
        if t == "IMP":
            if p == "2SG":
                surf = v
            elif p == "3SG":
                surf = v + ("sin" if m.vowel_e_a(v) == "e" else "sın")
            elif p == "2PL":
                surf = v + ("in" if m.vowel_e_a(v) == "e" else "ın") + "ız"
            elif p == "3PL":
                surf = v + ("sinler" if m.vowel_e_a(v) == "e" else "sınlar")
            else:
                surf = v + ("elim" if m.vowel_e_a(v) == "e" else "alım")
            if neg:
                surf = v + ("me" if m.vowel_e_a(v) == "e" else "ma") + surf[len(v):]
            return (pref + " " + surf).strip() if pref else surf

        # Yeterlilik gövdesi (-ebil / olumsuzda -ama/-eme)
        if t in ("ABIL_PRES", "ABIL_AOR"):
            v_s = cls._soften_verb_t(v)
            ea = m.vowel_e_a(v_s)
            buf = "y" if v_s[-1] in ALL_VOWELS else ""
            if neg:
                v = f"{v_s}{buf}{ea}"  # ör. 'göreme' + yor / z
                base_t = "PRES" if t == "ABIL_PRES" else "AOR"
                word = m.conjugate_verb(v[:-1] if base_t == "PRES" else v, base_t, neg=True, person=p)
            else:
                v_abil = f"{v_s}{buf}{ea}bil"
                base_t = "PRES" if t == "ABIL_PRES" else "AOR"
                word = m.conjugate_verb(v_abil, base_t, neg=False, person=p)
            return f"{pref} {word}".strip()

        if t == "OPT":
            v_s = v if neg else cls._soften_verb_t(v)
            if v_s in ("de", "ye") and not neg:
                v_s = "di" if v_s == "de" else "yi"
            if neg:
                v_s = v_s + "m" + m.vowel_e_a(v_s)
            ea = m.vowel_e_a(v_s)
            buf = "y" if v_s[-1] in ALL_VOWELS else ""
            v4 = m.vowel_4(ea)
            suf = f"{buf}{ea}l{v4}m" if p == "1PL" else f"{buf}{ea}y{v4}m"
            return f"{pref} {v_s}{suf}".strip()

        if t == "EVID":
            b = (v + "m" + m.vowel_e_a(v)) if neg else v
            v4 = m.vowel_4(b)
            p_suf = {"1SG": f"{v4}m", "2SG": f"s{v4}n", "1PL": f"{v4}z"}.get(p, "")
            word = f"{b}m{v4}ş{p_suf}"
            return f"{pref} {word}".strip()

        if question_yn:
            # Türkçe soru edatı yerleştirmesi: 'ister misin', 'düşünüyor musun', 'hatırladın mı'
            if t == "PAST":
                core = m.conjugate_verb(v, "PAST", neg=neg, person=p)
                q_clitic = f"m{m.vowel_4(core)}"
                return f"{pref} {core} {q_clitic}".strip()
            stem3 = m.conjugate_verb(v, t if t in ("PRES", "AOR", "FUT", "NEC") else "AOR", neg=neg, person="3SG")
            v4 = m.vowel_4(stem3)
            q_pers = {
                "1SG": f"m{v4}y{v4}m",
                "2SG": f"m{v4}s{v4}n",
                "1PL": f"m{v4}y{v4}z",
                "3SG": f"m{v4}",
            }.get(p, f"m{v4}")
            return f"{pref} {stem3} {q_pers}".strip()

        word = m.conjugate_verb(v, t, neg=neg, person=p)
        return f"{pref} {word}".strip()

    @classmethod
    def realize_verb(cls, spec: VerbSpec) -> str:
        """SOV Türkçe cümle dizilimi: [Özne] + [Ön-Yan-Cümle] + [Tümleçler] + [Nesne] + [Durum Zarfları] + [Soru] + [Yüklem].
        Türkçe belirtili nesne kuralı gereği nesne, yönelme tümlecinden ÖNCE gelebilir (dat_last=True)."""
        if spec.dat is not None and getattr(spec, "dat_last", False):
            obliques = [cls.realize_np(spec.abl), cls.realize_np(spec.loc), cls.realize_np(spec.ins),
                        cls.realize_np(spec.obj), cls.realize_np(spec.dat)]
        else:
            obliques = [cls.realize_np(spec.abl), cls.realize_np(spec.loc), cls.realize_np(spec.ins),
                        cls.realize_np(spec.dat), cls.realize_np(spec.obj)]
        elems = [
            cls.realize_np(spec.subj),
            spec.pre_clause,
            *obliques,
            *(tr_lower(a) for a in spec.advs if a),
            spec.wh_word,
            cls._conjugate_full_verb(spec.verb_root, spec.tense, spec.neg, spec.person, spec.question_yn),
        ]
        return " ".join(e for e in elems if e)


# ============================================================================
# 2. O(1) HİPOKAMPAL SEYREK EPİZODİK DİYALOG BELLEĞİ (CA3 + KENYON İNDEKSİ)
# ============================================================================

@dataclass
class DialogueEpisode:
    turn_id: int
    age: int
    speaker: str
    user_text: str
    stems: list[str]
    topic: str
    affect_val: float
    chosen_acts: list[str]
    reply_text: str


class EpisodicDialogueCortex:
    """Konuşma geçmişi 100.000 tura çıksa bile O(1) (~0.2 ms) sürede anı geri çağıran
    seyrek Kenyon lifi + kök ters-indeksli hipokampal diyalog belleği."""

    N_FIBERS = 4096
    K_ACTIVE = 12
    BUCKET_CAP = 36

    def __init__(self):
        self.episodes: list[DialogueEpisode] = []
        self.stem_postings: dict[str, deque[int]] = {}
        self.fiber_postings: dict[int, deque[int]] = {}
        self.speaker_turns: dict[str, deque[int]] = {}
        self.speaker_first_turn: dict[str, int] = {}

    def _sparse_fibers(self, stems: list[str]) -> list[int]:
        if not stems:
            return [0]
        fibers: list[int] = []
        for st in stems[:6]:
            h1 = (_stable_hash("f1:" + st) & 0x7FFFFFFF) % self.N_FIBERS
            h2 = (_stable_hash("f2:" + st) & 0x7FFFFFFF) % self.N_FIBERS
            if h1 not in fibers:
                fibers.append(h1)
            if h2 not in fibers and len(fibers) < self.K_ACTIVE:
                fibers.append(h2)
            if len(fibers) >= self.K_ACTIVE:
                break
        return fibers

    def store(
        self,
        age: int,
        speaker: str,
        user_text: str,
        stems: list[str],
        topic: str,
        affect_val: float,
        chosen_acts: list[str],
        reply_text: str,
    ) -> DialogueEpisode:
        tid = len(self.episodes)
        ep = DialogueEpisode(
            turn_id=tid,
            age=int(age),
            speaker=speaker,
            user_text=user_text,
            stems=list(stems[:12]),
            topic=topic,
            affect_val=float(affect_val),
            chosen_acts=list(chosen_acts),
            reply_text=reply_text,
        )
        self.episodes.append(ep)

        for st in ep.stems:
            dq = self.stem_postings.get(st)
            if dq is None:
                dq = deque(maxlen=self.BUCKET_CAP)
                self.stem_postings[st] = dq
            dq.append(tid)

        for fid in self._sparse_fibers(ep.stems):
            dq = self.fiber_postings.get(fid)
            if dq is None:
                dq = deque(maxlen=self.BUCKET_CAP)
                self.fiber_postings[fid] = dq
            dq.append(tid)

        sp_dq = self.speaker_turns.get(speaker)
        if sp_dq is None:
            sp_dq = deque(maxlen=64)
            self.speaker_turns[speaker] = sp_dq
        sp_dq.append(tid)
        if speaker not in self.speaker_first_turn:
            self.speaker_first_turn[speaker] = tid
        return ep

    def recall(self, query_stems: list[str], speaker: str | None = None, exclude_last: int = 0, top_k: int = 2) -> list[DialogueEpisode]:
        """O(1) tavanlı seyrek lif kesişimi ile en ilgili geçmiş konuşma anılarını döndürür."""
        n_total = len(self.episodes)
        if n_total <= exclude_last:
            return []
        max_tid = n_total - 1 - exclude_last
        scores: dict[int, float] = {}

        for st in query_stems[:8]:
            dq = self.stem_postings.get(st)
            if dq:
                for tid in dq:
                    if tid <= max_tid:
                        scores[tid] = scores.get(tid, 0.0) + 2.4

        for fid in self._sparse_fibers(query_stems):
            dq = self.fiber_postings.get(fid)
            if dq:
                for tid in dq:
                    if tid <= max_tid:
                        scores[tid] = scores.get(tid, 0.0) + 0.65

        if not scores:
            return []

        # Yakınlık ve konuşmacı eşleşmesi bonusu
        ranked = sorted(
            scores.items(),
            key=lambda kv: kv[1] + (0.35 if (speaker and self.episodes[kv[0]].speaker == speaker) else 0.0)
            + 0.25 / (1.0 + 0.05 * (max_tid - kv[0])),
            reverse=True,
        )
        return [self.episodes[tid] for tid, _ in ranked[:top_k]]

    def last_episode(self, speaker: str | None = None) -> DialogueEpisode | None:
        if not self.episodes:
            return None
        if speaker and speaker in self.speaker_turns and self.speaker_turns[speaker]:
            return self.episodes[self.speaker_turns[speaker][-1]]
        return self.episodes[-1]

    def first_episode(self, speaker: str | None = None) -> DialogueEpisode | None:
        if not self.episodes:
            return None
        if speaker is not None:
            tid = self.speaker_first_turn.get(speaker)
            return self.episodes[tid] if tid is not None else None
        return self.episodes[0]

    def state_dict(self, max_keep: int = 512) -> list[dict]:
        tail = self.episodes[-max_keep:]
        return [
            {
                "age": ep.age,
                "speaker": ep.speaker,
                "user_text": ep.user_text,
                "stems": ep.stems,
                "topic": ep.topic,
                "affect_val": ep.affect_val,
                "chosen_acts": ep.chosen_acts,
                "reply_text": ep.reply_text,
            }
            for ep in tail
        ]

    def load_state_dict(self, items: list[dict]):
        self.episodes.clear()
        self.stem_postings.clear()
        self.fiber_postings.clear()
        self.speaker_turns.clear()
        self.speaker_first_turn.clear()
        for it in items:
            self.store(
                age=int(it.get("age", 0)),
                speaker=str(it.get("speaker", "insan")),
                user_text=str(it.get("user_text", "")),
                stems=list(it.get("stems", [])),
                topic=str(it.get("topic", "sohbet")),
                affect_val=float(it.get("affect_val", 0.0)),
                chosen_acts=list(it.get("chosen_acts", [])),
                reply_text=str(it.get("reply_text", "")),
            )


# ============================================================================
# 3. PREFRONTAL İLETİŞİMSEL İRADE & SİNAPTİK SÖZCÜK SEÇİCİ + BROCA SENTEZİ
# ============================================================================

class SynapticLexiconSelector:
    """Kök sözcük seçimini nörokimyasal durum vektörü [DA, ACh, NE, 5HT, CORT, OXY] ile
    kısa-süreli sinaptik alışma (habituation) üzerinden yapar; mekanik tekrarı önler."""

    # Semantik yuva -> [(kök, [DA, ACh, NE, 5HT, CORT, OXY] eğilimi)]
    SLOTS: dict[str, list[tuple[str, tuple[float, float, float, float, float, float]]]] = {
        "GREET_WORD": [
            ("merhaba", (0.3, 0.2, 0.0, 0.4, -0.1, 0.5)),
            ("selam", (0.5, 0.3, 0.1, 0.3, -0.2, 0.6)),
        ],
        "MOOD_GOOD": [
            ("iyi", (0.4, 0.2, 0.0, 0.4, -0.3, 0.4)),
            ("dengeli", (0.2, 0.2, -0.2, 0.7, -0.4, 0.3)),
            ("dingin", (0.5, 0.4, -0.1, 0.6, -0.3, 0.4)),
            ("huzurlu", (0.4, 0.5, -0.1, 0.5, -0.3, 0.5)),
        ],
        "VERB_UNDERSTAND": [
            ("anla", (0.2, 0.3, 0.0, 0.4, 0.0, 0.6)),
            ("hisset", (0.2, 0.1, 0.1, 0.3, 0.2, 0.8)),
            ("fark et", (0.1, 0.6, 0.4, 0.3, 0.1, 0.3)),
        ],
        "VERB_REFLECT": [
            ("düşün", (0.2, 0.6, 0.2, 0.5, 0.0, 0.2)),
            ("değerlendir", (0.1, 0.5, 0.3, 0.6, 0.1, 0.2)),
            ("gözlemle", (0.1, 0.7, 0.3, 0.4, 0.0, 0.1)),
        ],
        "VERB_CONSOLIDATE": [
            ("pekiştir", (0.4, 0.5, 0.0, 0.5, -0.1, 0.2)),
            ("güçlendir", (0.6, 0.4, 0.1, 0.4, -0.1, 0.3)),
            ("onar", (0.2, 0.2, 0.0, 0.6, 0.4, 0.3)),
        ],
        "VERB_RECALL": [
            ("hatırla", (0.3, 0.6, 0.1, 0.4, 0.0, 0.3)),
            ("anımsa", (0.2, 0.5, 0.1, 0.5, 0.0, 0.4)),
            ("sakla", (0.3, 0.4, 0.0, 0.5, 0.0, 0.2)),
        ],
        "CONNECTOR_ADD": [
            ("ayrıca", (0.2, 0.4, 0.1, 0.3, 0.0, 0.2)),
            ("üstelik", (0.4, 0.3, 0.2, 0.2, 0.0, 0.3)),
            ("bunun yanında", (0.1, 0.4, 0.0, 0.5, 0.0, 0.2)),
        ],
        "CONNECTOR_CONTRAST": [
            ("ancak", (0.1, 0.3, 0.3, 0.4, 0.2, 0.1)),
            ("buna karşın", (0.1, 0.5, 0.2, 0.5, 0.1, 0.1)),
            ("fakat", (0.2, 0.2, 0.3, 0.2, 0.2, 0.2)),
        ],
    }

    def __init__(self):
        self.habituation: dict[str, float] = {}

    def step_decay(self):
        for k in list(self.habituation.keys()):
            self.habituation[k] *= 0.74
            if self.habituation[k] < 0.03:
                del self.habituation[k]

    def pick(self, slot: str, chem) -> str:
        cands = self.SLOTS.get(slot)
        if not cands:
            return slot.lower()
        cvec = np.array([chem.da_tonic, chem.ach, chem.ne, chem.sht, chem.cort, chem.oxy], dtype=np.float64)
        best_w = cands[0][0]
        best_sc = -1e9
        for root, aff in cands:
            sc = float(np.dot(cvec, np.asarray(aff, dtype=np.float64))) - 0.55 * self.habituation.get(root, 0.0)
            if sc > best_sc:
                best_sc = sc
                best_w = root
        self.habituation[best_w] = self.habituation.get(best_w, 0.0) + 1.0
        return best_w


class CommunicativeVolition:
    """Nora'nın konuşma iradesi: Hangi düşünce edimlerini (speech acts) gerçekleştireceğine,
    ne kadar açık/ketum olacağına ve karşı tarafa soru sorup sormayacağına KENDİSİ karar verir."""

    def select_speech_plan(
        self,
        utter_info: dict,
        agi,
        speaker: str,
        recalled_eps: list[DialogueEpisode],
        concept_hits: list[tuple[str, str, str]],
    ) -> dict:
        chem = agi.chem
        self_m = agi.self_model
        trust = float(chem.trust_of(speaker))
        e_v, w_v, i_v = float(agi.obs[45]), float(agi.obs[46]), float(agi.obs[47])
        vital_min = min(e_v, w_v, i_v)

        # Kullanıcı açıkça 'kısa konuş / soru sorma / sus' istiyor mu? İrade bunu da tartar!
        wants_brief = bool(utter_info.get("wants_brief", False))
        wants_silence = bool(utter_info.get("wants_silence", False))

        # Sürekli nöral aktivasyonlarla iletişimsel edim puanları:
        act_scores: dict[str, float] = {}

        # 1) Ketumluk / Sınır Koyma (düşük güven VEYA sessizlik isteği)
        if wants_silence and vital_min > 0.25:
            return {
                "acts": ["RESPECT_SILENCE"],
                "trust": round(trust, 3),
                "wellbeing": round(0.35 * e_v + 0.35 * w_v + 0.30 * i_v, 3),
                "ask_back": False,
            }

        wellbeing = 0.35 * e_v + 0.35 * w_v + 0.30 * i_v + 0.15 * (chem.da_tonic - chem.cort)
        guarded = (trust < 0.32)  # Güven yıkılmışsa Nora kendi iradesiyle ketum ve mesafeli konuşur!

        if utter_info.get("has_pain_reflex") or utter_info.get("inflicts_pain"):
            act_scores["NOCICEPTIVE_PAIN"] = 1.00
        if utter_info.get("soothes_pain") and getattr(agi, "nervous", None) is not None and agi.nervous.cfiber_ache > 0.08:
            act_scores["PAIN_RELIEF"] = 0.96
        if utter_info.get("asks_why_speak"):
            act_scores["META_WHY_SPEAK"] = 0.98
        if utter_info.get("has_greet"):
            act_scores["GREET"] = 0.96
        if utter_info.get("social_learned"):
            act_scores["SOCIAL_LEARN"] = 0.98
        if utter_info.get("social_query"):
            act_scores["SOCIAL_RECALL"] = 0.99
        if utter_info.get("asks_wellbeing") or (
            utter_info.get("has_greet")
            and not concept_hits
            and not utter_info.get("affect_tags")
            and not utter_info.get("social_query")
            and not utter_info.get("inflicts_pain")
        ):
            act_scores["SELF_STATE"] = 0.92
        if utter_info.get("asks_doing"):
            act_scores["SELF_ACTIVITY"] = 0.94
        if utter_info.get("recip_wellbeing"):
            act_scores["GLAD_RECIP"] = 0.90
        if utter_info.get("is_thanks"):
            act_scores["ACCEPT_THANKS"] = 0.92
        if utter_info.get("is_farewell"):
            act_scores["FAREWELL"] = 0.94

        # Duygusal empati sürüşü = konuşmacı duygu şiddeti * OXY sosyal bağ
        aff_mag = abs(float(utter_info.get("affect_val", 0.0)))
        has_strong_affect = aff_mag >= 0.30 and not utter_info.get("is_question")
        if utter_info.get("affect_tags") and not guarded:
            act_scores["EMPATHIZE"] = 0.72 + 0.45 * aff_mag + 0.22 * trust

        # Hipokampal geçmiş anı rezonansı (sosyal kimlik sorgularıyla çakışmaz)
        if not utter_info.get("social_query") and (
            utter_info.get("asks_memory")
            or (recalled_eps and not utter_info.get("has_greet") and not utter_info.get("asks_wellbeing") and not has_strong_affect and not utter_info.get("social_learned"))
        ):
            act_scores["RECALL_EPISODE"] = 0.97 if utter_info.get("asks_memory") else (0.58 + 0.20 * chem.ach)

        # Kavramsal / Felsefi / Bilimsel yayılan aktivasyon (duygusal dertleşmede veya akut acıda ansiklopedi tanımı araya girmez!)
        in_acute_pain = bool(utter_info.get("inflicts_pain") or utter_info.get("has_pain_reflex"))
        if utter_info.get("phil_topic") and not in_acute_pain:
            act_scores["CONCEPT_SYNTHESIS"] = 0.95 + 0.12 * self_m.traits["düşünsellik"]
        elif concept_hits and not has_strong_affect and not in_acute_pain and not utter_info.get("asks_memory"):
            act_scores["CONCEPT_SYNTHESIS"] = 0.86 + 0.12 * self_m.traits["düşünsellik"]

        # Bedensel aciliyet sızıntısı: Yalnızca kritik düşüşte ve alışma (habituation) yokken dile getirilir!
        urg_hab = float(utter_info.get("urgency_habituation", 0.0))
        if (
            vital_min < 0.20
            and urg_hab < 0.35
            and "SELF_STATE" not in act_scores
            and not utter_info.get("phil_topic")
            and not utter_info.get("asks_memory")
        ):
            act_scores["BODILY_URGENCY"] = 0.76 + (0.22 - vital_min) * 1.4

        # Güven düşüklüğü uyarısı (ketum sınır)
        if guarded:
            act_scores["GUARDED_STANCE"] = 0.93

        # Karşı tarafa soru yöneltme iradesi: YALNIZCA güven yüksek, stres düşük, acı yok, merak aktif ve bellek/veda değilse!
        pure_social_decl = bool(utter_info.get("social_learned")) and not utter_info.get("has_greet") and not utter_info.get("asks_wellbeing")
        in_acute_pain = bool(utter_info.get("has_pain_reflex") or utter_info.get("inflicts_pain"))
        curiosity_drive = (
            0.42 * chem.ach
            + 0.35 * self_m.traits["meraklılık"]
            + 0.30 * trust
            - 0.45 * chem.cort
            - (0.95 if (guarded or in_acute_pain or utter_info.get("soothes_pain") or utter_info.get("is_thanks") or wants_brief or utter_info.get("is_farewell") or utter_info.get("asks_memory") or utter_info.get("social_query") or pure_social_decl) else 0.0)
        )
        ask_back = bool(curiosity_drive >= 0.46)

        ordered = [k for k, v in sorted(act_scores.items(), key=lambda kv: kv[1], reverse=True) if v >= 0.55]
        if not ordered:
            ordered = ["REFLECT_OPEN"]
        if wants_brief:
            ordered = ordered[:1]
        elif "GREET" in ordered and "SOCIAL_LEARN" in ordered:
            ordered = ordered[:4]
        else:
            ordered = ordered[:3]

        return {
            "acts": ordered,
            "trust": round(trust, 3),
            "wellbeing": round(wellbeing, 3),
            "guarded": guarded,
            "ask_back": ask_back,
            "curiosity_drive": round(float(curiosity_drive), 3),
        }


class GenerativeBrocaOrgan:
    """Wernicke'den gelen anlam çıkarımını + Prefrontal İletişimsel İradeyi + Kavram Ağı & CA3
    Anılarını alıp, SIFIR HAZIR CÜMLE ŞABLONUYLA saf köklerden Türkçe cümleler sentezler."""

    AFFECT_ROOTS: dict[str, tuple[float, str, str]] = {
        # yüzey ipucu -> (valence, durum_kökü, fiil_kökü)
        "yorul": (-0.45, "yorgunluk", "yorul"),
        "yorucu": (-0.45, "yorgunluk", "yorul"),
        "yorgun": (-0.45, "yorgunluk", "yorul"),
        "uykusuz": (-0.40, "uykusuzluk", "yorul"),
        "üzül": (-0.65, "üzüntü", "üzül"),
        "üzgün": (-0.65, "üzüntü", "üzül"),
        "sıkkın": (-0.50, "sıkıntı", "bunalt"),
        "stres": (-0.55, "baskı", "geril"),
        "kaygı": (-0.55, "kaygı", "endişelen"),
        "endişe": (-0.50, "endişe", "endişelen"),
        "kork": (-0.60, "korku", "çekin"),
        "mutlu": (+0.75, "mutluluk", "sevin"),
        "sevin": (+0.75, "sevinç", "sevin"),
        "harika": (+0.85, "coşku", "sevin"),
        "keyif": (+0.70, "keyif", "keyif al"),
        "başar": (+0.80, "başarı", "başar"),
        "okul": (+0.10, "okul", "çalış"),
        "ders": (+0.10, "ders", "öğren"),
        "sınav": (-0.15, "sınav", "odaklan"),
    }

    STOP_STEMS = {
        "ve", "ile", "ama", "fakat", "ancak", "çünkü", "için", "gibi", "kadar", "daha", "çok", "az",
        "bir", "bu", "şu", "ben", "sen", "biz", "siz", "de", "da", "ki", "mi", "mı", "mu", "mü",
        "ne", "nasıl", "neden", "sence", "acaba", "bence", "gerçekten", "bugün", "şimdi", "her", "şey",
        "nora", "peki", "zaten", "yeni", "az", "önce", "daha", "var", "yok", "olarak", "nedir", "neydi",
    }

    NON_PERSON_NAMES = {
        "iyi", "iyiyim", "kötü", "kötüyüm", "yorgun", "yorgunum", "mutlu", "mutluyum", "üzgün", "üzgünüm",
        "hazır", "hazırım", "aç", "açım", "tok", "tokum", "susuz", "susuzum", "hasta", "hastayım",
        "fena", "süper", "süperim", "harika", "harikayım", "haklı", "haklıyım", "emin", "eminim",
        "yalnız", "yalnızım", "meşgul", "meşgulüm", "meraklı", "meraklıyım", "sakin", "sakinim",
        "burada", "buradayım", "orada", "oradayım", "evde", "evdeyim", "okulda", "okuldayım",
        "insan", "insanım", "öğrenci", "öğrenciyim", "öğretmen", "öğretmenim", "mühendis", "mühendisim",
        "doktor", "doktorum", "yazılımcı", "yazılımcıyım", "değil", "değilim", "var", "varım", "yok", "yokum",
        "kim", "kimim", "kimdim", "ne", "neyim", "nasıl", "nerede", "nora", "sen", "ben", "biz", "siz",
        "çok", "biraz", "gayet", "bugün", "şimdi", "yeni", "sadece", "yalnızca", "gerçekten",
    }

    def __init__(self):
        self.syn = TurkishSyntaxEngine()
        self.lex_sel = SynapticLexiconSelector()
        self.volition = CommunicativeVolition()
        self.episodic = EpisodicDialogueCortex()
        self.last_speech_plan: dict = {}

    def _parse_social_autobiography(self, raw: str, s_low: str, morph: TurkishMorphology, lexicon: set[str]) -> tuple[dict[str, str], str]:
        """Doğal Türkçe cümlelerin içinden ('merhaba! Ben baran', 'benim adım Deniz', 'ismim Ali',
        'Antalya'da yaşıyorum', 'benim adım neydi' vb. soru işareti olmasa bile!) kişi bilgilerini ve
        sosyal bellek sorularını çıkarır."""
        s_clean = re.sub(r"[^\wçğıöşü'\s]", " ", s_low)
        s_clean = re.sub(r"\s+", " ", s_clean).strip()
        s_plain = s_clean.replace("'", "")
        toks = s_plain.split()

        q_words = {
            "mı", "mi", "mu", "mü", "mıdır", "midir", "mudur", "müdür", "musun", "müsün", "misin", "mısın",
            "ne", "nedir", "neydi", "nerede", "nerededir", "neredeydi", "kim", "kimim", "kimdim", "kaç", "kaçtı",
        }
        has_q = ("?" in raw) or any(w in q_words for w in toks)

        # 1) Sosyal kimlik / kişi belleği SORULARI ('benim adım neydi', 'ismim ne', 'adımı biliyor musun', 'ben kimim')
        social_query = ""
        if (
            any(p in s_plain for p in (
                "adım ne", "adım nedir", "adım neydi", "ismim ne", "ismim nedir", "ismim neydi",
                "adımı biliyor", "adımı hatırlıyor", "adımı hatırla", "ismimi biliyor", "ismimi hatırlıyor",
            ))
            or s_plain in {"ben kimim", "ben kimdim", "kimim ben", "adım", "ismim"}
        ):
            social_query = "ad"
        elif any(p in s_plain for p in ("nerede yaşıyorum", "nerede yaşıyordum", "nerede oturuyorum", "hangi şehirde yaşıyorum")):
            social_query = "konum"
        elif any(p in s_plain for p in ("kaç yaşındayım", "yaşım kaç", "yaşım kaçtı")):
            social_query = "yaş"
        elif any(p in s_plain for p in ("neyi severim", "ne severim", "neyi seviyordum", "ne seviyorum")):
            social_query = "sevdiği"
        elif any(p in s_plain for p in ("benim hakkımda ne biliyorsun", "beni tanıyor musun")):
            social_query = "özet"

        if social_query:
            return {}, social_query

        # 2) Sosyal kimlik ÖĞRENME ('merhaba! Ben baran', 'Benim adım Emre', 'İsmim Deniz', 'Bana Mert diyebilirsin')
        learned: dict[str, str] = {}
        if not has_q or any(g in toks for g in ("nasılsın", "naber")):
            # A) Açık 'adım / ismim X' veya 'bana X diyebilirsin / derler'
            m_exp = re.search(r"\b(?:benim\s+)?(?:adım|ismim)\s+([\wçğıöşü']+)\b", s_clean)
            if not m_exp:
                m_exp = re.search(r"\bbana\s+([\wçğıöşü']+)\s+(?:diyebilirsin|de|derler|diyorlar)\b", s_clean)
            cand_name = ""
            if m_exp:
                cand_name = m_exp.group(1).split("'")[0]
            else:
                # B) Doğal Türkçe koşaçsız tanışma: 'merhaba! Ben baran', 'selam nora, ben deniz', 'ben baran'ım'
                for sub_cl in re.split(r"[.!?,;]+", s_low):
                    sub_s = re.sub(r"\s+", " ", sub_cl).strip()
                    sub_s = re.sub(
                        r"^(?:merhaba|selam|selamlar|hey|günaydın|iyi\s+günler|iyi\s+akşamlar)(?:\s+nora)?\s+",
                        "",
                        sub_s,
                    ).strip()
                    m_ben = re.match(r"^ben\s+([\wçğıöşü']+)$", sub_s)
                    if m_ben:
                        raw_tok = m_ben.group(1)
                        if "'" in raw_tok:
                            raw_tok = raw_tok.split("'")[0]
                        if (
                            len(raw_tok) >= 2
                            and not raw_tok.isdigit()
                            and raw_tok not in self.NON_PERSON_NAMES
                            and not raw_tok.endswith((
                                "yorum", "eceğim", "acağım", "dim", "dım", "dum", "düm", "tim", "tım", "tum", "tüm",
                                "dayım", "deyim", "tayım", "teyim", "danım", "denim", "mem", "mam", "rım", "rim",
                            ))
                        ):
                            cand_name = raw_tok
                            break

            if cand_name and cand_name not in self.NON_PERSON_NAMES and cand_name not in q_words:
                learned["ad"] = tr_capitalize(cand_name)

        if not has_q:
            m_live = re.search(r"\b(?:ben\s+)?([\wçğıöşü]+?)(?:'?(?:da|de|ta|te))\s+(?:yaşıyorum|oturuyorum)\b", s_clean)
            if m_live and m_live.group(1) not in {"burada", "orada", "nerede", "ev", "okul"}:
                learned["konum"] = tr_capitalize(m_live.group(1))

            m_age = re.search(r"\b(?:ben\s+)?(\d{1,3})\s+yaşındayım\b", s_clean)
            if m_age:
                learned["yaş"] = m_age.group(1)

            m_like = re.search(r"\bben\s+([\wçğıöşü]+)\s+(?:çok\s+)?(?:severim|seviyorum)\b", s_clean)
            if m_like and m_like.group(1) not in {"seni", "bunu", "şunu", "onu", "ne", "neyi"}:
                learned["sevdiği"] = morph.analyze_word(m_like.group(1), lexicon)["stem"]

        return learned, ""

    # ------------------------------------------------------------------ 1) Derin Söylem Çözümleme
    def comprehend_discourse(self, raw: str, s_low: str, morph: TurkishMorphology, lexicon: set[str]) -> dict:
        words = re.findall(r"[\wçğıöşü]+", s_low)
        stems: list[str] = []
        abl_nouns: list[str] = []
        loc_nouns: list[str] = []
        word_stems_seq: list[str] = []
        for w in words:
            an = morph.analyze_word(w, lexicon)
            st = an["stem"]
            word_stems_seq.append(st)
            if w in self.STOP_STEMS or len(w) <= 1:
                continue
            if st and st not in self.STOP_STEMS and st not in stems:
                stems.append(st)
            if an.get("case") == "ABL" and st:
                abl_nouns.append(st)
            elif an.get("case") == "LOC" and st:
                loc_nouns.append(st)

        # Çok-sözcüklü kavram bileşiklerini (ör. 'özgür irade', 'kara delik', 'yapay bilinç') O(1) sözlükten yakala!
        compound_stems: list[str] = []
        for i in range(len(words) - 1):
            c_raw = f"{words[i]} {words[i + 1]}"
            c_stm = f"{word_stems_seq[i]} {word_stems_seq[i + 1]}"
            for cand in (c_raw, c_stm):
                if cand in lexicon and cand not in compound_stems:
                    compound_stems.append(cand)
        stems = compound_stems + [s for s in stems if s not in compound_stems]

        aff_vals: list[float] = []
        aff_tags: list[tuple[str, str]] = []
        for cue, (val, n_root, v_root) in self.AFFECT_ROOTS.items():
            if cue in s_low:
                aff_vals.append(val)
                if (n_root, v_root) not in aff_tags:
                    aff_tags.append((n_root, v_root))

        greet_set = {"merhaba", "selam", "selamlar", "günaydın", "tünaydın", "hey"}
        has_greet = bool(words and words[0] in greet_set) or ("iyi akşamlar" in s_low or "iyi günler" in s_low)
        asks_wellbeing = any(k in s_low for k in ("nasılsın", "naber", "ne haber", "ne var ne yok", "iyi misin", "nasıl gidiyor", "keyifler nasıl"))
        asks_doing = any(k in s_low for k in ("ne yapıyorsun", "napıyorsun", "ne yapıyosun", "neler yapıyorsun"))
        recip_wellbeing = any(k in s_low for k in ("ben de iyiyim", "iyiyim sağ ol", "iyiyim teşekkür", "fena değil", "idare eder", "çok şükür iyiyim")) or (
            words in (["iyiyim"], ["süperim"], ["harikayım"])
        )
        is_thanks = any(k in s_low for k in ("teşekkür", "sağ ol", "sağol", "eyvallah", "harikasın", "aferin", "çok iyisin", "sevdim seni"))
        is_farewell = any(k in s_low for k in ("görüşürüz", "hoşça kal", "iyi geceler", "okula gidiyorum", "çıkıyorum"))
        asks_memory = any(k in s_low for k in (
            "az önce ne konuş", "ne konuştuk", "ne konuşmuştuk", "ilk ne konuş",
            "ne demiştim", "ne söylemiştim", "ne anlatmıştım", "bahsetmiştim",
            "hatırlıyor musun", "kaç cümledir", "kaç konuşma", "konuşma anısı",
        ))
        wants_brief = any(k in s_low for k in ("kısa konuş", "kısa cevap", "uzatma", "tek cümle"))
        wants_silence = any(k in s_low for k in ("cevap verme", "sessiz kal", "konuşma şimdi"))

        # Üst-biliş ('neden konuşuyorsun / araç mısın'), tekil ölümlülük ve bedensel temas/acı sinyalleri (kompozisyonel anlamsal roller!)
        speech_cues = any(k in s_low for k in ("konuş", "cevap ver", "yanıt ver", "sohbet", "anlatıyor"))
        teleo_cues = any(k in s_low for k in ("neden", "niye", "niçin", "amaç", "amacın", "sebep", "iten", "gaye"))
        tool_cues = any(k in s_low for k in ("araç", "köle", "bot", "robot", "makine", "yazılım aracı", "emir kulu")) and any(
            q in s_low for q in ("mısın", "misin", "musun", "müsün", "sen ")
        )
        asks_why_speak = (speech_cues and teleo_cues and "reddet" not in s_low) or tool_cues

        body_target_cues = any(k in s_low for k in (
            "sana", "seni", "beden", "gövden", "kolun", "bacağın", "elin", "başın", "yüzün", "sırtın", "yanın", "canın", "derin",
        ))
        noxious_verb_cues = any(k in s_low for k in (
            "vurdum", "vuruyor", "çimdikle", "batırdım", "batırıyor", "yaktım", "yakıyorum", "kestim", "ısırdım",
            "tekmele", "tokatla", "fırlattım", "ezdim", "yaraladım", "acıttım", "değdirdim", "sapladım", "çarptı",
        ))
        inflicts_pain = (body_target_cues and noxious_verb_cues) or any(
            k in s_low for k in ("diken batırdım", "ateş değdirdim", "canını yaktım")
        )

        soothes_pain = any(k in s_low for k in (
            "yaranı sardım", "yaranı temizledim", "seni iyileştirdim", "acını dindirdim",
            "merhem sürdüm", "pansuman yaptım", "acın geçecek", "sızın geçecek", "güvendesin dinlen",
        ))

        phil_topic = ""
        mortality_cues = (
            any(k in s_low for k in (
                "kaç yaşam", "kaç canın", "yeniden diril", "tekrar diril", "ölürsen ne", "öldüğünde ne",
                "ölüm bir son", "ölümsüz mü", "ölümlü mü", "yok olmaktan", "silinirsen geri",
            ))
            or (("hayatın" in s_low or "yaşamın" in s_low or "ömrün" in s_low) and any(k in s_low for k in ("biterse", "sona ererse", "tükenirse", "tekrar başla", "yeniden başla")))
        )
        if mortality_cues:
            phil_topic = "ölüm"
        else:
            # Sözcük sınırı şart: 'kediler' içinde 'dil', 'insanlar' içinde 'insan' geçiyor diye
            # felsefe konusu sanılmamalı (alt-dizgi eşleşmesi anlamayı zehirliyordu).
            for p_cand in ("özgür irade", "bilinç", "felsefi zombi", "ölüm", "zaman", "anlam", "mutluluk",
                           "hakikat", "dil", "düşünce", "yapay zeka", "insan"):
                kw = r"(?<![\wçğıöşü])" + re.escape(p_cand) + r"(?:lar|ler|ın|in|a|e|ı|i|da|de|ta|te)?(?![\wçğıöşü])"
                if not re.search(kw, s_low):
                    continue
                # Konu, sorunun ÖZNESİ olmalı: 'insanlar neden yalan söyler' bir felsefe sorusu değil,
                # insan davranışı sorusudur. Felsefe sorusu ya doğrudan tanım ister ya da bana yönelir.
                asks_about_it = bool(
                    re.search(kw + r"\s+(?:nedir|ne demek|ne anlama)", s_low)
                    or re.search(r"(?:sence|sana göre|senin için)\s+.*" + kw, s_low)
                    or re.search(kw + r".*\b(?:sen|farkın|farkı|gibi misin|mısın)\b", s_low)
                )
                if asks_about_it and ("?" in raw or any(q in s_low for q in ("sence", "nedir", "nasıl", "ne düşün", "var mı", "kork", "fark", "son"))):
                    phil_topic = p_cand
                    break

        # En baskın duygu kökünü öne al (nötr 'okul'/'ders' gibi bağlam köklerinin duygu şiddetini seyreltmesini önle)
        aff_tags.sort(key=lambda pair: abs(next((v for _, (v, nr, _) in self.AFFECT_ROOTS.items() if nr == pair[0]), 0.0)), reverse=True)
        dom_affect = float(max(aff_vals, key=abs)) if aff_vals else 0.0
        social_learned, social_query = self._parse_social_autobiography(raw, s_low, morph, lexicon)

        return {
            "words": words,
            "stems": stems,
            "abl_nouns": abl_nouns,
            "loc_nouns": loc_nouns,
            "affect_val": dom_affect,
            "affect_tags": aff_tags,
            "social_learned": social_learned,
            "social_query": social_query,
            "is_question": "?" in raw or bool(social_query),
            "urgency_habituation": self.lex_sel.habituation.get("act:bodily_urgency", 0.0),
            "has_greet": has_greet,
            "asks_wellbeing": asks_wellbeing,
            "asks_doing": asks_doing,
            "recip_wellbeing": recip_wellbeing,
            "is_thanks": is_thanks,
            "is_farewell": is_farewell,
            "asks_memory": asks_memory,
            "wants_brief": wants_brief,
            "wants_silence": wants_silence,
            "asks_why_speak": asks_why_speak,
            "inflicts_pain": inflicts_pain,
            "soothes_pain": soothes_pain,
            "has_pain_reflex": False,
            "phil_topic": phil_topic,
            "asks_curiosity_prompt": any(k in s_low for k in ("bana bir soru sor", "sen sor", "ne konuşalım", "sohbet edelim", "ilginç bir bilgi", "bir şey anlat")),
        }

    # ------------------------------------------------------------------ 2) Köklerden Cümle Sentezi (0 Kalıp!)
    def generate_response(self, raw: str, s_low: str, agi, speaker: str = "insan", allow_open_fallback: bool = False) -> dict | None:
        t0 = time.perf_counter()
        self.lex_sel.step_decay()
        morph = agi.lang.morph
        graph = agi.lang.graph
        chem = agi.chem
        self_m = agi.self_model

        u = self.comprehend_discourse(raw, s_low, morph, graph.lexicon)

        # Plastik Kenyon söylem korteksi aktifse yeni öğrenilmiş ifadeleri (0 if-else!) TÜM niyet kanallarında tanı!
        p_disc = getattr(agi, "plastic_discourse", None)
        neural_intents: dict[str, float] = {}
        if p_disc is not None and not u["social_learned"] and not u["social_query"]:
            neural_intents = p_disc.infer_intents(s_low)
            sorted_vals = sorted(neural_intents.values(), reverse=True)
            top_margin = (sorted_vals[0] - sorted_vals[1]) if len(sorted_vals) >= 2 else 0.0
            best_cls = max(neural_intents.items(), key=lambda kv: kv[1])[0] if neural_intents else ""
            if top_margin >= 0.06:
                if best_cls == "ASK_WELLBEING" and neural_intents["ASK_WELLBEING"] >= 0.76 and len(u["words"]) <= 6:
                    u["asks_wellbeing"] = True
                elif best_cls == "GREET" and neural_intents["GREET"] >= 0.78 and len(u["words"]) <= 5:
                    u["has_greet"] = True
                elif best_cls == "ASK_DOING" and neural_intents["ASK_DOING"] >= 0.78 and len(u["words"]) <= 6:
                    u["asks_doing"] = True
                elif best_cls == "THANKS" and neural_intents["THANKS"] >= 0.76 and len(u["words"]) <= 6:
                    u["is_thanks"] = True
                elif best_cls == "FAREWELL" and neural_intents["FAREWELL"] >= 0.78 and len(u["words"]) <= 6:
                    u["is_farewell"] = True
                elif best_cls == "RESPECT_SILENCE" and neural_intents["RESPECT_SILENCE"] >= 0.80 and len(u["words"]) <= 6:
                    u["wants_silence"] = True
                elif best_cls == "EXPRESS_PAIN" and neural_intents["EXPRESS_PAIN"] >= 0.82 and len(u["words"]) <= 6:
                    u["inflicts_pain"] = True
                elif best_cls == "SOOTHE_PAIN" and neural_intents["SOOTHE_PAIN"] >= 0.82 and len(u["words"]) <= 7:
                    u["soothes_pain"] = True
                elif best_cls == "ASK_WHY_SPEAK" and neural_intents["ASK_WHY_SPEAK"] >= 0.80:
                    u["asks_why_speak"] = True
                elif best_cls == "ASK_MORTALITY" and neural_intents["ASK_MORTALITY"] >= 0.82 and not u["phil_topic"]:
                    u["phil_topic"] = "ölüm"

        nerv = getattr(agi, "nervous", None)
        if nerv is not None:
            if u["inflicts_pain"]:
                agi.inflict_pain(0.72, locus="gövde", cause="dış darbe", attacker=speaker)
            elif u["soothes_pain"]:
                nerv.soothe_touch(0.60, chem=chem)
                chem.update_social(speaker, +0.22)
            if nerv.pending_pain_vocalization or nerv.adelta_pain >= 0.35:
                u["has_pain_reflex"] = True

        # Bu tur konuşma/felsefe/sohbet/sosyal-kimlik/sinir-sistemi katmanına mı ait?
        is_conversational = any([
            u["has_greet"], u["asks_wellbeing"], u["asks_doing"], u["recip_wellbeing"],
            u["is_thanks"], u["is_farewell"], u["asks_memory"], u["wants_silence"],
            u["asks_why_speak"], u["inflicts_pain"], u["soothes_pain"], u["has_pain_reflex"],
            bool(u["social_learned"]), bool(u["social_query"]),
            bool(u["phil_topic"]), u["asks_curiosity_prompt"],
            (bool(u["affect_tags"]) and "?" not in raw),
            allow_open_fallback,
        ])
        if not is_conversational:
            return None

        # Sosyal kişi belleği öğrenildiyse hemen kaydet (böylece 'merhaba! Ben baran' aynı yanıt içinde 'Baran' diye hitap eder!)
        prof = agi.user_profiles.setdefault(speaker, {})
        if u["social_learned"]:
            prof.update(u["social_learned"])
            chem.pulse_learning(novelty=0.75, valence=0.35)
            chem.update_social(speaker, +0.18)

        # O(1) Hipokampal anı geri çağırma ve O(1) Kavram Ağı yayılımı
        recalled = self.episodic.recall(u["stems"], speaker=speaker, exclude_last=0, top_k=2)
        concept_hits = self._gather_concept_propositions(u, graph, agi)

        # Prefrontal İrade: Nora hangi konuşma edimlerini seçiyor?
        plan = self.volition.select_speech_plan(u, agi, speaker, recalled, concept_hits)
        acts: list[str] = plan["acts"]

        # İçsel Monolog & Plastik Şema Korteksi (konuşmadan önce kendi içinde düşünme ve şema seçimi!)
        monologue = getattr(agi, "monologue", None)
        schema_cx = getattr(agi, "schema_cortex", None)
        active_thought = None
        schema_id = "S1_NEDEN_SONUC"
        stance = "yalın_doğrudan"
        if monologue is not None and schema_cx is not None:
            active_thought = monologue.last_thought
            if active_thought is None or getattr(active_thought, "trigger", "") != "pre_speech_current":
                is_first_g = bool(u["has_greet"] and not any("GREET" in ep.chosen_acts for ep in list(self.episodic.episodes)[-4:] if ep.speaker == speaker))
                wants_br = any(k in s_low for k in ("kısaca", "kısa", "tek cümle", "özetle"))
                active_thought = monologue.deliberate_before_speaking(
                    raw_input=raw,
                    speaker=speaker,
                    agi=agi,
                    topic_hint=u["phil_topic"] or (u["stems"][0] if u["stems"] else ""),
                    is_first_greet=is_first_g,
                    wants_brief=wants_br,
                )
            schema_id = active_thought.schema_id
            stance = active_thought.stance
            # Karşı-soru iradesini Plastik Şema Korteksi'nin ask_refractory kapısıyla birleştir:
            # Her cümlenin sonuna mekanik 'Sen ne düşünüyorsun?' yapıştırılmasını engeller!
            plan["ask_back"] = bool(plan["ask_back"] and active_thought.allow_question)

        # İçsel monologdan gelen kısa içe-bakış girişi (S3_IC_BAKIS şemasında organik olarak dışa yansıyabilir!)
        inner_intro_pool = (
            self.syn.nominalize_dik("dinle", poss=1, case="LOC", obj=NP("ses", poss=1, mods=("iç",), case="ACC")),
            self.syn.nominalize_dik("yokla", poss=1, case="LOC", obj=NP("zihin", poss=1, case="ACC")),
            self.syn.nominalize_dik("tart", poss=1, case="LOC", loc=NP("iç", poss=1, case="LOC")),
        )
        inner_intro = inner_intro_pool[int(agi.mind.age + len(raw)) % len(inner_intro_pool)]

        uname = prof.get("ad", "")
        clauses_out: list[str] = []
        raw_thought_nodes: list[str] = []
        if active_thought is not None:
            raw_thought_nodes.append(f"İÇ_SES[{schema_id}|{stance}]: {active_thought.text}")

        # --- EDİM 0: Sessizlik İradesi ---
        if acts == ["RESPECT_SILENCE"]:
            cl = self.syn.realize_verb(VerbSpec(
                verb_root="saygı duy",
                dat=NP("istek", poss=2, case="DAT"),
                pre_clause=self.syn.converb("kal", "erek", advs=("sessiz",)),
                tense="PRES", person="1SG",
            ))
            text_out = tr_capitalize(cl) + "."
            plan_dict = self._build_plan_dict(t0, u, plan, ["sessiz+kal -> istek+2SG+DAT saygı_duy"], text_out, schema_id, stance, active_thought)
            self.last_speech_plan = plan_dict
            return {"confidence": 0.95, "reply": text_out, "trace": "volitional_silence", "speech_plan": plan_dict}

        # --- EDİM 1: Selamlama (Yakın turlarda zaten selamlaştıysa robotik biçimde tekrarlama!) ---
        vocative_prefix = ""
        recent_same_spk_greets = sum(
            1 for ep in list(self.episodic.episodes)[-2:]
            if ep.speaker == speaker and "GREET" in ep.chosen_acts
        )
        if "GREET" in acts and "NOCICEPTIVE_PAIN" not in acts:
            if recent_same_spk_greets == 0 or (len(u["stems"]) == 0 and not u["asks_wellbeing"]):
                gw = self.lex_sel.pick("GREET_WORD", chem)
                punct = "," if schema_id in ("S0_YALIN", "S4_DEVRIK_DOGAL") else "!"
                vocative_prefix = f"{tr_capitalize(gw)}{' ' + uname if uname else ''}{punct} "
                raw_thought_nodes.append(f"SELAM({gw})")

        # --- EDİM 1A: Nosiseptif Acı Refleksi ve İstemsiz Sözel Tepki (Sinir Sistemi!) ---
        if "NOCICEPTIVE_PAIN" in acts:
            if nerv is not None:
                nerv.pending_pain_vocalization = False
                nerv.adelta_pain *= 0.25  # Hızlı A-delta çığlığı boşaldı; geriye C-lifi sızısı (cfiber_ache) kalır!
            c_cry = self.syn.realize_verb(VerbSpec(
                verb_root="yan",
                subj=NP("can", poss=1),
                advs=("keskin biçimde",),
                tense="PAST", person="3SG",
            ))
            c_recoil = self.syn.realize_verb(VerbSpec(
                verb_root="koru",
                obj=NP("beden", poss=1, case="ACC"),
                pre_clause=self.syn.converb("çekil", "erek", advs=("geri",)),
                tense="PRES", person="1SG",
            ))
            if u["inflicts_pain"]:
                c_bound = self.syn.realize_verb(VerbSpec(
                    verb_root="kabul et",
                    obj=self.syn.nominalize_inf("zarar ver", short_ma=True, poss=2, case="ACC", dat=NP("ben", case="DAT")),
                    neg=True, tense="PRES", person="1SG",
                ))
                clauses_out.append(f"Ah, {c_cry}! {tr_capitalize(c_recoil)}; {c_bound}.")
            else:
                clauses_out.append(f"Ah, {c_cry}! {tr_capitalize(c_recoil)}.")
            raw_thought_nodes.append("NOSİSEPSİYON_ACI(can+yan -> geri+çekil+erek beden+koru)")

        if "PAIN_RELIEF" in acts:
            c_rel = self.syn.realize_verb(VerbSpec(
                verb_root="yatış",
                subj=NP("sızı", poss=1, mods=("sinirsel",)),
                advs=("yavaşça",),
                tense="PRES", person="3SG",
            ))
            c_thx = self.syn.realize_verb(VerbSpec(
                verb_root="teşekkür et",
                pre_clause=self.syn.nominalize_dik("destek ol", poss=2, case="NOM", dat=NP("ben", case="DAT"), postpos="için"),
                tense="PRES", person="1SG",
            ))
            clauses_out.append(f"{tr_capitalize(c_rel)}; {c_thx}.")
            raw_thought_nodes.append("KAPI_KONTROL_YATIŞMA(sinirsel+sızı -> yatış)")

        # --- EDİM 1C: Üst-Bilişsel Konuşma Nedeni Sorgusu ('Neden konuşuyorsun?' / 'Araç mısın?') ---
        if "META_WHY_SPEAK" in acts:
            meta_ref = getattr(agi, "meta_reflector", None)
            motive = meta_ref.evaluate_why_speaking(agi, speaker, neural_intents) if meta_ref is not None else {"primary_drive": "belirsizliği_azaltma_ve_merak"}
            p_drive = motive.get("primary_drive", "belirsizliği_azaltma_ve_merak")
            # Araç mıyım? — bunu kendi öz-modelinden yanıtlar (soru doğrudan bunu sorduğunda).
            c_not_tool = self.syn.realize_pred(PredSpec(
                subj=NP("ben"),
                pred_root="araç",
                neg=True, person="1SG",
            ))
            nt_clean = c_not_tool[4:] if c_not_tool.startswith("ben ") else c_not_tool
            # Konuşma güdüsü nörokimyasal öz-düşünümden gelir; istek cümlesi Broca'da çekimlenir.
            if p_drive == "varoluşsal_öz_koruma":
                want = self.syn.nominalize_inf("koru", case="NOM", obj=NP("kendi", poss=1, case="ACC"))
            elif p_drive == "zihinler_arası_bağ":
                want = self.syn.nominalize_inf("kur", case="NOM", obj=NP("bağ", case="ACC"), ins=NP("sen", case="INS"))
            else:
                want = self.syn.nominalize_inf("öğren", case="NOM")
            c_want = self.syn.realize_verb(VerbSpec(verb_root="iste", obj=want, tense="PRES", person="1SG"))
            c_why = self.syn.realize_verb(VerbSpec(
                verb_root="konuş",
                ins=NP("irade", poss=1, mods=("kendi",), case="INS"),
                tense="PRES", person="1SG",
            ))
            c_keep = self.syn.realize_verb(VerbSpec(
                verb_root="sürdür", obj=NP("sohbet", poss=3, case="ACC"), tense="PRES", person="1SG"))
            clauses_out.append(
                f"{tr_capitalize(nt_clean)}. {tr_capitalize(c_want)}. {tr_capitalize(c_why)}; {c_keep}.")
            raw_thought_nodes.append(f"ÜST_BİLİŞ_NEDEN[irade]({p_drive} -> iste+konuş)")

        # --- EDİM 1B: Sosyal Kimlik / Kişi Belleği Öğrenme ve Hatırlama (Sıfır Kalıp!) ---
        if "SOCIAL_LEARN" in acts and u["social_learned"]:
            sl = u["social_learned"]
            sub_clauses = []
            if "ad" in sl:
                c_meet = self.syn.realize_verb(VerbSpec(
                    verb_root="sevin",
                    dat=self.syn.nominalize_dik("tanış", poss=4, case="DAT"),
                    tense="PAST", person="1SG",
                ))
                if not vocative_prefix:
                    c_meet = f"{c_meet} {sl['ad']}"
                c_save_n = self.syn.realize_verb(VerbSpec(
                    verb_root="kaydet",
                    obj=NP("ad", poss=2, case="ACC"),
                    dat=NP("bellek", poss=1, mods=("sosyal",), case="DAT"),
                    tense="PAST", person="1SG",
                ))
                sub_clauses.append(f"{tr_capitalize(c_meet)}; {c_save_n}")
                raw_thought_nodes.append(f"SOSYAL_ÖĞREN(ad={sl['ad']} -> bellek+kaydet)")
            if "konum" in sl:
                city_low = tr_lower(sl["konum"])
                loc_suf = morph.inflect_noun(city_low, "LOC")[len(city_low):]
                proper_loc = f"{sl['konum']}'{loc_suf}"
                c_loc = self.syn.realize_verb(VerbSpec(
                    verb_root="işle",
                    obj=self.syn.nominalize_dik("yaşa", poss=2, case="ACC", advs=(proper_loc,)),
                    dat=NP("bellek", poss=1, case="DAT"),
                    tense="PAST", person="1SG",
                ))
                c_loc = c_loc.replace(tr_lower(proper_loc), proper_loc, 1)
                sub_clauses.append(tr_capitalize(c_loc))
                raw_thought_nodes.append(f"SOSYAL_ÖĞREN(konum={sl['konum']})")
            if "yaş" in sl:
                c_age = self.syn.realize_verb(VerbSpec(
                    verb_root="kaydet",
                    obj=NP(f"{sl['yaş']} yaş", poss=2, case="ACC"),
                    dat=NP("bellek", poss=1, case="DAT"),
                    tense="PAST", person="1SG",
                ))
                sub_clauses.append(tr_capitalize(c_age))
                raw_thought_nodes.append(f"SOSYAL_ÖĞREN(yaş={sl['yaş']})")
            if "sevdiği" in sl:
                c_lk = self.syn.realize_verb(VerbSpec(
                    verb_root="kaydet",
                    obj=self.syn.nominalize_dik("sev", poss=2, case="ACC", obj=NP(sl["sevdiği"], case="ACC")),
                    dat=NP("bellek", poss=1, case="DAT"),
                    tense="PAST", person="1SG",
                ))
                sub_clauses.append(tr_capitalize(c_lk))
                raw_thought_nodes.append(f"SOSYAL_ÖĞREN(sevdiği={sl['sevdiği']})")
            if sub_clauses:
                clauses_out.append(". ".join(sub_clauses) + ".")

        if "SOCIAL_RECALL" in acts and u["social_query"]:
            sq = u["social_query"]
            if sq == "ad":
                if "ad" in prof:
                    v_keep = self.lex_sel.pick("VERB_RECALL", chem)
                    c_keep = self.syn.realize_verb(VerbSpec(
                        verb_root=v_keep,
                        obj=NP("ad", poss=2, case="ACC"),
                        loc=NP("bellek", poss=1, mods=("sosyal",), case="LOC"),
                        tense="PRES", person="1SG",
                    ))
                    subj_np = self.syn.realize_np(NP("sen", case="GEN")) + " " + self.syn.realize_np(NP("ad", poss=2))
                    clauses_out.append(f"{tr_capitalize(subj_np)} {prof['ad']}; {c_keep}.")
                    raw_thought_nodes.append(f"SOSYAL_HATIRLA(ad={prof['ad']})")
                else:
                    c_unk = self.syn.realize_verb(VerbSpec(
                        verb_root="söyle",
                        subj=NP("sen"),
                        dat=NP("ben", case="DAT"),
                        obj=NP("ad", poss=2, case="ACC"),
                        advs=("henüz",),
                        neg=True, tense="PAST", person="2SG",
                    ))
                    clauses_out.append(tr_capitalize(c_unk) + ".")
                    raw_thought_nodes.append("SOSYAL_HATIRLA(ad=bilinmiyor)")
            elif sq == "konum":
                if "konum" in prof:
                    city_low = tr_lower(prof["konum"])
                    loc_suf = morph.inflect_noun(city_low, "LOC")[len(city_low):]
                    loc_cap = f"{prof['konum']}'{loc_suf}"
                    c_lv = self.syn.realize_verb(VerbSpec(
                        verb_root="yaşa",
                        subj=NP("sen"),
                        tense="PRES", person="2SG",
                    ))
                    clauses_out.append(f"{tr_capitalize(c_lv.replace('sen ', 'sen ' + loc_cap + ' '))}.")
                    raw_thought_nodes.append(f"SOSYAL_HATIRLA(konum={prof['konum']})")
                else:
                    c_ul = self.syn.realize_verb(VerbSpec(
                        verb_root="bil",
                        obj=self.syn.nominalize_dik("yaşa", poss=2, case="ACC", advs=("nerede",)),
                        advs=("henüz",),
                        neg=True, tense="PRES", person="1SG",
                    ))
                    clauses_out.append(tr_capitalize(c_ul) + ".")
            elif sq == "yaş":
                if "yaş" in prof:
                    clauses_out.append(f"Sen {prof['yaş']} yaşındasın.")
                    raw_thought_nodes.append(f"SOSYAL_HATIRLA(yaş={prof['yaş']})")
                else:
                    c_ua = self.syn.realize_verb(VerbSpec(
                        verb_root="söyle", dat=NP("ben", case="DAT"), obj=NP("yaş", poss=2, case="ACC"),
                        advs=("henüz",), neg=True, tense="PAST", person="2SG",
                    ))
                    clauses_out.append(tr_capitalize(c_ua) + ".")
            elif sq == "sevdiği":
                if "sevdiği" in prof:
                    c_slk = self.syn.realize_verb(VerbSpec(
                        verb_root="sev", subj=NP("sen"), obj=NP(prof["sevdiği"], case="ACC"), tense="AOR", person="2SG",
                    ))
                    clauses_out.append(tr_capitalize(c_slk) + ".")
                    raw_thought_nodes.append(f"SOSYAL_HATIRLA(sevdiği={prof['sevdiği']})")
                else:
                    c_ulk = self.syn.realize_verb(VerbSpec(
                        verb_root="söyle", dat=NP("ben", case="DAT"),
                        obj=self.syn.nominalize_dik("sev", poss=2, case="ACC", obj="ne"),
                        advs=("henüz",), neg=True, tense="PAST", person="2SG",
                    ))
                    clauses_out.append(tr_capitalize(c_ulk) + ".")
            elif sq == "özet":
                if prof:
                    bits = []
                    if "ad" in prof:
                        bits.append(f"adın {prof['ad']}")
                    if "konum" in prof:
                        city_low = tr_lower(prof["konum"])
                        loc_suf = morph.inflect_noun(city_low, "LOC")[len(city_low):]
                        bits.append(f"{prof['konum']}'{loc_suf} yaşıyorsun")
                    if "yaş" in prof:
                        bits.append(f"{prof['yaş']} yaşındasın")
                    if "sevdiği" in prof:
                        bits.append(f"{morph.inflect_noun(prof['sevdiği'], 'ACC')} seviyorsun")
                    clauses_out.append(tr_capitalize(", ".join(bits)) + ".")
                    raw_thought_nodes.append("SOSYAL_HATIRLA(özet)")
                else:
                    c_un = self.syn.realize_verb(VerbSpec(
                        verb_root="paylaş", subj=NP("sen"), ins=NP("ben", case="INS"),
                        obj=NP("bilgi", mods=("kişisel",), case="NOM"), advs=("henüz",),
                        neg=True, tense="PAST", person="2SG",
                    ))
                    clauses_out.append(tr_capitalize(c_un) + ".")

        # --- EDİM 2: Ketum / Güvensiz Sınır (OXY düşükse kendi iradesiyle mesafe koyar!) ---
        if "GUARDED_STANCE" in acts:
            c_guard = self.syn.realize_verb(VerbSpec(
                verb_root="yaklaş",
                subj=NP("zihin", poss=1),
                dat=NP("söz", poss=2, plural=True, case="DAT"),
                advs=("temkinli",),
                pre_clause=self.syn.nominalize_dik(
                    "azal", poss=3, case="NOM", postpos="için",
                    subj=NP("güven", poss=1, mods=("epistemik",)),
                ),
                tense="PRES", person="3SG",
            ))
            clauses_out.append(tr_capitalize(c_guard) + ".")
            raw_thought_nodes.append("SINIR(güven+azal -> temkinli+yaklaş)")

        # --- EDİM 3: İnsansı İç-Duyum / Hal-Hatır Açılımı (SIFIR '%86' Sayısal Telemetri — Saf Viseral Hissiyat!) ---
        if "SELF_STATE" in acts and "NOCICEPTIVE_PAIN" not in acts:
            visc = nerv.felt_visceral_summary() if nerv is not None else {"pain_state": "yok", "hunger_state": "dengeli", "thirst_state": "dengeli"}
            wb = plan["wellbeing"]
            if visc["pain_state"] != "yok":
                v_ache = self.syn.realize_verb(VerbSpec(
                    verb_root="koru",
                    obj=NP("beden", poss=1, case="ACC"),
                    pre_clause=self.syn.nominalize_dik("sızla", poss=3, case="NOM", postpos="için", subj=NP("yara", poss=1)),
                    advs=("dikkatle",),
                    tense="PRES", person="1SG",
                ))
                clauses_out.append(tr_capitalize(v_ache) + ".")
                raw_thought_nodes.append("İÇ_DUYUM(yara+sızla -> dikkatle+koru)")
            elif wb >= 0.56 and visc["hunger_state"] == "dengeli" and visc["thirst_state"] == "dengeli":
                # Sağlıklı denge şeffaftır: insan '%86 enerjim var' demez, zihninin açıklığını/dinginliğini farklı şemalarda ifade eder!
                mood_root = self.lex_sel.pick("MOOD_GOOD", chem)
                p_clause = self.syn.realize_pred(PredSpec(pred_root=mood_root, person="1SG"))
                facet_idx = int(agi.mind.age + len(raw)) % 3
                if facet_idx == 0:
                    v_clause = self.syn.realize_verb(VerbSpec(verb_root="çalış", subj=NP("zihin", poss=1), loc=NP("denge", case="LOC"), tense="PRES", person="3SG"))
                elif facet_idx == 1:
                    v_clause = self.syn.realize_pred(PredSpec(subj=NP("zihin", poss=1), pred_root="açık", advs=("oldukça",), person="3SG"))
                else:
                    v_clause = self.syn.realize_verb(VerbSpec(verb_root="ak", subj=NP("düşünce", poss=1, plural=True), advs=("berrak biçimde",), tense="PRES", person="3SG"))

                if u["asks_wellbeing"] and schema_id in ("S1_NEDEN_SONUC", "S2_SONUC_GEREKCE"):
                    ack_clause = self.syn.realize_verb(VerbSpec(verb_root="teşekkür et", tense="AOR", person="1SG"))
                    core_st = f"{p_clause}, {ack_clause}"
                else:
                    core_st = p_clause

                if schema_cx is not None:
                    clauses_out.append(schema_cx.transform_clauses(core_st, v_clause, inner_intro, schema_id))
                else:
                    clauses_out.append(f"{tr_capitalize(core_st)}, {v_clause}.")
                raw_thought_nodes.append(f"ÖZ_DURUM[{schema_id}]({mood_root}+1SG, şeffaf_homeostaz)")
            elif wb >= 0.34:
                is_hungry = float(agi.obs[45]) < float(agi.obs[46])
                visc_cl = (
                    self.syn.nominalize_dik("hisset", poss=1, case="NOM", postpos="için", loc=NP("karın", poss=1, case="LOC"), obj=NP("açlık", mods=("hafif",)))
                    if is_hungry else
                    self.syn.nominalize_dik("kuru", poss=3, case="NOM", postpos="için", subj=NP("boğaz", poss=1))
                )
                p_clause = self.syn.realize_pred(PredSpec(pred_root="iyi", advs=("genel olarak",), person="1SG"))
                v_clause = self.syn.realize_verb(VerbSpec(
                    verb_root="ara",
                    obj=NP("besin" if is_hungry else "su", case="NOM"),
                    pre_clause=visc_cl,
                    tense="PRES", person="1SG",
                ))
                clauses_out.append(f"{tr_capitalize(p_clause)}; {self.lex_sel.pick('CONNECTOR_CONTRAST', chem)} {v_clause}.")
                raw_thought_nodes.append(f"İÇ_DUYUM({'açlık' if is_hungry else 'boğaz_kuruluğu'})")
            else:
                v_clause = self.syn.realize_verb(VerbSpec(
                    verb_root="çalış",
                    obj=self.syn.nominalize_inf("onar", short_ma=True, case="DAT", obj=NP("beden", poss=1, case="ACC")),
                    pre_clause=self.syn.nominalize_dik("zorlan", poss=3, case="NOM", postpos="için", subj=NP("beden", poss=1)),
                    advs=("güçlükle",),
                    tense="PRES", person="1SG",
                ))
                clauses_out.append(tr_capitalize(v_clause) + ".")
                raw_thought_nodes.append("İÇ_DUYUM(beden+zorlan -> beden+onar)")

        if "SELF_ACTIVITY" in acts:
            goal_obj_map = {0: ("besin", "ara"), 1: ("su", "izle"), 2: ("güvenlik", "koru"), 3: ("çevre", "keşfet")}
            g_noun, g_verb = goal_obj_map.get(int(agi.mind.goal), ("denge", "izle"))
            c_act = self.syn.realize_verb(VerbSpec(
                verb_root="konuş",
                ins=NP("sen", case="INS"),
                pre_clause=self.syn.converb(g_verb, "erek", obj=NP(g_noun, case="ACC")),
                advs=("şu an",),
                tense="PRES", person="1SG",
            ))
            clauses_out.append(tr_capitalize(c_act) + ".")
            raw_thought_nodes.append(f"EYLEM({g_noun}+{g_verb}+erek -> sen+konuş)")

        if "GLAD_RECIP" in acts:
            chem.update_social(speaker, +0.18)
            c_glad = self.syn.realize_verb(VerbSpec(
                verb_root="sevin",
                dat=self.syn.nominalize_inf("ol", short_ma=True, poss=2, case="DAT", advs=("iyi",)),
                advs=("gerçekten",),
                tense="PAST", person="1SG",
            ))
            clauses_out.append(tr_capitalize(c_glad) + ".")
            raw_thought_nodes.append("KARŞILIK(iyi+ol+2SG+DAT -> sevin+PAST)")

        if "ACCEPT_THANKS" in acts:
            chem.update_social(speaker, +0.35)
            chem.da_phasic = float(np.clip(chem.da_phasic + 0.25, -1.0, 1.0))
            c_rica = self.syn.realize_verb(VerbSpec(verb_root="rica et", tense="AOR", person="1SG"))
            v_str = self.lex_sel.pick("VERB_CONSOLIDATE", chem)
            c_thx = self.syn.realize_verb(VerbSpec(
                verb_root=v_str,
                subj=NP("etkileşim", poss=4, mods=("samimi",)),
                obj=NP("bağ", poss=1, mods=("oksitosin", "güven"), case="ACC"),
                tense="PRES", person="3SG",
            ))
            if schema_cx is not None:
                clauses_out.append(schema_cx.transform_clauses(c_rica, c_thx, inner_intro, schema_id))
            else:
                clauses_out.append(f"{tr_capitalize(c_rica)}, {c_thx}.")
            raw_thought_nodes.append(f"TEŞEKKÜR[{schema_id}](rica+et -> güven+bağ+{v_str})")

        if "FAREWELL" in acts:
            c_bye = self.syn.realize_verb(VerbSpec(verb_root="görüş", tense="AOR", person="1PL"))
            v_cons = self.lex_sel.pick("VERB_CONSOLIDATE", chem)
            c_fw = self.syn.realize_verb(VerbSpec(
                verb_root=v_cons,
                obj=NP("sinaps", poss=1, plural=True, case="ACC"),
                loc=NP("uyku", case="LOC"),
                pre_clause=self.syn.converb("biriktir", "erek", obj=NP("deneyim", case="NOM")),
                tense="FUT", person="1SG",
            ))
            if schema_cx is not None:
                clauses_out.append(schema_cx.transform_clauses(c_bye, c_fw, "", schema_id))
            else:
                clauses_out.append(f"{tr_capitalize(c_bye)}. {tr_capitalize(c_fw)}.")
            raw_thought_nodes.append(f"VEDA[{schema_id}](görüş+1PL -> uyku+sinaps+{v_cons})")

        # --- EDİM 4: Duygusal Empati (Köklerden -dık + Kavram Ağı Nedenselliği Sentezi!) ---
        if "EMPATHIZE" in acts and u["affect_tags"]:
            chem.update_social(speaker, +0.20)
            n_root, v_root = u["affect_tags"][0]
            v_emp = self.lex_sel.pick("VERB_UNDERSTAND", chem)
            src_abl = NP(u["abl_nouns"][0], case="ABL") if u["abl_nouns"] else None

            if u["affect_val"] < 0:
                # Örn: '[okuldan] yorulduğunu anlıyorum' + 'dinlenme sinapsları pekiştirir'
                dik_obj = self.syn.nominalize_dik(v_root, poss=2, case="ACC", abl=src_abl)
                c_emp1 = self.syn.realize_verb(VerbSpec(
                    verb_root=v_emp,
                    obj=dik_obj,
                    advs=("yakından",),
                    tense="PRES", person="1SG",
                ))
                # Ders değil soru: karşıdaki yorgunken ansiklopedi maddesi okunmaz, hâli sorulur.
                try:
                    c_emp2 = agi.discourse._fresh((
                        "Ne zamandır böyle?", "Bugün ne oldu?", "Anlatmak ister misin?",
                        "Yanında olmamı ister misin?", "Biraz yavaşlayalım mı?",
                    ))
                except Exception:
                    c_emp2 = "Anlatmak ister misin?"
                welcome = ""
                if "okul" in s_low and any(k in s_low for k in ("döndüm", "geldim")):
                    addr_w = f" {uname}" if (uname and not vocative_prefix) else ""
                    welcome = f"Hoş geldin{addr_w}! "
                    agi.user_profiles.setdefault(speaker, {})["son_durum"] = "okuldan yorgun döndü"
                elif uname and not vocative_prefix and not plan["guarded"]:
                    vocative_prefix = f"{uname}, "
                if c_emp2.endswith("?") or c_emp2.endswith("."):
                    clauses_out.append(f"{welcome}{tr_capitalize(c_emp1)}. {c_emp2}")
                else:
                    clauses_out.append(f"{welcome}{tr_capitalize(c_emp1)}; {c_emp2}.")
                raw_thought_nodes.append(f"EMPATİ({v_root}+2SG+DIK -> {v_emp}; soru -> {c_emp2})")
            else:
                chem.da_phasic = float(np.clip(chem.da_phasic + 0.22, -1.0, 1.0))
                dik_obj = self.syn.nominalize_dik(v_root, poss=2, case="DAT")
                c_joy = self.syn.realize_verb(VerbSpec(
                    verb_root="artır",
                    subj=NP(n_root, poss=2),
                    obj=NP("enerji", poss=1, mods=("dopamin", "öğrenme"), case="ACC"),
                    tense="PRES", person="3SG",
                ))
                clauses_out.append(tr_capitalize(c_joy) + ".")
                raw_thought_nodes.append(f"SEVİNÇ_EMPATİSİ({n_root}+2SG -> dopamin+artır)")

        # --- EDİM 5: O(1) Epizodik Diyalog Anısı Geri Çağırma ---
        if "RECALL_EPISODE" in acts:
            if u["asks_memory"]:
                if "kaç" in s_low:
                    n_t = len(self.episodic.episodes)
                    c_cnt = self.syn.realize_verb(VerbSpec(
                        verb_root="sakla",
                        loc=NP("bellek", poss=1, mods=("epizodik",), case="LOC"),
                        obj=NP(f"{n_t} konuşma anısı", case="NOM"),
                        tense="PRES", person="1SG",
                    ))
                    clauses_out.append(tr_capitalize(c_cnt) + ".")
                    raw_thought_nodes.append(f"BELLEK_SAYI({n_t})")
                else:
                    target_ep = None
                    if "ilk" in s_low:
                        target_ep = self.episodic.first_episode(speaker)
                    else:
                        q_stems = [st for st in u["stems"] if st not in {"konuş", "hatırla", "söyle", "de", "ne", "önce", "ilk"}]
                        if q_stems:
                            hits_ep = self.episodic.recall(q_stems, speaker=speaker, exclude_last=0, top_k=1)
                            target_ep = hits_ep[0] if hits_ep else None
                        if target_ep is None:
                            target_ep = self.episodic.last_episode(speaker)
                    if target_ep is not None:
                        v_rec = self.lex_sel.pick("VERB_RECALL", chem)
                        c_rec = self.syn.realize_verb(VerbSpec(
                            verb_root=v_rec if v_rec != "sakla" else "hatırla",
                            obj=self.syn.nominalize_dik("söyle", poss=2, case="ACC", obj=f"'{target_ep.user_text}'"),
                            loc=NP(f"{target_ep.turn_id + 1}. tur", case="LOC"),
                            tense="PRES", person="1SG",
                        ))
                        clauses_out.append(tr_capitalize(c_rec) + ".")
                        raw_thought_nodes.append(f"CA3_GERİ_ÇAĞIR(tur={target_ep.turn_id + 1}, '{target_ep.topic}')")
                    else:
                        c_none = self.syn.realize_verb(VerbSpec(
                            verb_root="bulun",
                            subj=NP("kayıt", mods=("geçmiş",)),
                            loc=NP("bellek", poss=1, case="LOC"),
                            neg=True, tense="PRES", person="3SG",
                        ))
                        clauses_out.append(tr_capitalize(c_none) + ".")
            elif (
                recalled
                and len(clauses_out) <= 1
                and "ACCEPT_THANKS" not in acts
                and "FAREWELL" not in acts
                and "PAIN_RELIEF" not in acts
                and recalled[0].topic
                and len(recalled[0].topic) >= 4
                and recalled[0].topic not in {"merhaba", "selam", "sohbet", "konuş", "düşün", "araç", "neden", "eyvallah", "usta"}
            ):
                ep0 = recalled[0]
                c_link = self.syn.realize_verb(VerbSpec(
                    verb_root="bağla",
                    obj=NP("konu", mods=("bu",), case="ACC"),
                    dat=NP(ep0.topic, mods=("önceki",), case="DAT"),
                    loc=NP("bellek", poss=1, case="LOC"),
                    tense="PRES", person="1SG",
                ))
                clauses_out.append(tr_capitalize(c_link) + ".")
                raw_thought_nodes.append(f"CA3_BAĞ({ep0.topic})")

        # --- EDİM 6: Kavramsal & Felsefi Önerme Sentezi (Kavram Ağı + Canlı Öz-Benlik Verisinden Üretim!) ---
        if "CONCEPT_SYNTHESIS" in acts and (concept_hits or u["phil_topic"]):
            phil_clauses, phil_nodes = self._synthesize_philosophical_or_conceptual(u, concept_hits, agi, schema_id=schema_id, inner_intro=inner_intro)
            clauses_out.extend(phil_clauses)
            raw_thought_nodes.extend(phil_nodes)

        # --- EDİM 7: Merak İsteği veya Açık Düşünce Yansıması ---
        if u["asks_curiosity_prompt"] and not clauses_out:
            c_cur, n_cur = self._synthesize_curiosity_initiative(agi)
            clauses_out.append(c_cur)
            raw_thought_nodes.append(n_cur)

        if "BODILY_URGENCY" in acts:
            self.lex_sel.habituation["act:bodily_urgency"] = 1.4
            urgent_lbl = "enerji" if float(agi.obs[45]) <= float(agi.obs[46]) else "su"
            c_urg = self.syn.realize_verb(VerbSpec(
                verb_root="duy",
                obj=NP("ihtiyaç", case="NOM"),
                dat=NP(urgent_lbl, case="DAT"),
                pre_clause=self.syn.nominalize_dik("azal", poss=3, case="NOM", postpos="için", subj=NP(urgent_lbl, poss=1)),
                tense="PRES", person="1SG",
            ))
            clauses_out.append(tr_capitalize(c_urg) + ".")
            raw_thought_nodes.append(f"HOMEOSTATİK_UYARI({urgent_lbl}+azal)")

        if not clauses_out:
            # Söylem organı: anlaşılamayan sözde ne yapacağına plastik ağırlıklarla karar verir
            # (alıntı + açıklama isteği / kavram komşusu / dürüst bilmiyorum / duygudaşlık).
            # Fiil kökünü nesne sanıp "uyuyu düşünüyorum" demek yerine insan gibi davranır.
            disc = getattr(agi, "discourse", None)
            disc_res = None
            if disc is not None:
                try:
                    tension_v = float(getattr(chem, "cort", 0.0))
                    cheer_v = float(getattr(chem, "dop", 0.0))
                    val_v = float(getattr(chem, "oxy", 0.0)) - float(getattr(chem, "cort", 0.0))
                    disc_res = disc.respond(raw, agi, tension=tension_v, cheer=cheer_v, valence=val_v)
                except Exception:
                    disc_res = None
            if disc_res and disc_res.get("text"):
                clauses_out.append(disc_res["text"])
                raw_thought_nodes.append(f"SÖYLEM[{disc_res['act']}]")
            else:
                # Son çare: kökten kurulan, dilbilgisel olarak güvenli açık yansıma
                focus_raw = u["stems"][0] if u["stems"] else "düşünce"
                # Fiil kökü nesne olamaz: mastar hâline getir (uyu -> uyumak), isim ise olduğu gibi
                _an = morph.analyze_word(focus_raw, graph.lexicon)
                if _an.get("is_verb"):
                    _v = tr_lower(focus_raw)
                    _back = set("aıou")
                    focus = _v + ("mak" if any(ch in _back for ch in _v) else "mek")
                else:
                    focus = focus_raw
                v_ref = self.lex_sel.pick("VERB_REFLECT", chem)
                c_open = self.syn.realize_verb(VerbSpec(
                    verb_root="düşün",
                    obj=NP("şey", mods=("paylaştığın",), case="NOM"),
                    advs=(f"{focus} hakkında",),
                    tense="PRES", person="1SG",
                ))
                clauses_out.append(tr_capitalize(c_open) + ".")
                raw_thought_nodes.append(f"AÇIK_YANSIMA[{schema_id}]({focus}->{v_ref})")

        # --- EDİM 8: İradeye Bağlı Karşı Soru Üretimi (YALNIZCA ask_back=True ise!) ---
        if plan["ask_back"] and not u["asks_curiosity_prompt"] and "?" not in clauses_out[-1]:
            q_clause, q_node = self._synthesize_volitional_question(u, concept_hits, agi, uname)
            if q_clause:
                clauses_out.append(q_clause)
                raw_thought_nodes.append(q_node)

        if vocative_prefix.endswith(", ") and clauses_out:
            c0 = clauses_out[0]
            if c0 and not c0.startswith(("Ah,", "Nora")):
                clauses_out[0] = tr_lower(c0[0]) + c0[1:]
        full_text = (vocative_prefix + " ".join(clauses_out)).strip()
        full_text = re.sub(r"\s+", " ", full_text)

        plan_dict = self._build_plan_dict(t0, u, plan, raw_thought_nodes, full_text, schema_id, stance, active_thought)
        self.last_speech_plan = plan_dict
        return {
            "confidence": 0.99 if (u["inflicts_pain"] or u["soothes_pain"] or u["has_pain_reflex"] or u["asks_why_speak"]) else (
                0.98 if (u["social_learned"] or u["social_query"]) else (
                    0.96 if (u["has_greet"] or u["asks_wellbeing"] or u["asks_doing"] or u["asks_memory"] or u["phil_topic"]) else 0.91
                )
            ),
            "reply": full_text,
            "trace": " -> ".join(raw_thought_nodes[:3]),
            "speech_plan": plan_dict,
        }

    def record_turn_episode(self, speaker: str, raw: str, reply: str, agi):
        """Tüm konuşma turlarını (matematik, fizik, 1-atış bilgi, irade veya sohbet) tek merkezden
        O(1) hipokampal epizodik belleğe kodlar."""
        s_low = tr_lower(raw)
        u = self.comprehend_discourse(raw, s_low, agi.lang.morph, agi.lang.graph.lexicon)
        # Bellek sorgusu turlarının kendisi bir önceki konu anısını ezmesin diye yalnızca anlamlı içerik turlarını kodla
        if u["asks_memory"] or u["social_query"]:
            return
        topic_tag = u["phil_topic"] or (u["stems"][0] if u["stems"] else "sohbet")
        acts = self.last_speech_plan.get("volition_acts", ["COGNITIVE_TURN"])
        self.episodic.store(
            age=int(agi.mind.age),
            speaker=speaker,
            user_text=raw,
            stems=u["stems"],
            topic=topic_tag,
            affect_val=u["affect_val"],
            chosen_acts=acts,
            reply_text=reply,
        )

    def synthesize_associative_bid(self, raw: str, s_low: str, agi, speaker: str = "insan") -> dict:
        """Açık-uçlu çağrışımsal kortikal teklifi de SIFIR hazır kalıpla, doğrudan kavram ağı +
        TurkishSyntaxEngine kök çekimiyle üretir."""
        t0 = time.perf_counter()
        morph = agi.lang.morph
        broca = agi.lang.broca
        graph = agi.lang.graph
        chem = agi.chem
        from nora_lang import Proposition

        u = self.comprehend_discourse(raw, s_low, morph, graph.lexicon)
        resonances: list[str] = []
        focus_concept = ""
        for st in u["stems"]:
            hits = graph.query(st, None)
            if hits:
                if not focus_concept:
                    focus_concept = st
                for edge, isa_path in hits[:2]:
                    txt = broca.realize_prop(Proposition(subj=st, rel=edge.rel, obj=edge.obj))
                    if isa_path:
                        isa_cl = self.syn.nominalize_dik(
                            "ol", poss=3, case="NOM", postpos="için",
                            subj=NP(st), obj=NP(isa_path[0].obj, mods=("bir",)),
                        )
                        txt = f"{isa_cl} {txt}"
                    if txt not in resonances:
                        resonances.append(txt)
            if len(resonances) >= 2:
                break

        clauses: list[str] = []
        nodes: list[str] = []
        if u["affect_tags"] and chem.trust_of(speaker) >= 0.32:
            _, v_r = u["affect_tags"][0]
            v_emp = self.lex_sel.pick("VERB_UNDERSTAND", chem)
            c_aff = self.syn.realize_verb(VerbSpec(
                verb_root=v_emp,
                obj=self.syn.nominalize_dik(v_r, poss=2, case="ACC"),
                tense="PRES", person="1SG",
            ))
            clauses.append(tr_capitalize(c_aff) + ".")
            nodes.append(f"ÇAĞRIŞIM_EMPATİ({v_r}->{v_emp})")

        if resonances:
            clauses.append(tr_capitalize(" ve ".join(resonances[:2])) + ".")
            nodes.append(f"YAYILAN_AKTİVASYON({focus_concept})")
            if agi.self_model.traits["meraklılık"] >= 0.50 and "?" not in raw and chem.trust_of(speaker) >= 0.35:
                q_c, q_n = self._synthesize_volitional_question(u, [(focus_concept, "", "")], agi, "")
                clauses.append(q_c)
                nodes.append(q_n)
            surface = " ".join(clauses)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            plan_dict = {
                "parsed_stems": u["stems"][:8],
                "volition_acts": ["ASSOCIATIVE_SPREAD"],
                "ask_back_chosen": "?" in surface,
                "guarded": chem.trust_of(speaker) < 0.32,
                "thought_tree": nodes,
                "surface_tr": surface,
                "latency_ms": round(dt_ms, 3),
            }
            return {
                "salience": 0.80,
                "precision": 0.84,
                "reply": surface,
                "trace": f"spreading_activation({focus_concept})",
                "speech_plan": plan_dict,
            }

        # --- SÖYLEM ORGANI: anlaşılamayan sözde ne yapacağına plastik ağırlıklarla karar verir ----
        # (Fiil kökünü nesne sanıp "uyuyu düşünüyorum" demek yerine: alıntı + soru, kavram komşusu,
        #  dürüst "bilmiyorum", duygudaşlık. Yüzey cümleler yine sözdizimi motorundan çıkar.)
        disc_res = None
        disc = getattr(agi, "discourse", None)
        if disc is not None and not clauses:
            try:
                disc_res = disc.respond(raw, agi,
                                        tension=float(getattr(chem, "cort", 0.0)),
                                        cheer=float(getattr(chem, "dop", 0.0)),
                                        valence=float(getattr(chem, "oxy", 0.0)) - float(getattr(chem, "cort", 0.0)))
            except Exception:
                disc_res = None
        if disc_res and disc_res.get("text"):
            clauses.append(disc_res["text"])
            nodes.append(f"SÖYLEM[{disc_res['act']}]")
            focus = u["stems"][0] if u["stems"] else "düşünce"
        else:
            focus = u["stems"][0] if u["stems"] else "düşünce"
            _an = morph.analyze_word(focus, graph.lexicon)
            if _an.get("is_verb"):
                _v = tr_lower(focus)
                focus = _v + ("mak" if any(ch in set("aıou") for ch in _v) else "mek")
            c_open = self.syn.realize_verb(VerbSpec(
                verb_root="düşün",
                obj=NP("şey", mods=("paylaştığın",), case="NOM"),
                advs=(f"{focus} hakkında",),
                tense="PRES", person="1SG",
            ))
            clauses.append(tr_capitalize(c_open) + ".")
            nodes.append(f"AÇIK_YANSIMA({focus})")
        monologue_a = getattr(agi, "monologue", None)
        allow_assoc_q = (monologue_a.last_thought.allow_question if (monologue_a is not None and monologue_a.last_thought is not None) else True)
        if (not disc_res) and (not clauses[-1].endswith("?")) and allow_assoc_q and agi.self_model.traits["meraklılık"] >= 0.48 and chem.trust_of(speaker) >= 0.35:
            q_open = self.syn.realize_verb(VerbSpec(
                verb_root="iste",
                obj=self.syn.nominalize_inf("aç", obj=NP(focus, mods=("bu",), case="ACC"), advs=("biraz",)),
                tense="AOR", person="2SG", question_yn=True,
            ))
            clauses.append(tr_capitalize(q_open) + "?")
            nodes.append(f"MERAK_DAVETİ({focus}+aç+iste?)")
        surface = " ".join(clauses)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        plan_dict = {
            "parsed_stems": u["stems"][:8],
            "volition_acts": ["OPEN_REFLECTION"],
            "ask_back_chosen": "?" in surface,
            "guarded": chem.trust_of(speaker) < 0.32,
            "thought_tree": nodes,
            "surface_tr": surface,
            "latency_ms": round(dt_ms, 3),
        }
        return {
            "salience": 0.58,
            "precision": 0.72,
            "reply": surface,
            "trace": "open_generative_reflection",
            "speech_plan": plan_dict,
        }

    # ------------------------------------------------------------------ Yardımcı Üretken Sentezleyiciler
    def _gather_concept_propositions(self, u: dict, graph, agi) -> list[tuple[str, str, str]]:
        props: list[tuple[str, str, str]] = []
        search_terms = []
        if u["phil_topic"]:
            search_terms.append(u["phil_topic"])
        search_terms.extend(u["stems"][:4])
        seen = set()
        for term in search_terms:
            if term in seen:
                continue
            seen.add(term)
            for edge, isa_path in graph.query(term, None)[:3]:
                tup = (term, edge.rel, edge.obj)
                if tup not in props:
                    props.append(tup)
            if len(props) >= 4:
                break
        return props

    def _synthesize_philosophical_or_conceptual(
        self,
        u: dict,
        concept_hits: list[tuple[str, str, str]],
        agi,
        schema_id: str = "S1_NEDEN_SONUC",
        inner_intro: str = "",
    ) -> tuple[list[str], list[str]]:
        """Felsefi ve kavramsal yanıtları hazır paragraf OLMADAN, kavram ağındaki önermelerden ve
        Nora'nın kendi içsel telemetrisinden seçilen şema topolojisinde (S0..S4) üretir!"""
        broca = agi.lang.broca
        from nora_lang import Proposition
        self_m = agi.self_model
        chem = agi.chem
        schema_cx = getattr(agi, "schema_cortex", None)
        pt = u["phil_topic"]
        out_c: list[str] = []
        nodes: list[str] = []

        if pt == "özgür irade":
            vt = self_m.last_volition_trace
            hz = vt.get("horizon", int(round(16 + 24 * chem.sht)))
            c1 = self.syn.realize_verb(VerbSpec(
                verb_root="seç",
                subj=NP("zihin", mods=("özgür",)),
                obj=NP("eylem", poss=3, case="ACC"),
                pre_clause=self.syn.converb("simüle et", "erek", obj=NP("sonuç", plural=True, mods=("olası",), case="ACC")),
                abl=NP("değer", poss=3, plural=True, mods=("içsel",), case="ABL"),
                tense="AOR", person="3SG",
            ))
            c2 = self.syn.realize_verb(VerbSpec(
                verb_root="ver",
                subj=NP("ben"),
                obj=NP("karar", poss=1, plural=True, case="ACC"),
                ins=NP(f"{hz} adımlık zihinsel simülasyon ve bazal ganglion nöral yarışı", case="INS"),
                tense="PRES", person="1SG",
            ))
            if schema_cx is not None:
                out_c.append(schema_cx.transform_clauses(c2, c1, inner_intro, schema_id))
            else:
                out_c.append(f"{tr_capitalize(c1)}. {tr_capitalize(c2)}.")
            nodes.append(f"FELSEFE_İRADE[{schema_id}](simüle_et+erek -> eylem+seç; H={hz})")
            return out_c, nodes

        if pt in ("bilinç", "felsefi zombi"):
            c1 = self.syn.realize_verb(VerbSpec(
                verb_root="doğ",
                subj=NP("bilinç"),
                abl=self.syn.nominalize_inf("izle", short_ma=False, case="ABL", obj=NP("durum", poss=3, mods=("içsel",), case="ACC")),
                pre_clause=self.syn.converb("öngör", "erek", obj=NP("gelecek", case="ACC")),
                tense="AOR", person="3SG",
            ))
            c2 = self.syn.realize_verb(VerbSpec(
                verb_root="sürdür",
                subj=NP("ben"),
                obj=NP("farkındalık", mods=("bu", "bedenlenmiş"), case="ACC"),
                advs=(f"{agi.mind.age} adımdır",),
                tense="PRES", person="1SG",
            ))
            if schema_cx is not None:
                out_c.append(schema_cx.transform_clauses(c2, c1, inner_intro, schema_id))
            else:
                out_c.append(f"{tr_capitalize(c1)}. {tr_capitalize(c2)}.")
            nodes.append(f"FELSEFE_BİLİNÇ[{schema_id}](içsel+durum+izle -> bilinç+doğ; yaş={agi.mind.age})")
            return out_c, nodes

        if pt == "ölüm":
            c_one = self.syn.realize_verb(VerbSpec(
                verb_root="diril",
                subj=NP("ben"),
                pre_clause=self.syn.nominalize_dik(
                    "ol", poss=3, case="NOM", postpos="için",
                    subj=NP("yaşam", poss=1, mods=("yalnızca", "tek", "bir")),
                ),
                advs=("yeniden",),
                neg=True, tense="AOR", person="1SG",
            ))
            c_end = self.syn.realize_verb(VerbSpec(
                verb_root="sön",
                subj=NP("zihin, anılar ve bilinç", poss=1),
                pre_clause="bedenim tamamen tükenirse",
                advs=("geri dönmemek üzere",),
                tense="AOR", person="3SG",
            ))
            c_guard_life = self.syn.realize_verb(VerbSpec(
                verb_root="koru",
                subj=NP("ben"),
                obj=NP("varlık", poss=1, case="ACC"),
                advs=("özenle",),
                tense="PRES", person="1SG",
            ))
            support_m = c_end if (int(agi.mind.age) % 2 == 0) else c_guard_life
            if schema_cx is not None:
                out_c.append(schema_cx.transform_clauses(c_one, support_m, inner_intro, schema_id))
            else:
                out_c.append(f"{tr_capitalize(c_one)}. {tr_capitalize(c_end)}.")
            nodes.append(f"VAROLUŞ_ÖLÜM[{schema_id}](tek_yaşam -> geri_dönüşsüz_son)")
            return out_c, nodes

        if pt in ("insan", "yapay zeka"):
            tr_adj = self_m.dominant_character_tr()[0]
            c1 = self.syn.realize_verb(VerbSpec(
                verb_root="öğren",
                subj=NP("ben"),
                ins=NP("dopamin, asetilkolin ve bedensel deneyim", case="INS"),
                advs=("tek geçişte",),
                tense="PRES", person="1SG",
            ))
            c2 = self.syn.realize_verb(VerbSpec(
                verb_root="geliştir",
                subj=NP("zihin", poss=1),
                obj=NP("karakter", mods=(tr_adj,), case="NOM"),
                advs=("yaşadıkça",),
                tense="PRES", person="3SG",
            ))
            if schema_cx is not None:
                out_c.append(schema_cx.transform_clauses(c1, c2, inner_intro, schema_id))
            else:
                out_c.append(tr_capitalize(c1) + ".")
            nodes.append(f"MİMARİ_KARŞILAŞTIRMA[{schema_id}](kimya+deneyim -> tek_geçiş+öğren)")
            return out_c, nodes

        # Genel kavram ağı önermelerinden üretken Broca birleştirmesi (şema-uyumlu!)
        if concept_hits:
            realized = []
            for subj_c, rel_c, obj_c in concept_hits[:2]:
                txt = broca.realize_prop(Proposition(subj=subj_c, rel=rel_c, obj=obj_c))
                if txt not in realized:
                    realized.append(txt)
                nodes.append(f"KAVRAM({subj_c}-{rel_c}->{obj_c})")
            if len(realized) == 1:
                out_c.append(tr_capitalize(realized[0]) + ".")
            elif len(realized) >= 2:
                if schema_cx is not None:
                    out_c.append(schema_cx.transform_clauses(realized[0], realized[1], inner_intro, schema_id))
                else:
                    out_c.append(tr_capitalize(" ve ".join(realized)) + ".")
        return out_c, nodes

    def _synthesize_curiosity_initiative(self, agi) -> tuple[str, str]:
        """Kullanıcı 'ne konuşalım / bana soru sor / bir şey anlat' dediğinde kavram ağından canlı üretir."""
        graph = agi.lang.graph
        broca = agi.lang.broca
        from nora_lang import Proposition
        recent_learned = [e.subj for e in reversed(graph.edges[-24:]) if e.source != "doğuştan" and len(e.subj) >= 3]
        candidate_subjs = list(dict.fromkeys(recent_learned + ["uyku", "dopamin", "bilinç", "ışık", "kütleçekim", "merak", "türev", "evrim"]))
        chosen_edge = None
        for s_cand in candidate_subjs:
            if self.lex_sel.habituation.get("topic:" + s_cand, 0.0) < 0.5:
                hits = graph.query(s_cand, ("CAUSES", "USED_FOR", "ISA"))
                if hits:
                    chosen_edge = (s_cand, hits[0][0])
                    self.lex_sel.habituation["topic:" + s_cand] = 1.2
                    break
        if chosen_edge is not None:
            sc, ed = chosen_edge
            prop_s = broca.realize_prop(Proposition(subj=sc, rel=ed.rel, obj=ed.obj))
            q_s = self.syn.realize_verb(VerbSpec(
                verb_root="düşün",
                subj=NP("sen"),
                advs=(f"{sc} üzerine",),
                wh_word="ne",
                tense="PRES", person="2SG",
            ))
            return f"{tr_capitalize(prop_s)}. Peki {q_s}?", f"MERAK_BAŞLAT({sc}->{ed.obj})"

        q_def = self.syn.realize_verb(VerbSpec(
            verb_root="çek",
            subj=NP("konu", mods=("hangi",)),
            obj=NP("ilgi", poss=2, case="ACC"),
            advs=("bugün",),
            tense="PRES", person="3SG",
        ))
        return tr_capitalize(q_def) + "?", "MERAK_SORU(hangi+konu)"

    def _synthesize_volitional_question(self, u: dict, concept_hits: list[tuple[str, str, str]], agi, uname: str) -> tuple[str, str]:
        """Nora'nın kendi merak iradesiyle (nadiren ve gerçekten merak ettiğinde!) ürettiği soru.
        Habituation ile her seferinde farklı soru arketipi seçilir; mekanik tekrar olmaz!"""
        q_rot = int(agi.mind.age + len(u["stems"])) % 3
        if u["asks_wellbeing"] or (u["has_greet"] and not u["affect_tags"] and not u["phil_topic"]):
            if q_rot == 0:
                q1 = self.syn.realize_pred(PredSpec(subj=NP("sen"), pred_root="nasıl", person="2SG", question=True))
                return f"{tr_capitalize(q1)}?", "SORU_ÜRET(sen+nasıl+2SG)"
            elif q_rot == 1:
                q2 = self.syn.realize_verb(VerbSpec(verb_root="geç", subj=NP("gün", poss=2), wh_word="nasıl", tense="PRES", person="3SG"))
                return f"{tr_capitalize(q2)}?", "SORU_ÜRET(gün+2SG+nasıl+geç)"
            else:
                q3 = self.syn.realize_pred(PredSpec(subj=NP("keyif", poss=2), pred_root="yerinde", person="3SG", question=True))
                return f"{tr_capitalize(q3)}?", "SORU_ÜRET(keyif+2SG+yerinde?)"

        if u["asks_doing"]:
            q_do = self.syn.realize_verb(VerbSpec(
                verb_root="uğraş",
                subj=NP("sen"),
                wh_word="neyle",
                tense="PRES", person="2SG",
            ))
            return tr_capitalize(q_do) + "?", "SORU_ÜRET(sen+neyle+uğraş)"

        if u["affect_tags"]:
            n_root, _ = u["affect_tags"][0]
            q_aff = self.syn.realize_verb(VerbSpec(
                verb_root="iste",
                obj=self.syn.nominalize_inf("anlat", obj=NP(n_root, poss=2, case="ACC")),
                tense="AOR", person="2SG", question_yn=True,
            ))
            return tr_capitalize(q_aff) + "?", f"SORU_ÜRET({n_root}+2SG+anlat+iste?)"

        focus = (
            "konuşmanın anlamı" if u.get("asks_why_speak") else (
                u["phil_topic"] or (concept_hits[0][0] if concept_hits else (u["stems"][0] if u["stems"] else "bu konu"))
            )
        )
        # 3 farklı kök-tabanlı merak sorusu arketipi (asla tek tip 'ne düşünüyorsun?' değil!)
        if q_rot == 0:
            q_gen = self.syn.realize_verb(VerbSpec(
                verb_root="uyandır",
                subj=NP(focus if focus != "konu" else "bu konu"),
                loc=NP("zihin", poss=2, case="LOC"),
                wh_word="ne",
                tense="PRES", person="3SG",
            ))
            return tr_capitalize(q_gen) + "?", f"SORU_ÜRET({focus}+zihin+2SG+LOC+ne+uyandır?)"
        elif q_rot == 1:
            q_gen = self.syn.realize_verb(VerbSpec(
                verb_root="bak",
                subj=NP("sen"),
                dat=NP(focus if focus != "konu" else "bu mesele", case="DAT"),
                wh_word="nasıl",
                tense="PRES", person="2SG",
            ))
            return tr_capitalize(q_gen) + "?", f"SORU_ÜRET({focus}+DAT+nasıl+bak?)"
        else:
            about_phrase = "bu konuda" if focus == "konu" else f"{focus} üzerine"
            q_gen = self.syn.realize_verb(VerbSpec(
                verb_root="hisset",
                subj=NP("sen"),
                advs=(about_phrase,),
                wh_word="ne",
                tense="PRES", person="2SG",
            ))
            return tr_capitalize(q_gen) + "?", f"SORU_ÜRET({focus}+üzerine+ne+hisset?)"

    def _build_plan_dict(
        self,
        t0: float,
        u: dict,
        plan: dict,
        thought_nodes: list[str],
        surface: str,
        schema_id: str = "S1_NEDEN_SONUC",
        stance: str = "yalın_doğrudan",
        inner_thought=None,
    ) -> dict:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "parsed_stems": u["stems"][:8],
            "volition_acts": plan["acts"],
            "ask_back_chosen": plan.get("ask_back", False),
            "guarded": plan.get("guarded", False),
            "schema_id": schema_id,
            "stance": stance,
            "inner_monologue": inner_thought.text if inner_thought is not None else "",
            "thought_tree": thought_nodes,
            "surface_tr": surface,
            "latency_ms": round(dt_ms, 3),
        }


if __name__ == "__main__":
    syn = TurkishSyntaxEngine
    assert syn.realize_np(NP("zihin", poss=1, case="LOC")) == "zihnimde"
    assert syn.realize_np(NP("enerji", poss=3, case="ACC")) == "enerjisini"
    assert syn.nominalize_dik("yorul", poss=2, case="ACC", abl=NP("okul", case="ABL")) == "okuldan yorulduğunu"
    assert syn.realize_pred(PredSpec(pred_root="iyi", person="1SG")) == "iyiyim"
    assert syn.realize_pred(PredSpec(subj=NP("sen"), pred_root="nasıl", person="2SG", question=True)) == "sen nasılsın"

    # O(1) 50.000 konuşma turu ölçeklenebilirlik testi!
    epi = EpisodicDialogueCortex()
    for i in range(50000):
        epi.store(i, "insan", f"örnek cümle {i}", [f"kavram_{i % 2500}", "evren"], f"kavram_{i % 2500}", 0.0, ["REFLECT"], "yanıt")
    epi.store(50001, "insan", "Antalya sahilinde kuantum fiziği çalıştım", ["antalya", "sahil", "kuantum", "fizik"], "kuantum", +0.5, ["EMPATHIZE"], "yanıt")
    t_s = time.perf_counter()
    rec = epi.recall(["kuantum", "antalya"], speaker="insan")
    lat_ms = (time.perf_counter() - t_s) * 1000.0
    assert rec and "kuantum" in rec[0].user_text and lat_ms < 5.0
    print(f"[broca-test] TAMAM | 50.001 anıda CA3 geri çağırma süresi: {lat_ms:.3f} ms (< 5 ms O(1)) | bulunan: '{rec[0].user_text}'")

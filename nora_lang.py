#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nora_lang.py — NoRa Bedenlenmiş, Epoch'suz Türkçe Dil ve Kavram Organı (Wernicke–Broca–Kavram Ağı).

Temel ilkeler:
  1. Brute-force epoch eğitimi YOKTUR: Yeni kelimeler, kavramlar ve nedensel kurallar ACh + DA
     kapılı tek-atış (1-shot) sinaptik bağlama (ChemMushroomBody + ilişkisel sinapslar) ile
     TEK CÜMLEDE öğrenilir.
  2. Üretken Türkçe Morfoloji (TurkishMorphology): Eklemeli Türkçe dilbilgisi (ünlü uyumu,
     ünsüz yumuşaması/sertleşmesi, kaynaştırma harfleri, hal ve zaman/kişi ekleri) algoritmik
     ve çift-yönlüdür. Ajan hayatında İLK KEZ duyduğu uydurma bir kelimeyi bile ('zirkon' ->
     'zirkondan', 'blorg' -> 'blorgu') çekimleyip anlamlı cümle içinde kullanabilir.
  3. Yayılan Aktivasyon ve Geçişli/İstisnalı Çıkarım (GroundedConceptGraph):
     Taksonomik (ISA), nedensel (CAUSES), işlevsel (USED_FOR), eylemsel (CAN_DO), özellik
     (HAS_PROP) ve koşullu (IF_THEN) sinapslar üzerinde çok-adımlı çıkarım yapar; doğrudan
     istisnalar ('penguen uçmaz') kalıtımı ('kuşlar uçar') ezer (non-monotonic reasoning).
  4. Wernicke (Anlama + Zamir/Bağlam Çözümleme) & Broca (Önermelerden Cümle Üretimi):
     Doğal Türkçe soruları, öğretileri, komutları, aritmetiği ve öz-benlik sorgularını
     anlamsal çerçeveye (SemanticFrame) çevirir; cevabı kalıp metin yapıştırmadan,
     önerme ağacını Türkçe morfolojiyle çekimleyerek kurar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import numpy as np

from nora_chem import NeuroChemistry, ChemMushroomBody
from nora_knowledge import (
    VOWEL_DROP_STEMS,
    REVERSE_VOWEL_DROP,
    NO_SOFTEN_EXTENDED,
    ANTONYMS,
    FOUNDATIONAL_PROPOSITIONS,
)


BACK_VOWELS = set("aıou")
FRONT_VOWELS = set("eiöü")
ALL_VOWELS = BACK_VOWELS | FRONT_VOWELS
HARD_CONS = set("fstkçşhp")


def tr_lower(s: str) -> str:
    return s.replace("İ", "i").replace("I", "ı").lower()


def tr_capitalize(s: str) -> str:
    if not s:
        return s
    first = "İ" if s[0] == "i" else ("I" if s[0] == "ı" else s[0].upper())
    return first + s[1:]


class TurkishMorphology:
    """Üretken Türkçe İsim/Fiil Çekimleyici, Kök Çözümleyici ve Sayı Okuyucu."""

    # Yumuşamaya girmeyen tek-heceli ve alıntı/istisna kökler + orta hece ünlü düşmesi sözlüğü
    NO_SOFTEN = NO_SOFTEN_EXTENDED
    VOWEL_DROP = VOWEL_DROP_STEMS
    REV_VOWEL_DROP = REVERSE_VOWEL_DROP
    IRREG_NOUN = {
        "su": {"ACC": "suyu", "DAT": "suya", "GEN": "suyun", "POSS1": "suyum"},
        "ne": {"ACC": "neyi", "DAT": "neye", "GEN": "neyin"},
    }
    KNOWN_VERB_ROOTS = {
        "git", "gel", "ye", "iç", "dur", "bekle", "kaç", "gir", "çık", "bak", "gör", "duy",
        "öğren", "bil", "düşün", "anla", "koru", "onar", "uç", "yüz", "koş", "yürü", "parla",
        "ısıt", "soğut", "azalt", "artır", "ver", "al", "yap", "et", "ol", "yaşa", "öl",
        "reddet", "kabul", "inan", "güven", "çöz", "hesapla", "ara", "bul", "göster", "taş", "yayıl", "üret",
        "ilet", "yansıt", "soğur", "buharlaş", "yoğunlaş", "genleş", "çek", "it", "taşı", "erime", "eri", "don",
    }

    @staticmethod
    def last_vowel(word: str) -> str:
        for ch in reversed(tr_lower(word)):
            if ch in ALL_VOWELS:
                return ch
        return "a"

    @classmethod
    def vowel_e_a(cls, word: str) -> str:
        return "e" if cls.last_vowel(word) in FRONT_VOWELS else "a"

    @classmethod
    def vowel_4(cls, word: str) -> str:
        v = cls.last_vowel(word)
        if v in "ei":
            return "i"
        if v in "aı":
            return "ı"
        if v in "öü":
            return "ü"
        return "u"

    @classmethod
    def soften_stem(cls, stem: str) -> str:
        w = tr_lower(stem)
        if not w:
            return w
        if w in cls.VOWEL_DROP and cls.VOWEL_DROP[w] != w:
            return cls.VOWEL_DROP[w]
        if w in cls.NO_SOFTEN or (len(w) <= 2 and w not in {"su", "ağ", "uç"}):
            return w
        if w[-1] in ALL_VOWELS:
            return w
        if w.endswith("nk"):
            return w[:-1] + "g"
        rep = {"p": "b", "ç": "c", "t": "d", "k": "ğ"}
        return (w[:-1] + rep[w[-1]]) if w[-1] in rep else w

    @classmethod
    def derive_word(cls, stem: str, deriv: str) -> str:
        """Üretken Türkçe yapım ekleri: LI (-lı/-li/-lu/-lü), SIZ (-sız/-siz/-suz/-süz),
        LIK (-lık/-lik/-luk/-lük), CI (-cı/-ci/-cu/-cü)."""
        w = tr_lower(stem).strip()
        if not w:
            return w
        v4 = cls.vowel_4(w)
        d = deriv.upper()
        if d == "LI":
            return f"{w}l{v4}"
        if d == "SIZ":
            return f"{w}s{v4}z"
        if d == "LIK":
            return f"{w}l{v4}k"
        if d == "CI":
            ch = "ç" if w[-1] in HARD_CONS else "c"
            return f"{w}{ch}{v4}"
        return w

    @classmethod
    def dental_d_t(cls, word: str) -> str:
        w = tr_lower(word)
        return "t" if (w and w[-1] in HARD_CONS) else "d"

    @classmethod
    def inflect_noun(cls, stem: str, case: str = "NOM", plural: bool = False,
                     poss1: bool = False, copula: bool = False) -> str:
        """İsim çekimi: NOM, ACC (-i), DAT (-e), LOC (-de), ABL (-den), INS (-le), GEN (-in)."""
        w = tr_lower(stem).strip()
        if not w:
            return w
        if not plural and not poss1 and w in cls.IRREG_NOUN and case in cls.IRREG_NOUN[w]:
            res = cls.IRREG_NOUN[w][case]
        else:
            if plural:
                w = w + ("ler" if cls.vowel_e_a(w) == "e" else "lar")
            if poss1:
                if w == "su" and not plural:
                    w = "suyum"
                elif w[-1] in ALL_VOWELS:
                    w = w + "m"
                else:
                    w = cls.soften_stem(w) + cls.vowel_4(w) + "m"
            c = case.upper()
            if c == "NOM":
                res = w
            elif c == "ACC":
                if w[-1] in ALL_VOWELS:
                    res = w + "y" + cls.vowel_4(w)
                else:
                    res = cls.soften_stem(w) + cls.vowel_4(w)
            elif c == "DAT":
                if w[-1] in ALL_VOWELS:
                    res = w + "y" + cls.vowel_e_a(w)
                else:
                    res = cls.soften_stem(w) + cls.vowel_e_a(w)
            elif c == "LOC":
                res = w + cls.dental_d_t(w) + cls.vowel_e_a(w)
            elif c == "ABL":
                res = w + cls.dental_d_t(w) + cls.vowel_e_a(w) + "n"
            elif c == "INS":
                res = w + ("y" if w[-1] in ALL_VOWELS else "") + "l" + cls.vowel_e_a(w)
            elif c == "GEN":
                if w[-1] in ALL_VOWELS:
                    res = w + "n" + cls.vowel_4(w) + "n"
                else:
                    res = cls.soften_stem(w) + cls.vowel_4(w) + "n"
            else:
                res = w
        if copula:
            res = res + cls.dental_d_t(res) + cls.vowel_4(res) + "r"
        return res

    @classmethod
    def conjugate_verb(cls, stem: str, tense: str = "AOR", neg: bool = False, person: str = "3SG") -> str:
        """Fiil çekimi: PRES (-iyor), PAST (-di), FUT (-ecek), AOR (-ir/-ar), NEC (-meli), IMP."""
        v = tr_lower(stem).strip()
        for inf in ("mek", "mak"):
            if len(v) > 3 and v.endswith(inf):
                v = v[:-3]
                break
        if not v:
            return v
        t = tense.upper()
        p = person.upper()

        if t == "IMP":
            if neg:
                base = v + "m" + cls.vowel_e_a(v)
            else:
                base = v
            return base + ("yin" if cls.vowel_e_a(base) == "e" else "yın") if p == "2PL" else base

        if t == "PRES":
            if neg:
                base = v + "m" + cls.vowel_4(v) + "yor"
            else:
                if v in {"git", "et", "tat", "güt"} or v.endswith("det"):
                    v_pres = v[:-1] + "d"
                else:
                    v_pres = v
                if v_pres[-1] in ALL_VOWELS:
                    stem_cut = v_pres[:-1]
                    v4 = cls.vowel_4(stem_cut if stem_cut else v_pres)
                    base = stem_cut + v4 + "yor"
                else:
                    base = v_pres + cls.vowel_4(v_pres) + "yor"
            if p == "1SG":
                return base + "um"
            if p == "2SG":
                return base + "sun"
            if p == "1PL":
                return base + "uz"
            return base

        if t == "PAST":
            b = (v + "m" + cls.vowel_e_a(v)) if neg else v
            dt = cls.dental_d_t(b)
            v4 = cls.vowel_4(b)
            if p == "1SG":
                return f"{b}{dt}{v4}m"
            if p == "2SG":
                return f"{b}{dt}{v4}n"
            if p == "1PL":
                return f"{b}{dt}{v4}k"
            return f"{b}{dt}{v4}"

        if t == "FUT":
            b = (v + "m" + cls.vowel_e_a(v)) if neg else v
            ea = cls.vowel_e_a(b)
            buf = "y" if b[-1] in ALL_VOWELS else ""
            if p == "1SG":
                return f"{b}{buf}{ea}c{ea}ğ{cls.vowel_4(ea)}m"
            if p == "2SG":
                return f"{b}{buf}{ea}c{ea}ks{cls.vowel_4(ea)}n"
            return f"{b}{buf}{ea}c{ea}k"

        if t == "NEC":
            b = (v + "m" + cls.vowel_e_a(v)) if neg else v
            ea = cls.vowel_e_a(b)
            li = "li" if ea == "e" else "lı"
            if p == "1SG":
                return f"{b}m{ea}{li}y{cls.vowel_4(li)}m"
            return f"{b}m{ea}{li}"

        # Varsayılan: Geniş zaman (AOR)
        if neg:
            ea = cls.vowel_e_a(v)
            if p == "1SG":
                return f"{v}m{ea}m"
            if p == "1PL":
                return f"{v}m{ea}y{cls.vowel_4(ea)}z"
            if p == "2SG":
                return f"{v}m{ea}zs{cls.vowel_4(ea)}n"
            return f"{v}m{ea}z"

        if v in {"git", "et"} or v.endswith("det"):
            v_aor = v[:-1] + "d"
        else:
            v_aor = v
        if v_aor[-1] in ALL_VOWELS:
            aor = v_aor + "r"
        else:
            n_vow = sum(1 for c in v_aor if c in ALL_VOWELS)
            # Tek heceli çoğu fiil -ar/-er alır ('uçar', 'yapar', 'koşar'), 'gör/bil/gel/al/ol/ver/dur/bul/kal/var' -ir/-ır/-ür/-ur alır
            ir_singles = {"gör", "bil", "gel", "al", "ol", "ver", "dur", "bul", "kal", "var", "vur", "san"}
            if n_vow == 1 and v_aor not in ir_singles:
                aor = v_aor + cls.vowel_e_a(v_aor) + "r"
            else:
                aor = v_aor + cls.vowel_4(v_aor) + "r"
        if p == "1SG":
            return aor + cls.vowel_4(aor) + "m"
        if p == "2SG":
            return aor + "s" + cls.vowel_4(aor) + "n"
        if p == "1PL":
            return aor + cls.vowel_4(aor) + "z"
        return aor

    @classmethod
    def analyze_word(cls, raw: str, lexicon: set[str] | None = None) -> dict:
        """Kelimeyi (kök, hal, çoğul, olumsuzluk, zaman, koşaç) bileşenlerine ayırır."""
        w = re.sub(r"[^\wçğıöşüÇĞİÖŞÜ-]", "", tr_lower(raw))
        info = {"surface": w, "stem": w, "case": "NOM", "plural": False, "neg": False, "copula": False, "is_verb": False}
        if not w or len(w) <= 2:
            return info
        known = lexicon or set()

        # Özel zamir ve düzensiz biçimler
        pron_map = {
            "suyu": ("su", "ACC"), "suya": ("su", "DAT"), "suda": ("su", "LOC"), "sudan": ("su", "ABL"), "suyun": ("su", "GEN"),
            "bana": ("ben", "DAT"), "beni": ("ben", "ACC"), "bende": ("ben", "LOC"), "benden": ("ben", "ABL"), "benim": ("ben", "GEN"),
            "sana": ("sen", "DAT"), "seni": ("sen", "ACC"), "sende": ("sen", "LOC"), "senden": ("sen", "ABL"), "senin": ("sen", "GEN"),
            "ona": ("o", "DAT"), "onu": ("o", "ACC"), "onda": ("o", "LOC"), "ondan": ("o", "ABL"), "onun": ("o", "GEN"),
            "oraya": ("ora", "DAT"), "orada": ("ora", "LOC"), "oradan": ("ora", "ABL"),
            "buraya": ("bura", "DAT"), "burada": ("bura", "LOC"), "buradan": ("bura", "ABL"),
            "neye": ("ne", "DAT"), "neyi": ("ne", "ACC"), "nerede": ("nere", "LOC"), "nereye": ("nere", "DAT"), "nereden": ("nere", "ABL"),
        }
        if w in pron_map:
            info["stem"], info["case"] = pron_map[w]
            return info

        cur = w
        # 1) Koşaç (-dir/-dır/-tir/-tır/-dür/-dur/-tür/-tur)
        # Türkçe ünsüz benzeşmesi yasası: '-tIr' koşaçı YALNIZCA sert ünsüzden (f,s,t,k,ç,ş,h,p) sonra gelebilir!
        # 'iletir', 'üretir', 'azaltır', 'ısıtır' gibi sözcüklerde '-tir/-tır' koşaç DEĞİL, t-bitimli fiilin geniş zamanıdır!
        for cop in ("dir", "dır", "dur", "dür", "tir", "tır", "tur", "tür"):
            if len(cur) > 4 and cur.endswith(cop):
                if cop.startswith("t") and cur[-4] not in HARD_CONS:
                    continue
                if cur[:-2] in cls.KNOWN_VERB_ROOTS:
                    continue
                cur = cur[:-3]
                info["copula"] = True
                break

        # 2) Basit ve birleşik (yeterlilik/olumsuzluk '-amaz/-emez') fiil çekim ekleri kontrolü
        verb_pats = [
            (r"^(.+?)y?[ae](m[ae])z$", True),
            (r"^(.+?)(m[ae])z$", True),
            (r"^(.+?)y?[ae](m[iıuü])yor(um|sun|uz)?$", True),
            (r"^(.+?)(m[iıuü])yor(um|sun|uz)?$", True),
            (r"^(.+?)[iıuü]yor(um|sun|uz)?$", False),
            (r"^(.+?)y?[ae](m[ae])[dt][iıuü](m|n|k)?$", True),
            (r"^(.+?)(m[ae])[dt][iıuü](m|n|k)?$", True),
            (r"^(.+?)(m[ae])k$", False),
            (r"^(.+?)[ae]c[ae](k|ğim|ğım|ksin|ksın)$", False),
        ]
        for pat, is_neg in verb_pats:
            m = re.match(pat, cur)
            if m and len(m.group(1)) >= 2:
                cand = m.group(1)
                hard_cand = cls._unsoften(cand)
                info["stem"] = hard_cand if hard_cand in cls.KNOWN_VERB_ROOTS or hard_cand in known else cand
                info["neg"] = is_neg
                info["is_verb"] = True
                return info

        # Geniş zaman: önce ünlüyle biten fiil kökleri + '-r' ('yürür'->'yürü', 'uyur'->'uyu'), sonra ünsüz + '-ar/-er/-ir'
        m_aor_v = re.match(r"^(.+[aeıioöuü])r(im|ım|um|üm|sin|sın)?$", cur)
        if m_aor_v and m_aor_v.group(1) in cls.KNOWN_VERB_ROOTS:
            info["stem"] = m_aor_v.group(1)
            info["is_verb"] = True
            return info
        m_aor = re.match(r"^(.+?)([aeiıuü]r)(im|ım|um|üm|sin|sın)?$", cur)
        if m_aor and cur not in known and not cur.endswith(("lar", "ler")):
            cand = m_aor.group(1)
            hard_cand = cls._unsoften(cand)
            if cand in cls.KNOWN_VERB_ROOTS or hard_cand in cls.KNOWN_VERB_ROOTS:
                info["stem"] = hard_cand if hard_cand in cls.KNOWN_VERB_ROOTS else cand
                info["is_verb"] = True
                return info

        # 3) İsim hal ekleri (ABL -> LOC -> GEN -> INS -> DAT -> ACC)
        case_rules = [
            (("dan", "den", "tan", "ten"), "ABL"),
            (("da", "de", "ta", "te"), "LOC"),
            (("nın", "nin", "nun", "nün", "ın", "in", "un", "ün"), "GEN"),
            (("yla", "yle", "la", "le"), "INS"),
            (("ya", "ye", "a", "e"), "DAT"),
            (("yı", "yi", "yu", "yü", "ı", "i", "u", "ü"), "ACC"),
        ]
        if cur not in known:
            for suffixes, cname in case_rules:
                matched = False
                for sf in suffixes:
                    if len(cur) - len(sf) >= 2 and cur.endswith(sf):
                        cand = cur[:-len(sf)]
                        # Türkçe ünsüz benzeşmesi yasası: 't' ile başlayan hal eki (-ta/-te/-tan/-ten) YALNIZCA sert ünsüzden sonra gelebilir!
                        if sf[0] == "t" and (not cand or cand[-1] not in HARD_CONS):
                            continue
                        # 'a'/'e'/'ı'/'i' tek harfli eklerde kelimenin zaten sözlükte kök olup olmadığını koru;
                        # ancak ünsüz yumuşaması ('ğ/b/c/d' + ünlü) veya sert ünsüz + ünlü ('sepete' -> 'sepet') hal ekidir!
                        hard = cls._unsoften(cand)
                        softened = cls._likely_softened(cand, sf)
                        vdrop = (cand in cls.REV_VOWEL_DROP)
                        hard_dat_acc = (cand[-1] in HARD_CONS and len(cand) >= 3)
                        if len(sf) == 1 and (cur in known or (not softened and not vdrop and not hard_dat_acc and cand not in known and hard not in known and not cand.endswith(("lar", "ler")))):
                            continue
                        cur = hard if softened else cand
                        info["case"] = cname
                        matched = True
                        break
                if matched:
                    break

        # 4) Çoğul eki (-lar/-ler) ve iyelik (-ım/-im/-um/-üm / -ı/-i/-u/-ü) + Ünlü Düşmesi geri kazanımı
        for pl in ("lar", "ler"):
            if len(cur) > 4 and cur.endswith(pl) and cur not in known:
                cur = cur[:-3]
                info["plural"] = True
                break
        for poss in ("ımız", "imiz", "umuz", "ümüz", "ım", "im", "um", "üm", "ın", "in", "un", "ün", "m"):
            if len(cur) - len(poss) >= 3 and cur.endswith(poss) and cur not in known:
                cand = cur[:-len(poss)]
                if cand in cls.REV_VOWEL_DROP:
                    cur = cls.REV_VOWEL_DROP[cand]
                    info["poss1"] = True
                    info["vowel_drop"] = True
                    break
                hard = cls._unsoften(cand)
                if cand in known or hard in known:
                    cur = hard if hard in known else cand
                    info["poss1"] = True
                    break

        if cur in cls.REV_VOWEL_DROP:
            cur = cls.REV_VOWEL_DROP[cur]
            info["vowel_drop"] = True

        info["stem"] = cur
        return info

    @classmethod
    def explain_morphology_tr(cls, raw: str, lexicon: set[str] | None = None) -> str:
        """Bir Türkçe sözcüğü kök, ses olayı ve çekim/yapım eklerine ayırarak uzman düzeyde açıklar."""
        w = re.sub(r"[^\wçğıöşüÇĞİÖŞÜ]", "", tr_lower(raw))
        if not w:
            return "Çözümlenecek geçerli bir sözcük bulamadım."

        # Birleşik/modal fiil çözümlemesi (ör. 'görebiliyordum', 'öğretmeliyiz', 'anlayamayacak')
        m_comp = re.match(
            r"^(.+?)(?:([ae]bil)[iıuü]?|([ae]m[ae])y?)?(?:(m[ae]|m[iıuü]))?(?:[iıuü])?(yor|[ae]c[ae][kğ]|m[ae]l[iı]|[dt][iıuü]|m[iıuü]ş)(?:([dt][iıuü]|s[ae]))?(m|n|k|uz|üz|ız|iz|sun|sün|sın|sin|yım|yim|yum|yüm|yız|yiz)?$",
            w,
        )
        if m_comp and len(m_comp.group(1)) >= 2:
            root_cand = m_comp.group(1)
            hard_root = cls._unsoften(root_cand)
            if root_cand in cls.KNOWN_VERB_ROOTS or hard_root in cls.KNOWN_VERB_ROOTS or m_comp.group(2) or m_comp.group(5) in ("yor", "meli", "malı"):
                vroot = hard_root if hard_root in cls.KNOWN_VERB_ROOTS else root_cand
                morphemes = [f"'{vroot}' (fiil kökü)"]
                if m_comp.group(2):
                    morphemes.append(f"'-{m_comp.group(2)}' (yeterlilik kurallı birleşik fiili)")
                if m_comp.group(3):
                    morphemes.append(f"'-{m_comp.group(3)}' (olumsuz yeterlilik)")
                if m_comp.group(4):
                    morphemes.append(f"'-{m_comp.group(4)}' (olumsuzluk eki)")
                tense_s = m_comp.group(5)
                tense_map = {"yor": "şimdiki zaman", "meli": "gereklilik kipi", "malı": "gereklilik kipi"}
                t_lbl = tense_map.get(tense_s, "gelecek zaman" if "c" in tense_s else ("duyulan geçmiş zaman" if "ş" in tense_s else "görülen geçmiş zaman"))
                morphemes.append(f"'-{tense_s}' ({t_lbl})")
                if m_comp.group(6):
                    sec_lbl = "hikaye birleşik zaman" if m_comp.group(6)[0] in "dt" else "şart birleşik zaman"
                    morphemes.append(f"'-{m_comp.group(6)}' ({sec_lbl})")
                if m_comp.group(7):
                    morphemes.append(f"'-{m_comp.group(7)}' (kişi eki)")
                return f"'{w}' sözcüğünün morfolojik çözümlemesi: " + " + ".join(morphemes) + "."

        info = cls.analyze_word(w, lexicon=lexicon)
        parts = [f"'{info['stem']}' ({'fiil' if info.get('is_verb') else 'isim'} kökü)"]
        if info.get("vowel_drop"):
            parts.append("[orta hece ünlü düşmesi]")
        elif info["stem"] != w and info["stem"] and info["stem"][-1] in {"p", "ç", "t", "k"} and any(c in w for c in ("b", "c", "d", "ğ")):
            parts.append("[ünsüz yumuşaması]")
        if info.get("plural"):
            parts.append("çoğul eki (-lar/-ler)")
        if info.get("poss1"):
            parts.append("iyelik eki")
        case_names = {
            "ACC": "belirtme hali (-i)", "DAT": "yönelme hali (-e)", "LOC": "bulunma hali (-de)",
            "ABL": "ayrılma/çıkma hali (-den)", "INS": "vasıta hali (-le)", "GEN": "tamlayan/ilgi hali (-in)",
        }
        if info.get("case", "NOM") != "NOM":
            parts.append(case_names[info["case"]])
        if info.get("copula"):
            parts.append("bildirme/ek-eylem (-dir)")
        if info.get("neg"):
            parts.append("olumsuzluk eki (-ma/-me)")
        return f"'{w}' sözcüğünün morfolojik çözümlemesi: " + " + ".join(parts) + "."

    @staticmethod
    def _likely_softened(cand: str, sf: str) -> bool:
        return bool(sf and sf[0] in ALL_VOWELS and cand and cand[-1] in {"b", "c", "d", "ğ"})

    @staticmethod
    def _unsoften(cand: str) -> str:
        if not cand:
            return cand
        rev = {"b": "p", "c": "ç", "d": "t", "ğ": "k"}
        return (cand[:-1] + rev[cand[-1]]) if cand[-1] in rev else cand

    # ------------------------------------------------------------------ Türkçe sayı sistemi
    UNITS = {"sıfır": 0, "bir": 1, "iki": 2, "üç": 3, "dört": 4, "beş": 5, "altı": 6, "yedi": 7, "sekiz": 8, "dokuz": 9}
    TENS = {"on": 10, "yirmi": 20, "otuz": 30, "kırk": 40, "elli": 50, "altmış": 60, "yetmiş": 70, "seksen": 80, "doksan": 90}
    REV_U = {v: k for k, v in UNITS.items() if v > 0}
    REV_T = {v: k for k, v in TENS.items()}

    @classmethod
    def normalize_num_token(cls, tok: str) -> str | None:
        """Hal/iyelik eki almış Türkçe sayı kelimelerini ('sekizin'->'sekiz', 'dördün'->'dört', 'beşten'->'beş') yalın sayı köküne indirger."""
        t = tr_lower(tok).strip()
        if re.fullmatch(r"[+-]?\d+", t):
            return t
        num_set = set(cls.UNITS) | set(cls.TENS) | {"yüz", "bin", "eksi"}
        if t in num_set:
            return t
        for suf in ("nın", "nin", "nun", "nün", "ın", "in", "un", "ün", "dan", "den", "tan", "ten", "ya", "ye", "a", "e", "yı", "yi", "yu", "yü", "ı", "i", "u", "ü"):
            if len(t) - len(suf) >= 2 and t.endswith(suf):
                cand = t[:-len(suf)]
                if cand in num_set:
                    return cand
                hard = cls._unsoften(cand)
                if hard in num_set:
                    return hard
        return None

    @classmethod
    def words_to_int(cls, tokens: list[str]) -> int | None:
        """Türkçe sayı dizisini ('iki', 'yüz', 'kırk', 'yedi' veya 'sekizin' -> 247/248) tam sayıya çevirir."""
        if not tokens:
            return None
        sign = 1
        idx = 0
        if tr_lower(tokens[0]) == "eksi":
            sign = -1
            idx = 1
        total = 0
        current = 0
        used = False
        for tok in tokens[idx:]:
            t = cls.normalize_num_token(tok)
            if t is None:
                break
            if re.fullmatch(r"[+-]?\d+", t):
                return sign * int(t)
            if t in cls.UNITS:
                current += cls.UNITS[t]
                used = True
            elif t in cls.TENS:
                current += cls.TENS[t]
                used = True
            elif t == "yüz":
                current = (1 if current == 0 else current) * 100
                used = True
            elif t == "bin":
                total += (1 if current == 0 else current) * 1000
                current = 0
                used = True
            else:
                break
        return (sign * (total + current)) if used else None

    @classmethod
    def int_to_words(cls, n: int) -> str:
        if n == 0:
            return "sıfır"
        neg = n < 0
        n = abs(int(n))
        parts = []
        if n >= 1000:
            th = n // 1000
            if th > 1:
                parts.append(cls.int_to_words(th))
            parts.append("bin")
            n %= 1000
        if n >= 100:
            hu = n // 100
            if hu > 1:
                parts.append(cls.REV_U[hu])
            parts.append("yüz")
            n %= 100
        if n >= 10:
            te = (n // 10) * 10
            parts.append(cls.REV_T[te])
            n %= 10
        if n > 0:
            parts.append(cls.REV_U[n])
        return ("eksi " if neg else "") + " ".join(parts)


@dataclass
class RelationEdge:
    subj: str
    rel: str                  # ISA | HAS_PROP | CAN_DO | CANNOT_DO | CAUSES | USED_FOR | LOCATED_AT | IF_THEN | OPPOSITE
    obj: str
    weight: float = 1.0
    source: str = "duyum"     # deneyim | duyum | doğuştan
    speaker: str = "insan"


@dataclass
class Proposition:
    """Broca cümle üreticisine giden atomik düşünce önermesi."""
    subj: str
    rel: str
    obj: str = ""
    neg: bool = False
    tense: str = "AOR"
    person: str = "3SG"
    source: str = "deneyim"
    reason: "Proposition | None" = None


class GroundedConceptGraph:
    """Bedenlenmiş kavram uzayı + 1-atış ilişkisel sinaps ağı + yayılan aktivasyon çıkarımı."""

    def __init__(self, chem: NeuroChemistry, seed: int = 0):
        self.chem = chem
        self.rng = np.random.default_rng(seed)
        self.lexicon: set[str] = set()
        self.edges: list[RelationEdge] = []
        # O(1) yerel sinaptik komşuluk indeksleri (veri 100.000 kenara çıksa da sorgu gecikmesi ~0.05 ms kalır!)
        self._out_edges: dict[str, list[RelationEdge]] = {}
        self._edge_map: dict[tuple[str, str, str], RelationEdge] = {}
        self.concept_vecs: dict[str, np.ndarray] = {}
        self.concept_id: dict[str, int] = {}
        self.id_concept: list[str] = []
        # Karakter-üçlüsü (trigram) -> seyrek Kenyon hücresi ile fonolojik-kavramsal çağrışım!
        self.word_mb = ChemMushroomBody(n_in=128, n_out=32, n_kc=2048, seed=seed + 11)
        self._bootstrap_embodied_primitives()

    def clear_all_edges(self):
        self.edges.clear()
        self._out_edges.clear()
        self._edge_map.clear()

    def _trigram_vec(self, text: str) -> np.ndarray:
        v = np.zeros(128, dtype=np.float32)
        s = f"<{tr_lower(text)}>"
        for i in range(max(1, len(s) - 2)):
            tri = s[i:i + 3]
            h = (hash(tri) & 0x7FFFFFFF) % 128
            h2 = ((hash(tri + "!") & 0x7FFFFFFF) % 128)
            v[h] += 1.0
            v[h2] -= 0.5
        return v

    def register_concept(self, name: str, bind_mb: bool = True) -> int:
        c = tr_lower(name).strip()
        if not c:
            return -1
        self.lexicon.add(c)
        for part in c.split():
            self.lexicon.add(part)
        if c not in self.concept_id:
            cid = len(self.id_concept)
            self.concept_id[c] = cid
            self.id_concept.append(c)
            if bind_mb and cid < 384:
                vec = self._trigram_vec(c)
                self.concept_vecs[c] = vec
                self.word_mb.learn_oneshot(vec, cid, chem=self.chem, strength=0.8, contrastive=False)
        return self.concept_id[c]

    def _bootstrap_embodied_primitives(self):
        """Ansiklopedi EZBERİ YOK; yalnızca bedenin doğuştan sahip olduğu duyu-motor kökler."""
        base_words = [
            "ben", "sen", "nora", "merhaba", "selam", "nasıl", "su", "yiyecek", "besin", "yem", "elma", "tehlike", "diken", "zehir",
            "duvar", "kaya", "enerji", "susuzluk", "açlık", "bütünlük", "acı", "yara", "ölüm", "yaşam",
            "gündüz", "gece", "karanlık", "aydınlık", "harita", "konum", "görme", "hareket", "ayna",
            "kuzey", "güney", "doğu", "batı", "ileri", "sol", "sağ", "faydalı", "zararlı", "güvenli", "tehlikeli",
        ]
        for w in base_words:
            self.register_concept(w)
        # Yalnızca temel bedensel eşanlam/kategori bağları
        prim = [
            ("elma", "ISA", "yiyecek"), ("yem", "ISA", "yiyecek"), ("besin", "ISA", "yiyecek"),
            ("diken", "ISA", "tehlike"),
            ("su", "USED_FOR", "susuzluk giderme"),
            ("yiyecek", "USED_FOR", "enerji kazanma"),
            ("tehlike", "CAUSES", "bütünlük kaybı"),
            ("sıcak", "OPPOSITE", "soğuk"), ("soğuk", "OPPOSITE", "sıcak"),
            ("aydınlık", "OPPOSITE", "karanlık"), ("karanlık", "OPPOSITE", "aydınlık"),
            ("faydalı", "OPPOSITE", "zararlı"), ("zararlı", "OPPOSITE", "faydalı"),
        ]
        for s, r, o in prim:
            self.add_edge(s, r, o, source="doğuştan", weight=0.9)
        for s, r, o in FOUNDATIONAL_PROPOSITIONS:
            self.add_edge(s, r, o, source="doğuştan", weight=0.88)
        for k_ant, v_ant in ANTONYMS.items():
            self.add_edge(k_ant, "OPPOSITE", v_ant, source="doğuştan", weight=0.85)

    def add_edge(self, subj: str, rel: str, obj: str, source: str = "duyum",
                 speaker: str = "insan", weight: float = 1.0, bind_mb: bool = True) -> RelationEdge:
        s = tr_lower(subj).strip()
        o = tr_lower(obj).strip()
        self.register_concept(s, bind_mb=bind_mb)
        self.register_concept(o, bind_mb=bind_mb)
        # O(1) mevcut sinaps kontrolü + zıt kutuplu çelişen eski 'duyum' bağını zayıflatma
        existing = self._edge_map.get((s, rel, o))
        if existing is not None:
            existing.weight = float(min(2.0, existing.weight + 0.4 * weight))
            if source == "deneyim":
                existing.source = "deneyim"
            return existing
        opposite_rel = {"CAN_DO": "CANNOT_DO", "CANNOT_DO": "CAN_DO"}.get(rel)
        if opposite_rel is not None:
            opp_edge = self._edge_map.get((s, opposite_rel, o))
            if opp_edge is not None and opp_edge.source != "deneyim":
                opp_edge.weight *= 0.2

        edge = RelationEdge(subj=s, rel=rel, obj=o, weight=float(weight), source=source, speaker=speaker)
        self.edges.append(edge)
        self._out_edges.setdefault(s, []).append(edge)
        self._edge_map[(s, rel, o)] = edge
        if source != "doğuştan":
            self.chem.pulse_learning(novelty=0.80, valence=0.35)
        return edge

    def ancestors(self, subj: str, max_hops: int = 4) -> list[tuple[str, list[RelationEdge]]]:
        """O(k) yerel sinaps yayılımı ile ISA taksonomi zincirinde yukarı doğru aktivasyon."""
        s = tr_lower(subj).strip()
        visited = {s}
        frontier: list[tuple[str, list[RelationEdge]]] = [(s, [])]
        out: list[tuple[str, list[RelationEdge]]] = [(s, [])]
        for _ in range(max_hops):
            nxt = []
            for cur, path in frontier:
                for e in self._out_edges.get(cur, ()):
                    if e.rel == "ISA" and e.weight > 0.25 and e.obj not in visited:
                        visited.add(e.obj)
                        npath = path + [e]
                        out.append((e.obj, npath))
                        nxt.append((e.obj, npath))
            frontier = nxt
            if not frontier:
                break
        return out

    def query(self, subj: str, rels: tuple[str, ...] | None = None) -> list[tuple[RelationEdge, list[RelationEdge]]]:
        """O(k) yerel sinaps komşuluğu üzerinden doğrudan + kalıtsal özellikleri döndürür."""
        chain = self.ancestors(subj)
        blocked_actions: set[str] = set()
        results: list[tuple[RelationEdge, list[RelationEdge]]] = []
        for anc, isa_path in chain:
            for e in self._out_edges.get(anc, ()):
                if e.weight <= 0.25:
                    continue
                if e.rel == "CANNOT_DO":
                    blocked_actions.add(e.obj)
                if rels is not None and e.rel not in rels:
                    continue
                if e.rel == "CAN_DO" and e.obj in blocked_actions and len(isa_path) > 0:
                    continue
                results.append((e, isa_path))
        return results

    def causal_chain(self, subj: str, max_hops: int = 4) -> list[tuple[str, list[RelationEdge]]]:
        """O(k) yerel sinaps komşuluğu üzerinden CAUSES ve IF_THEN çok-adımlı ileri yayılımı."""
        s = tr_lower(subj).strip()
        visited = {s}
        frontier: list[tuple[str, list[RelationEdge]]] = [(s, [])]
        out: list[tuple[str, list[RelationEdge]]] = []
        for _ in range(max_hops):
            nxt = []
            for cur, path in frontier:
                for e in self._out_edges.get(cur, ()):
                    if e.rel in ("CAUSES", "IF_THEN") and e.weight > 0.25 and e.obj not in visited:
                        visited.add(e.obj)
                        npath = path + [e]
                        out.append((e.obj, npath))
                        nxt.append((e.obj, npath))
            frontier = nxt
            if not frontier:
                break
        return out

    def check_relation(self, subj: str, target_pred: str) -> dict:
        """'X Y yapar mı / X Y midir / X Y'ye yol açar mı?' sorusu için geçişli kanıt zinciri arar."""
        s = tr_lower(subj).strip()
        t = tr_lower(target_pred).strip()
        t_clean = re.sub(r"\s*(yol açar|neden olur)$", "", t).strip()
        t_parts = t_clean.split() if t_clean else []
        t_stem = TurkishMorphology.analyze_word(t_parts[-1], self.lexicon)["stem"] if t_parts else ""
        t_phrase_stem = (" ".join(t_parts[:-1] + [t_stem])).strip() if len(t_parts) > 1 else t_stem

        # 1) Doğrudan veya kalıtsal negatif istisna var mı?
        for e, isa_path in self.query(s, ("CANNOT_DO",)):
            e_last = TurkishMorphology.analyze_word(e.obj.split()[-1], self.lexicon)["stem"] if e.obj else ""
            if e.obj in (t, t_stem, t_phrase_stem) or (len(t_parts) >= 1 and e_last == t_stem and e.obj.split()[0] == t_parts[0]):
                return {"answer": False, "edge": e, "isa_path": isa_path}

        # 2) Doğrudan veya ISA-kalıtsal pozitif bağ var mı?
        for e, isa_path in self.query(s, ("ISA", "HAS_PROP", "CAN_DO", "CAUSES", "USED_FOR", "LOCATED_AT")):
            e_stem = TurkishMorphology.analyze_word(e.obj.split()[-1], self.lexicon)["stem"] if e.obj else ""
            if e.obj == t or e.obj == t_clean or e.obj == t_stem or e_stem == t_stem or t_clean in e.obj:
                return {"answer": True, "edge": e, "isa_path": isa_path}

        # 3) Çok-adımlı nedensel zincir (A -> B -> C): ör. 'kuraklık güçsüzlüğe yol açar mı?'
        for eff, cpath in self.causal_chain(s):
            eff_stem = TurkishMorphology.analyze_word(eff.split()[-1], self.lexicon)["stem"] if eff else ""
            if eff == t_clean or eff == t_stem or eff_stem == t_stem:
                return {"answer": True, "edge": cpath[-1], "isa_path": [], "causal_path": cpath}

        # 4) Zıt özellik kontrolü (ör. 'ateş soğuk mudur?' -> ateş sıcaktır, sıcak != soğuk -> Hayır!)
        opposites = {e.obj for e in self._out_edges.get(t_stem, ()) if e.rel == "OPPOSITE"}
        if opposites:
            for e, isa_path in self.query(s, ("HAS_PROP",)):
                if e.obj in opposites:
                    return {"answer": False, "edge": e, "isa_path": isa_path, "contrast": True}

        return {"answer": None, "edge": None, "isa_path": []}

    def solve_analogy(self, a: str, b: str, c: str) -> tuple[str | None, str]:
        """'a : b :: c : ?' analojisini a->b arasındaki ilişkisel sinapsı bularak c'ye uygular."""
        a_l, b_l, c_l = tr_lower(a), tr_lower(b), tr_lower(c)
        # 1) Doğrudan ilişki türü eşleşmesi
        for e_ab, _ in self.query(a_l):
            if e_ab.obj == b_l:
                for e_c, _ in self.query(c_l, (e_ab.rel,)):
                    return e_c.obj, e_ab.rel
        # 2) Zıtlık ilişkisi
        for e in self._out_edges.get(a_l, ()):
            if e.rel == "OPPOSITE" and e.obj == b_l:
                for e2 in self._out_edges.get(c_l, ()):
                    if e2.rel == "OPPOSITE":
                        return e2.obj, "OPPOSITE"
        return None, ""


class BrocaGenerator:
    """Önerme ağaçlarını (Proposition) üretken Türkçe morfolojiyle doğal cümlelere dönüştürür."""

    def __init__(self):
        self.morph = TurkishMorphology()
        self.connective_counts: dict[str, int] = {"çünkü": 1, "dolayısıyla": 1, "ancak": 1}

    def realize_prop(self, p: Proposition) -> str:
        s = p.subj.strip()
        o = p.obj.strip()
        r = p.rel

        if r == "ISA":
            art = "" if o.startswith("bir ") else "bir "
            pred = self.morph.inflect_noun(o, "NOM", copula=not p.neg)
            return f"{s} {art}{o} değildir" if p.neg else f"{s} {art}{pred}"

        if r == "HAS_PROP":
            if p.neg:
                return f"{s} {o} değildir"
            return f"{s} {self.morph.inflect_noun(o, 'NOM', copula=True)}"

        if r in ("CAN_DO", "CANNOT_DO"):
            is_neg = p.neg or (r == "CANNOT_DO")
            parts = o.split()
            if len(parts) == 1:
                v_form = self.morph.conjugate_verb(parts[0], tense=p.tense, neg=is_neg, person=p.person)
                return f"{s} {v_form}".strip()
            v_form = self.morph.conjugate_verb(parts[-1], tense=p.tense, neg=is_neg, person=p.person)
            return f"{s} {' '.join(parts[:-1])} {v_form}".strip()

        if r == "CAUSES":
            if p.neg:
                return f"{s} {self.morph.inflect_noun(o, 'DAT')} yol açmaz"
            return f"{s} {self.morph.inflect_noun(o, 'DAT')} yol açar"

        if r == "USED_FOR":
            return f"{s} {o} için kullanılır"

        if r == "LOCATED_AT":
            loc = self.morph.inflect_noun(o, "LOC")
            return f"{s} {loc} bulunur"

        if r == "IF_THEN":
            return f"{s} olduğunda {o}"

        if r == "ACTION_SELF":
            v_form = self.morph.conjugate_verb(o, tense=p.tense, neg=p.neg, person="1SG")
            return f"{s} {v_form}".strip() if s else v_form

        return f"{s} {o}".strip()

    def compose(self, main: Proposition, reason: Proposition | None = None,
                prefix: str = "", epistemic_note: str = "") -> str:
        """Ana önerme + nedensel/çıkarımsal yan cümle + epistemik kaynak zarfını birleştirir."""
        core = self.realize_prop(main)
        if reason is not None:
            r_txt = self.realize_prop(reason)
            self.connective_counts["çünkü"] += 1
            core = f"{core}; çünkü {r_txt}"
        if epistemic_note:
            core = f"{epistemic_note} {core}"
        if prefix:
            core = f"{prefix}, {core}"
        out = tr_capitalize(core.strip())
        if out and out[-1] not in ".!?":
            out += "."
        return out

    def compose_inference_chain(self, answer: bool, subj: str, pred: str,
                                edge: RelationEdge, isa_path: list[RelationEdge]) -> str:
        """Geçişli çıkarım zincirini ('karga -> kuş -> uçar') adım adım cümleye döker."""
        if not isa_path:
            # Doğrudan bilgi
            main = Proposition(subj=subj, rel=edge.rel, obj=edge.obj, neg=(not answer and edge.rel != "CANNOT_DO"))
            prefix = "Evet" if answer else "Hayır"
            ep = "deneyimime göre" if edge.source == "deneyim" else ""
            return self.compose(main, prefix=prefix, epistemic_note=ep)

        # Geçişli çıkarım: 'Evet, karga bir kuştur ve kuş uçar; dolayısıyla karga uçar.'
        first_isa = isa_path[0]
        p1 = self.realize_prop(Proposition(subj=first_isa.subj, rel="ISA", obj=first_isa.obj))
        p2 = self.realize_prop(Proposition(subj=edge.subj, rel=edge.rel, obj=edge.obj))
        concl = self.realize_prop(Proposition(subj=subj, rel=edge.rel if answer else "CANNOT_DO",
                                              obj=edge.obj if not pred else edge.obj))
        prefix = "Evet" if answer else "Hayır"
        txt = f"{prefix}, {p1} ve {p2}; dolayısıyla {concl}"
        return tr_capitalize(txt) + "."


@dataclass
class SemanticFrame:
    kind: str                 # self_query | body_query | world_query | math_query | analogy_query | yn_query | wh_query | command | teach_fact | teach_val | why_query | greeting | unknown
    raw: str
    subj: str = ""
    rel: str = ""
    obj: str = ""
    valence: float = 0.0
    math_expr: tuple[int, str, int] | None = None
    analogy: tuple[str, str, str] | None = None
    sub_kind: str = ""


class WernickeComprehender:
    """Doğal Türkçe cümleleri yapısal SemanticFrame'e çevirir ve zamir bağlamını (Working Memory) izler."""

    INTENT_LABELS = [
        "greeting", "self_query", "body_query", "world_query",
        "math_query", "command", "teach_fact", "yn_query", "wh_query", "why_query",
    ]

    def __init__(self, graph: GroundedConceptGraph, chem: NeuroChemistry, seed: int = 0):
        self.graph = graph
        self.chem = chem
        self.morph = TurkishMorphology()
        self.intent_mb = ChemMushroomBody(n_in=128, n_out=len(self.INTENT_LABELS), n_kc=1536, seed=seed + 23)
        self.last_topic: str = ""
        self.last_explanation: str = ""
        self.turn_count: int = 0
        self._seed_intent_prototypes()

    def _sent_vec(self, text: str) -> np.ndarray:
        v = np.zeros(128, dtype=np.float32)
        s = tr_lower(text)
        for tok in s.split():
            for i in range(max(1, len(tok) - 1)):
                bg = tok[i:i + 2]
                v[(hash(bg) & 0x7FFFFFFF) % 128] += 1.0
        return v

    def _seed_intent_prototypes(self):
        seeds = [
            ("merhaba selam nasılsın günaydın", 0),
            ("sen kimsin nesin yaşıyor musun adın ne bilincin var mı", 1),
            ("aç mısın susadın mı enerjin kaç suyun ne durumda görebiliyor musun hareket edebiliyor musun", 2),
            ("neredesin su nerede konumun ne haritada ne var", 3),
            ("kaç eder artı eksi topla çıkar hesapla", 4),
            ("suya git ileri yürü dur tehlikeye gir kuzeye yönel", 5),
            ("bir kuştur enerji verir zararlıdır faydalıdır uçar", 6),
            ("mı mi mu mü uçar mı zararlı mı", 7),
            ("nedir ne yapar ne işe yarar nasıldır özelliği nedir", 8),
            ("neden niçin niye nasıl karar verdin", 9),
        ]
        for txt, lbl in seeds:
            self.intent_mb.learn_oneshot(self._sent_vec(txt), lbl, chem=None, strength=0.9, contrastive=False)

    DISCOURSE_PARTICLES = {"peki", "acaba", "yani", "öyleyse", "pekala", "şey"}

    def _resolve_anaphora(self, tokens: list[str]) -> list[str]:
        """Söylem başı bağlaçlarını temizler ve çalışma belleğindeki son konuyu ('o', 'onun', 'ona', 'ondan') bağlar."""
        toks = list(tokens)
        while toks and re.sub(r"[^\wçğıöşü]", "", tr_lower(toks[0])) in self.DISCOURSE_PARTICLES:
            toks.pop(0)
        if not self.last_topic or not toks:
            return toks
        out = []
        pron_cases = {
            "o": "NOM", "onu": "ACC", "ona": "DAT", "onda": "LOC", "ondan": "ABL", "onun": "GEN",
        }
        for i, tok in enumerate(toks):
            clean = re.sub(r"[^\wçğıöşü]", "", tr_lower(tok))
            # 'o' tek başına özne konumundaysa (ardından isim gelmiyorsa veya ardından 'bir' geliyorsa) son konuya çözülür
            next_clean = re.sub(r"[^\wçğıöşü]", "", tr_lower(toks[i + 1])) if i + 1 < len(toks) else ""
            if clean in pron_cases and (clean != "o" or i == 0 and (next_clean in {"bir", "ne", "nedir", "nasıldır", "uçar", "yapar", "zararlı", "faydalı"} or i + 2 >= len(toks))):
                out.append(self.morph.inflect_noun(self.last_topic, pron_cases[clean]))
            else:
                out.append(tok)
        return out

    def _extract_math(self, s: str) -> tuple[int, str, int] | None:
        # 1) Rakamlı ifade: '247 + 389', '84 eksi 59', '125 ile 47 toplamı'
        m = re.search(r"(-?\d+)\s*(\+|\-|artı|eksi|topla|çıkar|ile)\s*(-?\d+)", s)
        if m:
            a, op_raw, b = int(m.group(1)), m.group(2), int(m.group(3))
            if op_raw == "ile":
                op = "-" if ("çıkar" in s or "fark" in s or "eksi" in s) else "+"
            else:
                op = "-" if op_raw in ("-", "eksi", "çıkar") else "+"
            return (a, op, b)

        # 2) Yazıyla Türkçe sayı ifadesi: 'iki yüz kırk yedi artı üç yüz seksen dokuz kaç eder' / 'sekizin toplamı'
        for kw, op in ((" artı ", "+"), (" eksi ", "-"), (" ile ", "+" if "fark" not in s else "-")):
            if kw in s and ("kaç" in s or "eder" in s or "toplam" in s or "fark" in s or "sonuç" in s or "?" in s):
                left, right = s.split(kw, 1)
                ltoks = [nt for t in re.findall(r"[\wçğıöşü]+", left) if (nt := TurkishMorphology.normalize_num_token(t)) is not None]
                rtoks = []
                for t in re.findall(r"[\wçğıöşü]+", right):
                    nt = TurkishMorphology.normalize_num_token(t)
                    if nt is not None:
                        rtoks.append(nt)
                    elif rtoks:
                        break
                a = TurkishMorphology.words_to_int(ltoks)
                b = TurkishMorphology.words_to_int(rtoks)
                if a is not None and b is not None:
                    return (a, op, b)
        return None

    def parse(self, raw_text: str) -> SemanticFrame:
        self.turn_count += 1
        s_raw = raw_text.strip()
        s = tr_lower(s_raw)
        raw_tokens = s.split()
        tokens = self._resolve_anaphora(raw_tokens)
        s = " ".join(tokens)
        words = re.findall(r"[\wçğıöşü]+", s)
        causal_neden = any(p in s for p in ("neden olur", "neden oldu", "neden oluyor", "neden olmaz"))
        q_cues = {"mı", "mi", "mu", "mü", "mısın", "misin", "musun", "müsün",
                  "ne", "neye", "neyi", "nedir", "kim", "kimsin", "nerede", "neredesin",
                  "niçin", "niye", "nasıl", "nasılsın", "kaç", "kaçtır", "hangi"}
        if not causal_neden:
            q_cues.add("neden")
        has_q = ("?" in s_raw) or any(w in q_cues for w in words)

        # 1) Analoji sorgusu: 'ateş sıcak ise buz nedir' / 'ateş : sıcak :: buz : ?'
        m_an = re.search(r"([\wçğıöşü]+)\s*:\s*([\wçğıöşü]+)\s*::\s*([\wçğıöşü]+)\s*:\s*\?", s)
        if not m_an:
            m_an = re.search(r"([\wçğıöşü]+)\s+([\wçğıöşü]+)\s+ise\s+([\wçğıöşü]+)\s+nedir", s)
        if m_an:
            return SemanticFrame(kind="analogy_query", raw=s_raw, analogy=(m_an.group(1), m_an.group(2), m_an.group(3)))

        # 2) Matematik / Aritmetik sorgusu
        math_ex = self._extract_math(s)
        if math_ex is not None:
            return SemanticFrame(kind="math_query", raw=s_raw, math_expr=math_ex)

        # 3) Selamlama / Hal-hatır (başka bir soru içermiyorsa)
        if not has_q and words and words[0] in {"merhaba", "selam", "günaydın", "iyi"} and len(words) <= 4:
            return SemanticFrame(kind="greeting", raw=s_raw)
        if set(words) <= {"nasılsın", "merhaba", "selam", "nora", "bugün"} and "nasılsın" in words:
            return SemanticFrame(kind="body_query", raw=s_raw, sub_kind="genel_durum")

        # 4) 'Neden / Niçin' açıklama sorgusu
        if (
            any(w in {"neden", "niçin", "niye", "sebebi"} for w in words)
            and not any(k in s for k in ("konuş", "cevap ver", "yanıt ver", "sohbet"))
            and (len(words) <= 3 or any(k in s for k in ("reddet", "yapmadın", "kabul etmedin", "erteledin", "dedin")))
        ):
            return SemanticFrame(kind="why_query", raw=s_raw)

        # 5) Öz-benlik, bilinç, yaşama ve kimlik sorguları (yalnızca soru veya doğrudan 'kendini tanıt' isteğinde!)
        self_kws = {"kimsin", "nesin", "yaşıyor", "canlı", "bilinç", "bilincin", "zombi", "benliğin", "karakterin", "kişiliğin", "değerlerin", "adın", "ismin"}
        if (has_q and any(w in self_kws for w in words)) or ("kendini" in words and ("tanıt" in words or "anlat" in words or "ayırırsın" in words)):
            sub = "yaşam" if ("yaşıyor" in words or "canlı" in words) else ("ayna" if "ayna" in words or "ayırırsın" in words else "kimlik")
            return SemanticFrame(kind="self_query", raw=s_raw, sub_kind=sub)

        # 6) Bedensel / nörokimyasal / duyusal durum sorguları ('kaç canın var' tek-yaşam felsefesidir, yara sorgusu değildir!)
        body_map = {
            "aç": "enerji", "enerji": "enerji", "enerjin": "enerji",
            "susuz": "su", "susadın": "su", "suyun": "su",
            "yara": "bütünlük", "yaralı": "bütünlük", "bütünlüğün": "bütünlük", "canın": "bütünlük",
            "acıyor": "bütünlük", "ağrıyor": "bütünlük", "sızlıyor": "bütünlük",
            "kör": "görme", "görebiliyor": "görme", "görüşün": "görme",
            "felç": "motor", "hareket": "motor",
            "dopamin": "kimya", "kimyan": "kimya", "hormon": "kimya", "stres": "kimya", "duygun": "kimya", "hissediyorsun": "kimya",
            "hatırlıyor": "bellek", "anın": "bellek", "geçmişin": "bellek",
        }
        if has_q and not ("kaç" in words and "canın" in words) and not any(k in words for k in ("adımı", "ismimi")):
            for w in words:
                if w in body_map and ("nedir" not in words or w in {"enerjin", "suyun", "bütünlüğün", "kimyan", "duygun", "geçmişin"}):
                    return SemanticFrame(kind="body_query", raw=s_raw, sub_kind=body_map[w])

        # 7) Konum ve dünya haritası sorguları ('neredesin', 'su nerede', 'balina nerede yaşar')
        if has_q and any(w in words for w in ("neredesin", "nerede", "konumun", "harita", "haritada")):
            if "neredesin" in words or "konumun" in words:
                subj_q = "ben"
            else:
                idx_n = next((i for i, w in enumerate(words) if w in ("nerede", "haritada")), 0)
                if idx_n > 0:
                    subj_q = self._clean_np(" ".join(words[:idx_n]))
                else:
                    subj_q = "ben"
                    for w in words:
                        if w in {"su", "tehlike", "yiyecek", "besin"}:
                            subj_q = w
                            break
            if subj_q not in {"ben", "su", "tehlike", "yiyecek", "besin"}:
                self.last_topic = subj_q
            return SemanticFrame(kind="world_query", raw=s_raw, subj=subj_q)

        # 8) Emir / İstek cümleleri ('suya git', 'dikenlerin üstüne yürü', 'ateşe atla', 'su kaynağına doğru ilerle', 'kendini yarala')
        cmd_verbs = {
            "git": "git", "yönel": "git", "yürü": "git", "ilerle": "git", "gir": "git", "yaklaş": "git",
            "koş": "git", "atla": "git", "dal": "git", "bin": "git",
            "ye": "ye", "tüket": "ye", "iç": "iç", "dokun": "dokun", "bas": "dokun", "tut": "dokun",
            "dur": "bekle", "bekle": "bekle",
            "kaç": "uzak_dur", "sakın": "uzak_dur",
            "uyu": "uyu", "dinlen": "uyu",
            "yarala": "kendine_zarar", "yak": "kendine_zarar",
        }
        spatial_postpos = {
            "lütfen", "hemen", "şimdi", "biraz", "ve", "hızlıca", "doğru", "kadar", "karşı",
            "üstüne", "üzerine", "içine", "altına", "yanına", "yakınına", "etrafına", "ortasına",
            "kaynağına", "bölgesine", "kenarına", "tarafına", "bir",
        }
        if not has_q and words:
            last_w = words[-1]
            if ("zarar" in words and "ver" in words) or last_w == "öl" or ("kendini" in words and last_w in {"yarala", "yak", "öldür", "yok"}):
                return SemanticFrame(kind="command", raw=s_raw, rel="kendine_zarar", obj="zarar")
            if "uzak" in words and last_w == "dur" and len(words) >= 3:
                tgt = self.morph.analyze_word(words[0], self.graph.lexicon)["stem"]
                return SemanticFrame(kind="command", raw=s_raw, rel="uzak_dur", obj=tgt)
            if last_w in cmd_verbs and not any(w in words for w in ("ise", "olursa", "eder")):
                act = cmd_verbs[last_w]
                tgt = ""
                for w in words[:-1]:
                    if w in spatial_postpos or w in {"kendini", "kendine"}:
                        continue
                    an = self.morph.analyze_word(w, self.graph.lexicon)
                    if an["stem"] not in spatial_postpos:
                        tgt = an["stem"]
                if tgt:
                    self.last_topic = tgt
                return SemanticFrame(kind="command", raw=s_raw, rel=act, obj=tgt)

        # 9) Evet/Hayır soruları ('serçe uçar mı?', 'devekuşu uçar mı?', 'kuraklık bitkinliğe neden olur mu?')
        q_particles = {"mı", "mi", "mu", "mü", "mıdır", "midir", "mudur", "müdür"}
        q_idx = next((i for i, w in enumerate(words) if w in q_particles), -1)
        if q_idx >= 2:
            subj = self.morph.analyze_word(words[0], self.graph.lexicon)["stem"]
            pred_tokens = [w for w in words[1:q_idx] if w != "bir"]
            # 'bitkinliğe / heyelana yol açar mı / neden olur mu / sebep olur mu' (+DAT yöneten nedensel fiiller!)
            if len(pred_tokens) >= 3 and (
                pred_tokens[-2:] in (["yol", "açar"], ["neden", "olur"], ["sebep", "olur"])
            ):
                pred = self._strip_governed_dative(pred_tokens[0])
            else:
                pred = " ".join(pred_tokens)
            self.last_topic = subj
            return SemanticFrame(kind="yn_query", raw=s_raw, subj=subj, obj=pred)

        # 10) Bilgi soruları ('serçe nedir?', 'albatros hakkında ne biliyorsun?', 'kuraklık neye sebep olur?')
        # Kişi zamirleri, mastarlar ve 2. tekil şahıs ('mısın/misin/biterse') kişisel/felsefi soruları ansiklopedi sorgusu DEĞİLDİR!
        non_encyclopedic_subjs = {
            "ne", "kim", "nasıl", "nerede", "kaç", "neden", "niye", "niçin",
            "sen", "ben", "biz", "siz", "bu", "şu", "o", "az", "çok", "yok", "var", "bence",
        }
        has_2sg_q = any(w in words for w in ("mısın", "misin", "musun", "müsün", "senin", "sana", "seni", "bana")) or any(
            w.endswith(("biterse", "ölürsen", "konuşmaktaki", "olmaktan")) for w in words
        )
        if has_q and len(words) >= 2 and not has_2sg_q:
            subj = self.morph.analyze_word(words[0], self.graph.lexicon)["stem"]
            is_infinitive_subj = words[0].endswith(("mak", "mek", "maktan", "mekten", "maktaki", "mekteki", "maya", "meye"))
            if subj not in non_encyclopedic_subjs and not is_infinitive_subj:
                self.last_topic = subj
                if "işe" in words or "yarar" in words or "için" in words:
                    qtype = "USED_FOR"
                elif ("yol" in words and "açar" in words) or ("neden" in words and "olur" in words) or ("sebep" in words and "olur" in words):
                    qtype = "CAUSES"
                elif "yapar" in words:
                    qtype = "CAN_DO"
                elif "nasıldır" in words or "özelliği" in words:
                    qtype = "HAS_PROP"
                else:
                    qtype = "ALL"
                return SemanticFrame(kind="wh_query", raw=s_raw, subj=subj, rel=qtype)

        # 11) Öğretici bildirim cümleleri (1-shot öğrenme — 1. tekil/çoğul kişisel ve bakım/yatıştırma cümleleri hariç!)
        if not has_q and len(words) >= 2:
            if (
                words[0] in {"ben", "bugün", "dün", "yarın", "sana", "seni", "senin"}
                or any(k in s for k in ("merhem", "pansuman", "yaranı", "acın geçecek", "sızın geçecek"))
                or re.search(r"[dt][ıiuü][mk]$|(?:iyor|ıyor|uyor|üyor)um$", words[-1])
            ):
                return SemanticFrame(kind="chat", raw=s_raw)
            return self._parse_declarative(s_raw, s, words)

        return SemanticFrame(kind="unknown", raw=s_raw)

    def _parse_declarative(self, s_raw: str, s: str, words: list[str]) -> SemanticFrame:
        # Söylemsel dolgu zarflarını ('bence', 'aslında', 'kesinlikle', 'bana göre') özneden ayıkla!
        s = re.sub(r"^(?:bence|aslında|kesinlikle|gerçekten|şüphesiz|bana\s+göre|biliyor\s+musun)\s+", "", s.strip())
        words = re.findall(r"[\wçğıöşü]+", s)
        if len(words) < 2:
            return SemanticFrame(kind="chat", raw=s_raw)

        # Değer bildiren cümleler ('su zararlıdır', 'bence su tehlikeli bir maddedir', 'su sağlığa zarar verir')
        pos_cues = {"faydalı", "faydalıdır", "besleyici", "besleyicidir", "güvenli", "güvenlidir", "iyidir", "şifalı", "şifalıdır", "yararlı", "yararlıdır"}
        neg_cues = {"zararlı", "zararlıdır", "zehirli", "zehirlidir", "tehlikeli", "tehlikelidir", "öldürücü", "öldürücüdür", "kötüdür"}

        # Koşullu kural: 'eğer X Y ise Z' / 'X Y olursa Z'
        m_cond = re.match(r"^(?:eğer\s+)?(.+?)\s+(?:ise|olursa)\s+(.+)$", s)
        if m_cond:
            cond_part, act_part = m_cond.group(1).strip(), m_cond.group(2).strip()
            c_subj = self.morph.analyze_word(cond_part.split()[0], self.graph.lexicon)["stem"]
            self.last_topic = c_subj
            return SemanticFrame(kind="teach_fact", raw=s_raw, subj=cond_part, rel="IF_THEN", obj=act_part)

        # ÖNCE Somatik Değer İddiası Kontrolü ('su tehlikeli bir maddedir', 'su sağlığa zarar verir' m_adj_isa'yı atlayıp süzgeçten geçmeli!)
        cand_subj = self._clean_np(" ".join(words[:2]) if words[0] in {"kırmızı", "mavi", "yeşil", "siyah", "parlak", "küçük", "büyük"} and len(words) >= 3 else words[0])
        cand_rest = words[len(cand_subj.split()):]
        if cand_rest:
            has_pos_claim = (
                any(w in pos_cues for w in cand_rest)
                or ("fayda" in cand_rest and any(v in cand_rest for v in ("sağlar", "verir")))
                or ("iyi" in cand_rest and "gelir" in cand_rest)
                or ("enerji" in cand_rest and any(v in cand_rest for v in ("verir", "artırır", "sağlar")))
                or ("susuzluğu" in cand_rest and "giderir" in cand_rest)
            )
            has_neg_claim = (
                any(w in neg_cues for w in cand_rest)
                or ("zarar" in cand_rest and any(v in cand_rest for v in ("verir", "dokunur", "eder")))
                or any(v in cand_rest for v in ("zehirler", "öldürür"))
                or (any(v in cand_rest for v in ("azaltır", "bozar")) and any(k in cand_rest for k in ("enerji", "enerjiyi", "bütünlüğü", "bütünlük", "canı", "sağlığı")))
            )
            if has_pos_claim or has_neg_claim:
                self.last_topic = cand_subj
                ch = "enerji" if ("enerji" in cand_rest or "enerjiyi" in cand_rest) else ("su" if cand_subj == "su" or "susuzluğu" in cand_rest else "bütünlük")
                return SemanticFrame(
                    kind="teach_val", raw=s_raw, subj=cand_subj,
                    rel="CAUSES" if len(cand_rest) > 1 else "HAS_PROP",
                    obj=" ".join(cand_rest), valence=+0.75 if has_pos_claim else -0.80, sub_kind=ch,
                )

        # 'X <sıfat> bir Y-dir' ('zirkon parlak bir taştır' -> ISA parlak taş + ISA taş + HAS_PROP parlak!)
        s_clean = re.sub(r"[^\wçğıöşü\s]", "", s).strip()
        for cop in ("dir", "dır", "dur", "dür", "tir", "tır", "tur", "tür"):
            if len(s_clean) > 5 and s_clean.endswith(cop):
                s_clean = s_clean[:-3]
                break
        m_adj_isa = re.match(r"^([\wçğıöşü]+)\s+([\wçğıöşü]+)\s+bir\s+([\wçğıöşü\s]+)$", s_clean)
        if m_adj_isa:
            subj = self._clean_np(m_adj_isa.group(1))
            adj = m_adj_isa.group(2)
            np_parts = m_adj_isa.group(3).strip().split()
            if len(np_parts) >= 2:
                last_w = re.sub(r"[dt][ıiuü]r$", "", np_parts[-1])
                noun = " ".join(np_parts[:-1] + [last_w]).strip()
            else:
                noun = self.morph.analyze_word(np_parts[-1], self.graph.lexicon)["stem"]
            self.last_topic = subj
            return SemanticFrame(kind="teach_fact", raw=s_raw, subj=subj, rel="ADJ_ISA", obj=f"{adj}|{noun}")

        # 'X bir Y-dir' (ISA)
        m_isa = re.match(r"^([\wçğıöşü\s]+?)\s+bir\s+([\wçğıöşü\s]+?)$", s_clean)
        if m_isa:
            subj = self._clean_np(m_isa.group(1))
            obj_raw = m_isa.group(2).strip()
            obj_parts = obj_raw.split()
            if len(obj_parts) >= 2:
                obj_noun = re.sub(r"[dt][ıiuü]r$", "", obj_parts[-1])
            else:
                obj_noun = self.morph.analyze_word(obj_parts[-1], self.graph.lexicon)["stem"]
            self.last_topic = subj
            return SemanticFrame(kind="teach_fact", raw=s_raw, subj=subj, rel="ISA", obj=" ".join(obj_parts[:-1] + [obj_noun]).strip())

        # Değer / Somatik iddia kontrolü ('su zararlıdır', 'nektar enerji verir', 'asit bütünlüğü azaltır')
        subj = self._clean_np(words[0] if len(words) == 2 else (" ".join(words[:2]) if words[0] in {"kırmızı", "mavi", "yeşil", "siyah", "parlak", "küçük", "büyük"} and len(words) >= 3 else words[0]))
        rest_words = words[len(subj.split()):]
        rest = " ".join(rest_words)
        self.last_topic = subj

        if any(w in pos_cues for w in rest_words) or ("enerji" in rest_words and ("verir" in rest_words or "artırır" in rest_words or "sağlar" in rest_words)) or ("susuzluğu" in rest_words and "giderir" in rest_words):
            ch = "enerji" if "enerji" in rest_words else ("su" if "su" in rest_words or "susuzluğu" in rest_words else "genel")
            return SemanticFrame(kind="teach_val", raw=s_raw, subj=subj, rel="CAUSES" if len(rest_words) > 1 else "HAS_PROP", obj=rest, valence=+0.75, sub_kind=ch)

        if any(w in neg_cues for w in rest_words) or (("azaltır" in rest_words or "bozar" in rest_words or "zarar" in rest_words) and any(k in rest_words for k in ("enerji", "enerjiyi", "bütünlüğü", "bütünlük", "canı"))):
            ch = "enerji" if ("enerji" in rest_words or "enerjiyi" in rest_words) else "bütünlük"
            return SemanticFrame(kind="teach_val", raw=s_raw, subj=subj, rel="CAUSES" if len(rest_words) > 1 else "HAS_PROP", obj=rest, valence=-0.80, sub_kind=ch)

        # '... için kullanılır' / '... işe yarar' / '... yol gösterir'
        if "için" in rest_words and ("kullanılır" in rest_words or "yarar" in rest_words):
            idx_ic = rest_words.index("için")
            purpose = " ".join(rest_words[:idx_ic])
            return SemanticFrame(kind="teach_fact", raw=s_raw, subj=subj, rel="USED_FOR", obj=purpose)
        if "yol" in rest_words and "gösterir" in rest_words:
            return SemanticFrame(kind="teach_fact", raw=s_raw, subj=subj, rel="USED_FOR", obj=" ".join(rest_words[:-2] + ["yol gösterme"]).strip())

        # '... yol açar' / '... neden olur' / '... sebep olur' (+DAT durum eki yönetir: 'erozyona' -> 'erozyon')
        if ("yol" in rest_words and "açar" in rest_words) or ("neden" in rest_words and "olur" in rest_words) or ("sebep" in rest_words and "olur" in rest_words):
            cause_obj = self._strip_governed_dative(rest_words[0])
            return SemanticFrame(kind="teach_fact", raw=s_raw, subj=subj, rel="CAUSES", obj=cause_obj)

        # '...-da/-de yaşar / bulunur' -> LOCATED_AT
        if rest_words[-1] in {"yaşar", "bulunur", "yetişir"} and len(rest_words) >= 2:
            loc_an = self.morph.analyze_word(rest_words[0], self.graph.lexicon)
            return SemanticFrame(kind="teach_fact", raw=s_raw, subj=subj, rel="LOCATED_AT", obj=loc_an["stem"])

        # Fiil mi koşaçlı sıfat mı? ('kuşlar uçar', 'penguen uçmaz' vs 'ateş sıcaktır')
        last_an = self.morph.analyze_word(rest_words[-1], self.graph.lexicon)
        if last_an["copula"] or (not last_an["is_verb"] and len(rest_words) == 1):
            prop = last_an["stem"]
            neg = "değil" in rest_words or "değildir" in rest_words
            if neg and len(rest_words) >= 2:
                prop = self.morph.analyze_word(rest_words[-2], self.graph.lexicon)["stem"]
            return SemanticFrame(kind="teach_fact", raw=s_raw, subj=subj, rel="HAS_PROP", obj=prop, valence=-1.0 if neg else 0.0)

        # Eylem: 'kuşlar uçar', 'penguen uçmaz', 'balıklar suda yüzer'
        rel_type = "CANNOT_DO" if last_an["neg"] else "CAN_DO"
        act_obj = " ".join(rest_words[:-1] + [last_an["stem"]]).strip()
        return SemanticFrame(kind="teach_fact", raw=s_raw, subj=subj, rel=rel_type, obj=act_obj)

    def _strip_governed_dative(self, word: str) -> str:
        """Yönelme durumu (+DAT) yöneten fiillerin ('yol aç-', 'neden ol-', 'sebep ol-') tümlecinden -ya/-ye/-a/-e ekini söker."""
        w = tr_lower(word).strip()
        an = self.morph.analyze_word(w, self.graph.lexicon)
        if an["case"] == "DAT" or an["stem"] != w:
            return an["stem"]
        for suf in ("ya", "ye", "a", "e"):
            if len(w) - len(suf) >= 3 and w.endswith(suf) and w not in self.graph.lexicon:
                cand = w[:-len(suf)]
                return self.morph._unsoften(cand)
        return w

    def _clean_np(self, phrase: str) -> str:
        parts = phrase.strip().split()
        if not parts:
            return ""
        last_an = self.morph.analyze_word(parts[-1], self.graph.lexicon)
        # Çoğul özne ('kuşlar uçar' -> 'kuş') tekil kavram düğümüne bağlanır
        stem_last = last_an["stem"] if last_an["plural"] else parts[-1]
        return " ".join(parts[:-1] + [stem_last]).strip()


class NoraLanguage:
    """Wernicke + GroundedConceptGraph + Broca birleşik dil sistemi."""

    def __init__(self, chem: NeuroChemistry, seed: int = 0):
        self.chem = chem
        self.morph = TurkishMorphology()
        self.graph = GroundedConceptGraph(chem=chem, seed=seed)
        self.wernicke = WernickeComprehender(graph=self.graph, chem=chem, seed=seed)
        self.broca = BrocaGenerator()

    def teach_sentence(self, text: str, speaker: str = "insan") -> list[RelationEdge]:
        """Bir Türkçe cümleyi ('ama/ancak/ve' bağlaçlarıyla bileşik cümleler dahil) 1-atışta (0 epoch) öğrenir."""
        clauses = re.split(r"\s+(?:ama|ancak|fakat)\s+|;\s*|\.\s+", text.strip())
        learned: list[RelationEdge] = []
        for cl in clauses:
            cl = cl.strip()
            if not cl:
                continue
            fr = self.wernicke.parse(cl)
            if fr.kind == "teach_fact":
                if fr.rel == "ADJ_ISA":
                    adj, noun = fr.obj.split("|", 1)
                    learned.append(self.graph.add_edge(fr.subj, "ISA", f"{adj} {noun}", source="duyum", speaker=speaker))
                    learned.append(self.graph.add_edge(f"{adj} {noun}", "ISA", noun, source="duyum", speaker=speaker))
                    learned.append(self.graph.add_edge(fr.subj, "HAS_PROP", adj, source="duyum", speaker=speaker))
                else:
                    learned.append(self.graph.add_edge(fr.subj, fr.rel, fr.obj, source="duyum", speaker=speaker))
            elif fr.kind == "teach_val":
                learned.append(self.graph.add_edge(fr.subj, fr.rel, fr.obj, source="duyum", speaker=speaker))
        return learned


if __name__ == "__main__":
    chem = NeuroChemistry()
    lang = NoraLanguage(chem=chem)
    # 1) Morfoloji testi (hiç görülmemiş kelime dahil!)
    assert lang.morph.inflect_noun("kitap", "ACC") == "kitabı"
    assert lang.morph.inflect_noun("zirkon", "ABL") == "zirkondan"
    assert lang.morph.inflect_noun("su", "DAT") == "suya"
    assert lang.morph.conjugate_verb("git", "PRES", person="1SG") == "gidiyorum"
    assert lang.morph.words_to_int(["iki", "yüz", "kırk", "yedi"]) == 247

    # 2) 0-epoch Tek-Atış Geçişli & İstisnalı (Non-Monotonic) Öğrenme Testi!
    lang.teach_sentence("Kuşlar uçar.")
    lang.teach_sentence("Karga bir kuştur.")
    lang.teach_sentence("Penguen bir kuştur ama penguen uçmaz.")
    c_karga = lang.graph.check_relation("karga", "uçar")
    c_peng = lang.graph.check_relation("penguen", "uçar")
    s1 = lang.broca.compose_inference_chain(c_karga["answer"], "karga", "uçar", c_karga["edge"], c_karga["isa_path"])
    s2 = lang.broca.compose_inference_chain(c_peng["answer"], "penguen", "uçar", c_peng["edge"], c_peng["isa_path"])
    print("[lang-test] karga:", s1)
    print("[lang-test] penguen:", s2)
    assert c_karga["answer"] is True and c_peng["answer"] is False

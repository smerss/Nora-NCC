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

from nora._paths import REPO_ROOT


from dataclasses import dataclass, field
import os
import re
import numpy as np

from nora_chem import NeuroChemistry, ChemMushroomBody, HDCMushroomBody
from nora_apply import Interpreter
from nora_forge import OperatorForge
from nora_synth import Synthesizer
from nora_knowledge import (
    PROPER_NOUNS,
    VOWEL_DROP_STEMS,
    REVERSE_VOWEL_DROP,
    NO_SOFTEN_EXTENDED,
    CORE_TURKISH_VERBS,
    ANTONYMS,
    FOUNDATIONAL_PROPOSITIONS,
)


BACK_VOWELS = set("aıou")
FRONT_VOWELS = set("eiöü")
ALL_VOWELS = BACK_VOWELS | FRONT_VOWELS
# Arapça/Farsça alıntılarda â/î/û da ünlüdür: 'zekâ' -> 'zekâyı', 'kâğıt' -> 'kâğıdı'.
SOFT_VOWELS = ALL_VOWELS | set("âîû")
HARD_CONS = set("fstkçşhp")

# -t ile biten fiillerde ünlü önünde t>d yumuşaması SÖZLÜKSELDİR: "hisset->hissediyorum" ama
# "tüket->tüketiyorum", "ilet->iletiyorum". Tek heceliler (git/et/tat/güt) ve -det (kaydet/reddet) zaten kurallı.
SOFTEN_T_VERBS = frozenset({"git", "et", "tat", "güt", "hisset", "affet", "seyret", "zannet", "hükmet", "kaybet"})


def soften_verb_t(v: str) -> str:
    """Ünlü önünde t>d yumuşaması (git->gid, hisset->hissed, kaydet->kayded); sözlükte olmayan fiil
    olduğu gibi kalır (tüket->tüket, ilet->ilet). Kural sözlükseldir, tahmin edilmez."""
    return (v[:-1] + "d") if (v.endswith("det") or v in SOFTEN_T_VERBS) else v



def _stable_hash(x: str) -> int:
    """Süreçler arası kararlı 32-bit FNV-1a (Python hash() PYTHONHASHSEED'e bağlıdır)."""
    h = 2166136261
    for b in x.encode("utf-8"):
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h


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
        "yakala", "tüket", "boz", "kur", "düzelt", "sakla", "kaydet", "ölç", "say", "sırala",
        "dönüştür", "böl", "birleştir", "kır", "aç", "kapat", "doldur", "boşalt",
    } | CORE_TURKISH_VERBS

    # --- ÜRETKEN SÖZCÜK EDİNİMİ -------------------------------------------------
    # Yaşam boyunca duyulan yeni fiil kökleri burada birikir; sınıf düzeyinde olduğu için
    # çözümleyici, üretici ve tüm korteksler aynı sözlüğü anında görür.
    LEARNED_VERB_ROOTS: set[str] = set()
    _VERB_EVIDENCE: dict[str, set[str]] = {}
    # Çekimli fiil olabilecek yüzeyler (ucuz ön süzgeç; asıl kararı SENTEZ verir).
    _VERBISH = re.compile(
        r"(yor|d[ıiuü]|t[ıiuü]|[ae]c[ae][kğ]|m[ıiuü]ş|[ıiuü]r|m[ae]l[ıi]|s[ae])"
        r"(m|n|k|iz|ız|niz|nız|lar|ler|sın|sin|sun|sün|yım|yim|yız|yiz|sınız|siniz)?$"
    )

    @classmethod
    def verb_roots_state(cls) -> list[str]:
        return sorted(cls.LEARNED_VERB_ROOTS)

    @classmethod
    def reset_learned_roots(cls) -> None:
        """Yeni doğan her zihin kendi sözlüğüyle doğar: süreç düzeyi öğrenme sızmaz.
        (Doğuştan gelen çekirdek kökler korunur; öğrenilenler kayıttan geri yüklenir.)"""
        for r in list(cls.LEARNED_VERB_ROOTS):
            cls.KNOWN_VERB_ROOTS.discard(r)
        cls.LEARNED_VERB_ROOTS.clear()
        cls._VERB_EVIDENCE.clear()
        cls._VERB_INDEX.clear()
        cls._VERB_INDEX_ROOTS.clear()
        cls._VERB_INDEX_PENDING.clear()
        cls._ANALYZE_CACHE.clear()

    @classmethod
    def load_verb_roots(cls, roots) -> int:
        n = 0
        for r in list(roots or ()):
            if cls.learn_verb_root(r):
                n += 1
        return n

    @classmethod
    def learn_verb_root(cls, cand: str, lex: set[str] | None = None) -> bool:
        """Yeni fiil kökü edin. Sözlükte AD olarak duran gövde fiil sayılmaz (belirsizlikte susar)."""
        r = tr_lower(str(cand or "")).strip()
        for inf in ("mek", "mak"):
            if len(r) > len(inf) + 1 and r.endswith(inf):
                r = r[:-3]
                break
        if len(r) < 2 or r in cls.KNOWN_VERB_ROOTS:
            return False
        if lex is not None and r in lex:
            return False
        cls.KNOWN_VERB_ROOTS.add(r)
        cls.LEARNED_VERB_ROOTS.add(r)
        if cls._VERB_INDEX_ROOTS:
            cls._VERB_INDEX_PENDING.add(r)      # arka planda indekslenir
        return True

    CONVERB_KINDS = ("erek", "ip", "inca", "ken", "dikce", "madan", "ali", "diginda", "maksizin")

    @classmethod
    def converb_form(cls, root: str, kind: str = "erek", neg: bool = False) -> str:
        """Zarf-fiil (ulaç) yüzeyi. Tek sesbilim kaynağı: çözümleme de bununla sentez yapar."""
        k = str(kind or "").lower()
        v = tr_lower(str(root or "")).strip()
        for inf in ("mek", "mak"):
            if len(v) > len(inf) + 1 and v.endswith(inf):
                v = v[:-3]
                break
        if not v:
            return ""
        # 'de-'/'ye-' ünlüyle başlayan ulaç ekinden önce daralır: diyerek, diyip, diyince, diyeli
        if v in ("de", "ye") and not neg and k in ("erek", "arak", "ip", "ip ", "inca", "ince",
                                                   "-inca", "nca", "ali", "eli", "-alı"):
            v = "di" if v == "de" else "yi"
        v_s = soften_verb_t(v)
        buf = "y" if v_s[-1] in ALL_VOWELS else ""
        v4 = cls.vowel_4(v_s)
        ea = cls.vowel_e_a(v_s)
        body = (v + "m" + cls.vowel_e_a(v)) if neg else v
        _negv4 = "ı" if ea == "a" else "i"          # -mA'dan sonra dar ünlü: okuma-y-ıp
        if k in ("erek", "arak"):
            return f"{v_s}{buf}{ea}r{ea}k" if not neg else f"{body}y{ea}r{ea}k"
        if k in ("ip", "ip "):
            return f"{v_s}{buf}{v4}p" if not neg else f"{body}y{_negv4}p"
        if k in ("inca", "ince", "-inca", "nca"):
            if neg:
                return f"{body}y{'ınca' if ea == 'a' else 'ince'}"      # gelmeyince, okumayınca
            return f"{v_s}{buf}{v4}n{('ca' if v4 in 'ıu' else 'ce')}"
        if k in ("ken", "-ken"):
            # Geniş zaman + ken: 'konuşurken', 'başlarken', 'konuşmazken'
            aor = cls.conjugate_verb(body, "AOR", person="3SG")
            return aor + "ken"
        if k in ("dikce", "dıkça", "dikçe", "-dıkça"):
            b = body
            dt = cls.dental_d_t(b)
            b4 = cls.vowel_4(b)          # okumadıkça (ma->ı), gelmedikçe (me->i)
            return f"{b}{dt}{b4}k{'ça' if b4 in 'ıu' else 'çe'}"
        if k in ("madan", "meden", "-madan"):
            return f"{v}m{ea}d{ea}n"
        if k in ("ali", "eli", "-alı"):
            return f"{v_s}{buf}{'alı' if ea == 'a' else 'eli'}"
        if k in ("diginda", "dığında", "diğinde", "-dığında"):
            b = body
            dt = cls.dental_d_t(b)
            b4 = cls.vowel_4(b)          # okumadığında, gelmediğinde
            return f"{b}{dt}{b4}ğ{b4}nd{'a' if b4 in 'ıu' else 'e'}"
        if k in ("maksizin", "meksizin", "-maksızın"):
            return f"{v}m{ea}k{'sızın' if ea == 'a' else 'sizin'}"
        return ""

    @classmethod
    def _paradigm_forms(cls, root: str) -> dict[str, str]:
        """Bir kökün üretilebilen tüm basit kipleri: yüzey -> kip."""
        out: dict[str, str] = {}
        for t in ("PRES", "PAST", "FUT", "AOR", "NEC", "EVID", "OPT"):
            for p in ("1SG", "2SG", "3SG", "1PL", "2PL", "3PL"):
                try:
                    out.setdefault(cls.conjugate_verb(root, t, person=p), t)
                except Exception:
                    continue
        return out

    @classmethod
    def observe_verb_usage(cls, text: str, lex: set[str] | None = None) -> list[str]:
        """Sözcedeki bilinmeyen fiilleri ÇÖZÜMLEME-SENTEZ ile sınar.

        Yöntem: her olası gövde adayını kendi çekim üreticisiyle çek ve duyulan yüzey biçimini
        birebir üretiyor mu diye bak. Tutuyorsa kök gerçektir (ses olayları dâhil doğrulanmış olur).
        Mastar (-mak/-mek) kesin kanıttır; diğer kiplerde iki ayrı zamanda tanıklık aranır.
        """
        learned: list[str] = []
        for tok in re.findall(r"[\wçğıöşü]+", tr_lower(text or ""))[:24]:
            if len(tok) < 3 or tok in cls.KNOWN_VERB_ROOTS:
                continue
            if lex is not None and tok in lex:
                continue
            # (a) Mastar: tartışmasız fiil kanıtı
            inf_stem = ""
            for inf in ("mak", "mek"):
                if tok.endswith(inf) and len(tok) - 3 >= 2:
                    inf_stem = tok[:-3]
                    break
            if inf_stem and (lex is None or inf_stem not in lex):
                if cls.learn_verb_root(inf_stem, lex):
                    learned.append(inf_stem)
                continue
            if not cls._VERBISH.search(tok):
                continue
            # (b) Çözümleme zaten fiil diyorsa dokunma (bilinen kök)
            try:
                if cls.analyze_word(tok, lex).get("is_verb"):
                    continue
            except Exception:
                pass
            # (c) Sentezle doğrulama
            stem_hit, tense_hit = "", ""
            for cut in range(1, min(6, len(tok) - 2) + 1):
                cand = tok[:-cut]
                for base in (cand, cls._unsoften(cand)):
                    if len(base) < 3 or (lex is not None and base in lex):
                        continue
                    if base in cls.KNOWN_VERB_ROOTS:
                        continue
                    hit = cls._paradigm_forms(base).get(tok, "")
                    if hit:
                        stem_hit, tense_hit = base, hit
                        break
                if stem_hit:
                    break
            if not stem_hit:
                continue
            ev = cls._VERB_EVIDENCE.setdefault(stem_hit, set())
            ev.add(tense_hit)
            # Görülen/şimdiki/gelecek/duyulan geçmiş tek başına yeterince ayırt edicidir;
            # geniş zaman ve kip ekleri iki tanık ister (ad-fiil karışmasın).
            confident = tense_hit in ("PRES", "PAST", "FUT", "EVID")
            if confident or len(ev) >= 2:
                if cls.learn_verb_root(stem_hit, lex):
                    learned.append(stem_hit)
        if learned:
            cls._ANALYZE_CACHE.clear()
        return learned

    # Ettirgen/dönüşlü/işteş çatı ekleri: 'başlat'->'başla', 'çökert'->'çök', 'temizlet'->'temizle'
    VOICE_SUFFIXES = ("tır", "tir", "tur", "tür", "dır", "dir", "dur", "dür", "ır", "ir", "ur", "ür",
                      "ar", "er", "ıl", "il", "ul", "ül", "ın", "in", "un", "ün", "t", "ş", "r")

    @classmethod
    def deep_verb_root(cls, cand: str, max_depth: int = 3) -> str | None:
        """Çatı eklerini yinelemeli olarak soyarak ('çökert'->'çöker'->'çök') bilinen fiil kökünü bulur."""
        if cand in cls.KNOWN_VERB_ROOTS:
            return cand
        frontier = [cand]
        for _ in range(max_depth):
            nxt: list[str] = []
            for c in frontier:
                for sf in cls.VOICE_SUFFIXES:
                    if len(c) - len(sf) >= 2 and c.endswith(sf):
                        base = c[:-len(sf)]
                        if base in cls.KNOWN_VERB_ROOTS:
                            return base
                        hard = cls._unsoften(base)
                        if hard in cls.KNOWN_VERB_ROOTS:
                            return hard
                        if len(base) >= 3:
                            nxt.append(base)
            frontier = nxt
            if not frontier:
                break
        return None

    @classmethod
    def strip_passive_voice(cls, cand: str) -> tuple[str, bool]:
        """Edilgen / dönüşlü Türkçe fiil gövdesinden ('ölçül'->'ölç', 'gözlemlen'->'gözlemle',
        'tanımlan'->'tanımla', 'belirlen'->'belirle', 'korun'->'koru', 'bozul'->'boz') etken kökü bulur."""
        if cand in cls.KNOWN_VERB_ROOTS:
            return cand, False
        # Ünlüyle biten kök + '-n' ('gözlemlen' -> 'gözlemle', 'korun' -> 'koru', 'tanımlan' -> 'tanımla')
        if len(cand) >= 4 and cand.endswith("n") and cand[-2] in ALL_VOWELS:
            base = cand[:-1]
            if base in cls.KNOWN_VERB_ROOTS:
                return base, True
        # Ünsüzle biten kök + '-ıl/-il/-ul/-ül' veya 'l'-bitimli kök + '-ın/-in/-un/-ün'
        for psuf in ("ıl", "il", "ul", "ül", "ın", "in", "un", "ün"):
            if len(cand) - 2 >= 2 and cand.endswith(psuf):
                base = cand[:-2]
                hard_b = cls._unsoften(base)
                if hard_b in cls.KNOWN_VERB_ROOTS:
                    return hard_b, True
                if base in cls.KNOWN_VERB_ROOTS:
                    return base, True
        return cand, False

    _SOFT_MAP = {"â": "a", "î": "i", "û": "u"}

    @staticmethod
    def last_vowel(word: str) -> str:
        for ch in reversed(tr_lower(word)):
            if ch in ALL_VOWELS:
                return ch
            if ch in TurkishMorphology._SOFT_MAP:
                # 'â/î/û' ünlü uyumunda düz karşılığıyla işlem görür: zekâ -> zekâya, kâğıt -> kâğıdı
                return TurkishMorphology._SOFT_MAP[ch]
        return "a"

    @classmethod
    def vowel_e_a(cls, word: str) -> str:
        return "e" if cls.last_vowel(word) in FRONT_VOWELS else "a"

    # İnce ünlüyle çekimlenen alıntı kökler ('dikkat'->'dikkatim', 'saat'->'saatim', 'harf'->'harfim')
    FRONT_HARMONY_EXCEPTIONS: set[str] = {
        "dikkat", "saat", "harf", "kalp", "rol", "gol", "hayal", "ihtimal", "hal", "alkol",
        "kabul", "usul", "amel", "ecel", "sual", "kemal", "meal", "sual", "hilal", "lisan",
        "idrak", "irtibat", "itibar", "ihtisas", "ihtilal", "istiklal", "iktisat", "ithal", "ihal",
    }

    @classmethod
    def vowel_4(cls, word: str) -> str:
        w = tr_lower(word)
        if w in cls.FRONT_HARMONY_EXCEPTIONS:
            return "i"
        v = cls.last_vowel(w)
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
        # Arapça kökenli ikiz ünsüzlü kökler: zıt -> zıddı, his -> hissi, hak -> hakkı, ret -> reddi.
        _GEMINATE = {"zıt": "zıdd", "his": "hiss", "hak": "hakk", "ret": "redd", "tıp": "tıbb",
                     "hac": "hacc", "sır": "sırr", "zam": "zamm", "af": "aff", "us": "uss", "hat": "hatt"}
        if w in _GEMINATE:
            return _GEMINATE[w]
        if w in cls.VOWEL_DROP and cls.VOWEL_DROP[w] != w:
            return cls.VOWEL_DROP[w]
        if w in cls.NO_SOFTEN or (len(w) <= 2 and w not in {"su", "ağ", "uç"}):
            return w
        if w[-1] in ALL_VOWELS:
            return w
        if w.endswith("nk"):
            return w[:-1] + "g"
        # Ünsüz kümesinden sonra gelen -k YUMUŞAMAZ: 'kuark' -> 'kuarkın', 'park' -> 'parkı'
        # ('renk' gibi -nk'liler yukarıdaki kurala girer; bu kural sözlük gerektirmez).
        if w.endswith("k") and len(w) >= 2 and w[-2] not in ALL_VOWELS:
            return w
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
                     poss1: bool = False, copula: bool = False, poss3: bool = False) -> str:
        """İsim çekimi: NOM, ACC (-i), DAT (-e), LOC (-de), ABL (-den), INS (-le), GEN (-in).
        İyelik: poss1 (-Im), poss3 (-(s)I) — kaynaştırma ve ünlü düşmesi kurallarıyla."""
        w = tr_lower(stem).strip()
        if not w:
            return w
        if not plural and not poss1 and not poss3 and w in cls.IRREG_NOUN and case in cls.IRREG_NOUN[w]:
            res = cls.IRREG_NOUN[w][case]
        else:
            # Ünlü düşmesi (boyun -> boynu, ağız -> ağzı): sözlük temelli, tahmin değil.
            if (not plural) and w in cls.REV_VOWEL_DROP and case.upper() != "NOM":
                w = cls.REV_VOWEL_DROP[w]
            if plural:
                w = w + ("ler" if cls.vowel_e_a(w) == "e" else "lar")
            if poss1:
                if w == "su" and not plural:
                    w = "suyum"
                elif w[-1] in SOFT_VOWELS:
                    w = w + "m"
                else:
                    w = cls.soften_stem(w) + cls.vowel_4(w) + "m"
            elif poss3:
                # 3. tekil iyelik: ünlüyle biten köke '-sI', ünsüzle biten köke '-I' ('fonksiyon-u', 'araba-sı')
                if w == "su" and not plural:
                    w = "suyu"
                else:
                    v4 = cls.vowel_4(w)
                    w = (w + "s" + v4) if w[-1] in SOFT_VOWELS else (cls.soften_stem(w) + v4)
                if case.upper() == "NOM":
                    res = w
                    if copula:
                        res = res + cls.dental_d_t(res) + cls.vowel_4(res) + "r"
                    return res
            c = case.upper()
            if poss3 and c != "NOM":
                # 3. tekil iyelikten sonra hal ekleri kaynaştırma 'n' ile gelir ('fonksiyonu-n-u', 'arabası-n-a')
                v4p = cls.vowel_4(w)
                vep = cls.vowel_e_a(w)
                if c == "ACC":
                    res = w + "n" + v4p
                elif c == "DAT":
                    res = w + "n" + vep
                elif c == "LOC":
                    res = w + "n" + cls.dental_d_t(w) + vep
                elif c == "ABL":
                    res = w + "n" + cls.dental_d_t(w) + vep + "n"
                elif c == "GEN":
                    res = w + "n" + v4p + "n"
                elif c == "INS":
                    res = w + "n" + "l" + vep
                else:
                    res = w
                if copula:
                    res = res + cls.dental_d_t(res) + cls.vowel_4(res) + "r"
                return res
            if c == "NOM":
                res = w
            elif c == "ACC":
                if w[-1] in SOFT_VOWELS:
                    res = w + "y" + cls.vowel_4(w)
                else:
                    res = cls.soften_stem(w) + cls.vowel_4(w)
            elif c == "DAT":
                if w[-1] in SOFT_VOWELS:
                    res = w + "y" + cls.vowel_e_a(w)
                else:
                    res = cls.soften_stem(w) + cls.vowel_e_a(w)
            elif c == "LOC":
                res = w + cls.dental_d_t(w) + cls.vowel_e_a(w)
            elif c == "ABL":
                res = w + cls.dental_d_t(w) + cls.vowel_e_a(w) + "n"
            elif c == "INS":
                res = w + ("y" if w[-1] in SOFT_VOWELS else "") + "l" + cls.vowel_e_a(w)
            elif c == "GEN":
                if w[-1] in SOFT_VOWELS:
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
        """Fiil çekimi: PRES (-iyor), PAST (-di), FUT (-ecek), AOR (-ir/-ar), NEC (-meli), IMP.

        Aynı (kök, kip, olumsuzluk, kişi) dörtlüsü sıkça sorulur (çözümleme + üretim + çatı araması);
        sonuç önbelleklenir -> tur gecikmesi O(1) kalır.
        """
        _k = (stem, tense, bool(neg), person)
        _c = cls.__dict__.get("_CONJ_CACHE")
        if _c is None:
            _c = {}
            cls._CONJ_CACHE = _c
        _hit = _c.get(_k)
        if _hit is not None:
            return _hit
        _out = cls._conjugate_verb_uncached(stem, tense, neg, person)
        if len(_c) > 60000:
            _c.clear()
        _c[_k] = _out
        return _out

    @classmethod
    def _conjugate_verb_uncached(cls, stem: str, tense: str = "AOR", neg: bool = False,
                                 person: str = "3SG") -> str:
        v = tr_lower(stem).strip()
        for inf in ("mek", "mak"):
            if len(v) > 3 and v.endswith(inf):
                v = v[:-3]
                break
        if not v:
            return v
        t = tense.upper()
        p = person.upper()
        # Birleşik zamanlar (hikâye/rivayet/şart) tek sesbilim çekirdeğinden türetilir.
        _cmp = globals().get("COMPOUND_MAP") or {}
        if t in _cmp:
            _b, _a = _cmp[t]
            return cls.compound_tense(v, _b, _a, neg=neg, person=p)
        # 'de-' ve 'ye-' ünlü başlayan ekten önce daralır: diyorum, diyecek, yiyeyim (olumsuzda daralmaz: demiyorum)
        if v in ("de", "ye") and not neg and t in ("PRES", "FUT", "OPT"):
            v = "di" if v == "de" else "yi"

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
                v_pres = soften_verb_t(v)
                if v in ("di", "yi"):
                    base = v + "yor"
                elif v_pres[-1] in ALL_VOWELS:
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
            if p == "2PL":
                return base + "sunuz"
            if p in ("3PL", "PL"):
                return base + ("ler" if base[-1] in FRONT_VOWELS else "lar")
            return base

        if t == "PAST":
            b = (v + "m" + cls.vowel_e_a(v)) if neg else v
            dt = cls.dental_d_t(b)
            v4 = cls.vowel_4(b)
            ea_r = cls.vowel_e_a(b)
            if p == "1SG":
                return f"{b}{dt}{v4}m"
            if p == "2SG":
                return f"{b}{dt}{v4}n"
            if p == "1PL":
                return f"{b}{dt}{v4}k"
            if p == "2PL":
                return f"{b}{dt}{v4}n{v4}z"
            if p in ("3PL", "PL"):
                return f"{b}{dt}{v4}l{ea_r}r"
            return f"{b}{dt}{v4}"

        if t == "FUT":
            v_soft = v
            if not neg:
                v_soft = soften_verb_t(v)
            b = (v + "m" + cls.vowel_e_a(v)) if neg else v_soft
            ea = cls.vowel_e_a(b)
            buf = "y" if b[-1] in ALL_VOWELS else ""
            kk = "ğ" if ea == "e" else "ğ"
            if p == "1SG":
                return f"{b}{buf}{ea}c{ea}{kk}{cls.vowel_4(ea)}m"
            if p == "2SG":
                return f"{b}{buf}{ea}c{ea}ks{cls.vowel_4(ea)}n"
            if p == "1PL":
                return f"{b}{buf}{ea}c{ea}{kk}{cls.vowel_4(ea)}z"
            if p == "2PL":
                return f"{b}{buf}{ea}c{ea}ks{cls.vowel_4(ea)}n{cls.vowel_4(ea)}z"
            if p in ("3PL", "PL"):
                return f"{b}{buf}{ea}c{ea}kl{ea}r"
            return f"{b}{buf}{ea}c{ea}k"

        if t in ("EVID", "MIS"):
            # Duyulan geçmiş (-mış/-miş/-muş/-müş): 'saklamışım', 'saklamışsın', 'saklamış'
            b = (v + "m" + cls.vowel_e_a(v)) if neg else v
            v4 = cls.vowel_4(b)
            mis = "m" + v4 + "ş"
            if p == "1SG":
                return f"{b}{mis}{v4}m"
            if p == "2SG":
                return f"{b}{mis}s{v4}n"
            if p == "1PL":
                return f"{b}{mis}{v4}z"
            if p == "2PL":
                return f"{b}{mis}s{v4}n{v4}z"
            if p in ("3PL", "PL"):
                return f"{b}{mis}l{cls.vowel_e_a(b)}r"
            return f"{b}{mis}"

        if t in ("OPT", "ISTEK"):
            # İstek kipi (-a/-e): 'saklayayım', 'saklayasın', 'saklaya'
            v_s = (v + "m" + cls.vowel_e_a(v)) if neg else soften_verb_t(v)
            if v_s in ("de", "ye") and not neg:
                v_s = "di" if v_s == "de" else "yi"
            ea = cls.vowel_e_a(v_s)
            buf = "y" if v_s[-1] in ALL_VOWELS else ""
            v4 = cls.vowel_4(ea)
            if p == "1SG":
                return f"{v_s}{buf}{ea}y{v4}m"
            if p == "2SG":
                return f"{v_s}{buf}{ea}s{v4}n"
            if p == "1PL":
                return f"{v_s}{buf}{ea}l{v4}m"
            if p == "2PL":
                return f"{v_s}{buf}{ea}s{v4}n{v4}z"
            if p in ("3PL", "PL"):
                return f"{v_s}{buf}{ea}l{ea}r"
            return f"{v_s}{buf}{ea}"

        if t in ("COND", "SA"):
            # Şart kipi (-sa/-se): 'saklasam', 'saklasan', 'saklasa'
            b = (v + "m" + cls.vowel_e_a(v)) if neg else v
            ea = cls.vowel_e_a(b)
            if p == "1SG":
                return f"{b}s{ea}m"
            if p == "2SG":
                return f"{b}s{ea}n"
            if p == "1PL":
                return f"{b}s{ea}k"
            if p == "2PL":
                return f"{b}s{ea}n{v4z if False else cls.vowel_4(ea)}z"
            if p in ("3PL", "PL"):
                return f"{b}s{ea}l{ea}r"
            return f"{b}s{ea}"

        if t == "NEC":
            b = (v + "m" + cls.vowel_e_a(v)) if neg else v
            ea = cls.vowel_e_a(b)
            li = "li" if ea == "e" else "lı"
            if p == "1SG":
                return f"{b}m{ea}{li}y{cls.vowel_4(li)}m"
            if p == "2SG":
                return f"{b}m{ea}{li}s{cls.vowel_4(li)}n"
            if p == "1PL":
                return f"{b}m{ea}{li}y{cls.vowel_4(li)}z"
            if p == "2PL":
                return f"{b}m{ea}{li}s{cls.vowel_4(li)}n{cls.vowel_4(li)}z"
            if p in ("3PL", "PL"):
                return f"{b}m{ea}{li}l{ea}r"
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
            if p == "2PL":
                return f"{v}m{ea}zs{cls.vowel_4(ea)}n{cls.vowel_4(ea)}z"
            if p in ("3PL", "PL"):
                return f"{v}m{ea}zl{ea}r"
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
        if p == "2PL":
            return aor + "s" + cls.vowel_4(aor) + "n" + cls.vowel_4(aor) + "z"
        if p in ("3PL", "PL"):
            return aor + ("ler" if cls.vowel_e_a(aor) == "e" else "lar")
        return aor

    _ANALYZE_CACHE: dict[tuple[str, int, int], dict] = {}

    @classmethod
    def _finish_noun(cls, info: dict, cur: str) -> dict:
        """Erken dönüşlerde ortak bitiriş: gövdeyi yaz, alanları tutarlı bırak."""
        if cur in cls.REV_VOWEL_DROP:
            cur = cls.REV_VOWEL_DROP[cur]
            info["vowel_drop"] = True
        info["stem"] = cur
        return info

    _VOCAB_CACHE: dict = {}

    #: TÜRKÇEDE SÖZCÜK SONU ÜNSÜZ ÖBEKLERİ (kapalı liste, fonotaktik).
    #: "tahm|in" gibi soymaları elemek için: "hm" Türkçede hece sonu olamaz, öyleyse
    #: "tahm" diye bir gövde YOKTUR; "tahmin" bütün hâliyle gövdedir.
    _FINAL_PAIRS = frozenset({
        "lp", "lk", "lt", "lç", "lm", "lf", "ls", "lz",
        "nk", "nt", "nç", "nc", "ns", "nz",
        "rd", "rt", "rk", "rç", "rm", "rn", "rl", "rs", "rz", "rf",
        "st", "sk", "şk", "şp", "şt", "sp",
        "kt", "kş", "ft", "ht", "pt", "zd", "zm", "hn",
    })

    @classmethod
    def _plausible_stem(cls, stem: str) -> bool:
        """Bu gövde Türkçe bir sözcük biçiminde mi? (yalnız fonotaktik kapı)"""
        stem = str(stem or "")
        if len(stem) < 2:
            return False
        if not any(ch in ALL_VOWELS for ch in stem):
            return False
        if stem[-1] in ALL_VOWELS:
            return True
        if len(stem) >= 2 and stem[-2] not in ALL_VOWELS and stem[-1] not in ALL_VOWELS:
            return stem[-2:] in cls._FINAL_PAIRS
        return True

    @classmethod
    def _known_vocab(cls, lexicon: set[str] | None) -> set[str]:
        """Sözlük + gündelik ad hazinesi — KOPYASIZ ve önbellekli.

        Her sözcükte 50k'lık sözlüğü kopyalamak O(n) olurdu (O(1) hız sözü bozulur);
        bu yüzden birleşim sözlüğün (id, uzunluk) anahtarıyla bir kez kurulur.
        """
        try:
            from nora_knowledge import COMMON_NOUNS
        except Exception:
            COMMON_NOUNS = set()
        if not lexicon:
            return COMMON_NOUNS
        key = (id(lexicon), len(lexicon))
        hit = cls._VOCAB_CACHE.get(key)
        if hit is not None:
            return hit[1]
        merged = set(lexicon) | COMMON_NOUNS
        if len(cls._VOCAB_CACHE) > 64:
            cls._VOCAB_CACHE.clear()
        # Sözlüğün KENDİSİ de saklanır: id() yeniden kullanılamasın (yanlış eşleşme olmasın).
        cls._VOCAB_CACHE[key] = (lexicon, merged)
        return merged

    @classmethod
    def strip_poss1(cls, w: str, lexicon: set[str] | None = None) -> str | None:
        """1. tekil iyelik ekini (-Im) sözlük doğrulamasıyla soyar: 'kardeşim' -> 'kardeş'.

        Sözcük bütün hâliyle sözlükte duruyorsa bile iyelik okuması geçerlidir; bu, konuşanın
        kendi yakınlarını ('kardeşim', 'annem') aktarırken 2. kişiye çevirmek için gerekir.
        """
        if not w:
            return None
        known = cls._known_vocab(lexicon)
        low = tr_lower(w)
        for suf in ("ımız", "imiz", "umuz", "ümüz", "ım", "im", "um", "üm", "m"):
            if not low.endswith(suf) or len(low) - len(suf) < 3:
                continue
            base = low[: -len(suf)]
            for cand in (base, cls._unsoften(base), cls._undouble(base),
                         base + "a", base + "e", base + "ı", base + "i"):
                if cand and (cand in known or cand in cls.KNOWN_VERB_ROOTS):
                    if cand in known or cls._unsoften(cand) in known:
                        return cand
        return None

    @classmethod
    def analyze_word(cls, raw: str, lexicon: set[str] | None = None) -> dict:
        """Sözlük boyu anahtarlı önbellek: aynı sözcük her turda yeniden çözümlenmez (tur gecikmesi O(1))."""
        key = (raw, len(lexicon) if lexicon else 0, len(cls.KNOWN_VERB_ROOTS))
        hit = cls._ANALYZE_CACHE.get(key)
        if hit is not None:
            return dict(hit)
        info = cls._analyze_word_uncached(raw, lexicon)
        if len(cls._ANALYZE_CACHE) > 40000:
            cls._ANALYZE_CACHE.clear()
        cls._ANALYZE_CACHE[key] = dict(info)
        return info

    @classmethod
    def _analyze_word_uncached(cls, raw: str, lexicon: set[str] | None = None) -> dict:
        """Sözlük-duyarlı çözümleme: ham çözümlemeyi yapar, sonra uydurma kök soymayı geri alır.
        ('zürafa'->'züraf'+DAT, 'matematik'->'mat'+fiil gibi hatalar anlamayı bozuyordu.)"""
        # SÖZLÜK KORUMASI: yüzey biçim sözlükte kayıtlıysa soyulmaz ('bitki'→'bit', 'dünya'→'dün'
        # gibi uydurma kesmeler buradan çıkar). Sözlük yoksa ya da sözcük kayıtlı değilse eski yol.
        surface = re.sub(r"[^\wçğıöşüÇĞİÖŞÜ-]", "", tr_lower(raw))
        if lexicon and len(surface) > 2 and surface in lexicon:
            return {"surface": surface, "stem": surface, "case": "NOM", "plural": False, "neg": False,
                    "copula": False, "is_verb": False}
        info = cls._analyze_word_raw(raw, lexicon)
        known = cls._known_vocab(lexicon)
        # FONOTAKTİK KAPI (sözlükten bağımsız): soyulan gövde Türkçe biçimde değilse
        # soyulmamış yüzey gövdedir. 'tahmin'i 'tahm'+GEN yapmak uydurma bir kök üretir;
        # bu, anlamayı bozar ("Tahm hakkında bilgim yok" gibi cümleler).
        _stem_try = info.get("stem") or ""
        _surf = info.get("surface") or str(raw or "")
        if (_stem_try and _stem_try != _surf and not info.get("is_verb")
                and not cls._plausible_stem(_stem_try) and cls._plausible_stem(_surf)):
            return {**info, "stem": _surf, "case": "NOM"}
        if not known:
            return info
        w, stem = info.get("surface", ""), info.get("stem", "")
        if w in known and (stem not in known or info.get("is_verb")) and not info.get("vowel_drop") and not info.get("poss"):
            # Sözlükte bütün hâliyle duran sözcük, kök soyma tahmininden üstündür:
            # 'yemek' (aş) 'ye+mek' diye fiil köküne indirgenmemeli; 'hikâye' 'hikâ'+DAT olmamalı.
            return {**info, "stem": w, "case": "NOM", "is_verb": False, "neg": False}
        # DOĞRULANMAMIŞ KÖK İŞARETİ: tek harfli bir hâl eki soyyorsa ama ne yüzey ne de
        # gövde sözlükte varsa, gövde bir TAHMİNDİR ('pusula' -> 'pusul'? 'sepet-e' -> 'sepet'?).
        # Çözümleme bozulmaz (DAT/LOC doğru kalır), ama bu kökü ADI GİBİ kullanmayız:
        # söz üretirken duyulan biçime döneriz. Böylece ne 'sepet' kaybolur ne 'pusul' uydurulur.
        if (w not in known and stem not in known and not info.get("is_verb")
                and info.get("case") in ("DAT", "ACC", "GEN", "LOC", "ABL")
                and len(w) - len(stem) == 1 and len(stem) >= 3):
            info["unverified_stem"] = True
        if (w not in known and stem not in known and info.get("is_verb")
                and not info.get("voice") and not info.get("derived_root")
                and stem not in cls.KNOWN_VERB_ROOTS):
            # 'matematik' -> 'mat' (fiil) gibi köksüz çözümlemeler anlamayı bozar; sözlükte
            # olmayan bir fiil kökü asla "gerçek" sayılmaz, kelime olduğu gibi alınır.
            return {**info, "stem": w, "case": "NOM", "is_verb": False, "neg": False}
        return info

    # (ek, zaman, kişi) — uzun ekler önce denenir
    _FINITE_SUFFIXES: tuple[tuple[str, str, str], ...] = (
        ("yelim", "OPT", "1PL"), ("yalım", "OPT", "1PL"), ("elim", "OPT", "1PL"), ("alım", "OPT", "1PL"),
        ("yorsunuz", "PRES", "2PL"), ("yorsun", "PRES", "2SG"), ("yoruz", "PRES", "1PL"),
        ("yorum", "PRES", "1SG"), ("yorlar", "PRES", "3PL"), ("yorsunuz", "PRES", "2PL"),
        ("mışsınız", "EVID", "2PL"), ("mışım", "EVID", "1SG"), ("mışsın", "EVID", "2SG"),
        ("mışız", "EVID", "1PL"), ("mışlar", "EVID", "3PL"),
        ("dınız", "PAST", "2PL"), ("tiniz", "PAST", "2PL"), ("dım", "PAST", "1SG"),
        ("dim", "PAST", "1SG"), ("dum", "PAST", "1SG"), ("düm", "PAST", "1SG"),
        ("tım", "PAST", "1SG"), ("tim", "PAST", "1SG"), ("tum", "PAST", "1SG"), ("tüm", "PAST", "1SG"),
        ("dın", "PAST", "2SG"), ("din", "PAST", "2SG"), ("dun", "PAST", "2SG"), ("dün", "PAST", "2SG"),
        ("tın", "PAST", "2SG"), ("tin", "PAST", "2SG"), ("tun", "PAST", "2SG"), ("tün", "PAST", "2SG"),
        ("dık", "PAST", "1PL"), ("dik", "PAST", "1PL"), ("duk", "PAST", "1PL"), ("dük", "PAST", "1PL"),
        ("tık", "PAST", "1PL"), ("tik", "PAST", "1PL"), ("tuk", "PAST", "1PL"), ("tük", "PAST", "1PL"),
        ("dılar", "PAST", "3PL"), ("diler", "PAST", "3PL"), ("dular", "PAST", "3PL"), ("düler", "PAST", "3PL"),
        ("tılar", "PAST", "3PL"), ("tiler", "PAST", "3PL"),
        ("acaksın", "FUT", "2SG"), ("eceksin", "FUT", "2SG"), ("acağım", "FUT", "1SG"), ("eceğim", "FUT", "1SG"),
        ("acak", "FUT", "3SG"), ("ecek", "FUT", "3SG"),
        ("malıyım", "NEC", "1SG"), ("meliyim", "NEC", "1SG"), ("malı", "NEC", "3SG"), ("meli", "NEC", "3SG"),
        ("dı", "PAST", "3SG"), ("di", "PAST", "3SG"), ("du", "PAST", "3SG"), ("dü", "PAST", "3SG"),
        ("tı", "PAST", "3SG"), ("ti", "PAST", "3SG"), ("tu", "PAST", "3SG"), ("tü", "PAST", "3SG"),
        ("yor", "PRES", "3SG"), ("mış", "EVID", "3SG"), ("miş", "EVID", "3SG"),
        ("muş", "EVID", "3SG"), ("müş", "EVID", "3SG"),
        # GENİŞ ZAMAN (aorist) — kişi ekleriyle birlikte: 'gelirim', 'okursun', 'yazarlar'
        ("ırsınız", "AOR", "2PL"), ("irsınız", "AOR", "2PL"), ("ursunuz", "AOR", "2PL"),
        ("ürsünüz", "AOR", "2PL"), ("arsınız", "AOR", "2PL"), ("ersiniz", "AOR", "2PL"),
        ("ırım", "AOR", "1SG"), ("irim", "AOR", "1SG"), ("urum", "AOR", "1SG"), ("ürüm", "AOR", "1SG"),
        ("arım", "AOR", "1SG"), ("erim", "AOR", "1SG"),
        ("ırsın", "AOR", "2SG"), ("irsin", "AOR", "2SG"), ("ursun", "AOR", "2SG"), ("ürsün", "AOR", "2SG"),
        ("arsın", "AOR", "2SG"), ("ersin", "AOR", "2SG"),
        ("ırız", "AOR", "1PL"), ("iriz", "AOR", "1PL"), ("uruz", "AOR", "1PL"), ("ürüz", "AOR", "1PL"),
        ("arız", "AOR", "1PL"), ("eriz", "AOR", "1PL"),
        ("ırlar", "AOR", "3PL"), ("irler", "AOR", "3PL"), ("urlar", "AOR", "3PL"), ("ürler", "AOR", "3PL"),
        ("arlar", "AOR", "3PL"), ("erler", "AOR", "3PL"),
        ("ır", "AOR", "3SG"), ("ir", "AOR", "3SG"), ("ur", "AOR", "3SG"), ("ür", "AOR", "3SG"),
        ("ar", "AOR", "3SG"), ("er", "AOR", "3SG"),
        ("r", "AOR", "3SG"),
        # ünlü köklerde aorist: 'okurum', 'anlarım' (kök + r + kişi)
        ("rım", "AOR", "1SG"), ("rim", "AOR", "1SG"), ("rum", "AOR", "1SG"), ("rüm", "AOR", "1SG"),
        ("rsın", "AOR", "2SG"), ("rsin", "AOR", "2SG"), ("rsun", "AOR", "2SG"), ("rsün", "AOR", "2SG"),
        ("rız", "AOR", "1PL"), ("riz", "AOR", "1PL"), ("ruz", "AOR", "1PL"), ("rüz", "AOR", "1PL"),
        ("rsınız", "AOR", "2PL"), ("rsiniz", "AOR", "2PL"), ("rsunuz", "AOR", "2PL"), ("rsünüz", "AOR", "2PL"),
        ("rlar", "AOR", "3PL"), ("rler", "AOR", "3PL"),
    )

    _FINITE_BY_LAST: dict[str, tuple] = {}

    # --- ÇEKİM İNDEKSİ ---------------------------------------------------------
    # Üretim tek kaynak: her kök için çekilebilen TÜM yüzeyler bir kez sentezlenir,
    # çözümleme bu indeksten O(1) okur. Böylece üretim ile çözümleme asla ayrışmaz.
    _VERB_INDEX: dict[str, tuple] = {}
    _VERB_INDEX_ROOTS: set = set()
    _VERB_INDEX_PENDING: set = set()
    _INDEX_TENSES: tuple = ("PRES", "PAST", "FUT", "AOR", "NEC", "EVID", "OPT", "COND",
                            "PRES_HIK", "AOR_HIK", "FUT_HIK", "EVID_HIK", "NEC_HIK",
                            "PRES_RIV", "AOR_RIV", "FUT_RIV", "PRES_SART", "AOR_SART",
                            "FUT_SART", "EVID_SART", "NEC_SART")

    @classmethod
    def _index_root(cls, root: str) -> None:
        """Bir kökün çekilebilen TÜM yüzeylerini indekse yazar (fiil, ulaç, sıfat-fiil, emir)."""
        if root in cls._VERB_INDEX_ROOTS or not root or len(root) < 2:
            return
        if root.endswith(("ma", "me")) and len(root) > 3 and (
                root[:-2] in cls.KNOWN_VERB_ROOTS or cls._unsoften(root[:-2]) in cls.KNOWN_VERB_ROOTS):
            cls._VERB_INDEX_ROOTS.add(root)      # 'okuma' bir AD; çekim onun kökünden gelir
            return
        cls._VERB_INDEX_ROOTS.add(root)

        def put(surf: str, entry: tuple) -> None:
            if not surf or surf == root:
                return
            bucket = cls._VERB_INDEX.get(surf)
            if bucket is None:
                cls._VERB_INDEX[surf] = (entry,)
            elif entry not in bucket and len(bucket) < 5:
                cls._VERB_INDEX[surf] = bucket + (entry,)

        # 1) bitmiş fiil çekimleri (basit + birleşik zaman)
        for t in cls._INDEX_TENSES:
            for p in ("1SG", "2SG", "3SG", "1PL", "2PL", "3PL"):
                for ng in (False, True):
                    try:
                        put(cls.conjugate_verb(root, t, neg=ng, person=p), (root, t, p, ng))
                    except Exception:
                        continue
        # 2) ulaçlar
        _conv_map = {"erek": "EREK", "arak": "EREK", "ip": "IP", "inca": "INCA", "ken": "KEN",
                     "dikce": "DIKCA", "madan": "MADAN", "ali": "ALI", "diginda": "DIGINDA",
                     "maksizin": "MAKSIZIN"}
        if hasattr(cls, "converb_form"):
            for k in cls.CONVERB_KINDS:
                for ng in (False, True):
                    try:
                        surf = cls.converb_form(root, k, neg=ng)
                    except Exception:
                        continue
                    put(surf, (root, "CONV:" + _conv_map.get(k, k.upper()), "", ng))
        # 3) sıfat-fiiller / isim-fiiller
        if hasattr(cls, "participle_form"):
            for k in ("an", "dik", "acak", "mis", "maz", "is", "ma"):
                try:
                    surf = cls.participle_form(root, k)
                except Exception:
                    continue
                put(surf, (root, "PART:" + k.upper(), "", False))
        # 4) emir kipi (kişiye göre gövde değişir)
        if hasattr(cls, "imperative_form"):
            for p in ("2SG", "2PL", "3SG", "3PL"):
                for ng in (False, True):
                    try:
                        surf = cls.imperative_form(root, p, neg=ng)
                    except Exception:
                        continue
                    put(surf, (root, "IMP", p, ng))

    @classmethod
    def _ensure_verb_index(cls) -> None:
        if not cls._VERB_INDEX:
            import os
            import pickle
            cache_file = os.path.join(REPO_ROOT, "data", "verb_index_cache.pkl")
            if os.path.isfile(cache_file):
                try:
                    with open(cache_file, "rb") as f:
                        data = pickle.load(f)
                    cls._VERB_INDEX = data.get("index", {})
                    cls._VERB_INDEX_ROOTS = set(data.get("roots", ()))
                except Exception:
                    pass
        for r in tuple(cls.KNOWN_VERB_ROOTS):
            if r not in cls._VERB_INDEX_ROOTS:
                cls._index_root(r)

    @classmethod
    def drain_index_pending(cls) -> int:
        """Yeni öğrenilen kökleri ARKA PLANDA indeksle (konuşma turuna yük binmesin)."""
        n = 0
        for r in tuple(cls._VERB_INDEX_PENDING):
            if r in cls.KNOWN_VERB_ROOTS:
                cls._index_root(r)
                n += 1
            cls._VERB_INDEX_PENDING.discard(r)
        return n

    @classmethod
    def _index_lookup(cls, w: str) -> dict | None:
        cls._ensure_verb_index()
        bucket = cls._VERB_INDEX.get(w)
        if not bucket:
            return None
        root, tag, person, neg = bucket[0]
        if tag.startswith("CONV:"):
            return {"stem": root, "case": "NOM", "is_verb": True,
                    "converb": tag[5:], "neg": neg}
        if tag.startswith("PART:"):
            pk = tag[5:]
            out = {"stem": root, "case": "NOM", "is_verb": True, "participle": pk, "neg": neg}
            if pk in ("MA", "IS"):
                out["verbal_noun"] = pk
            return out
        return {"stem": root, "case": "NOM", "is_verb": True, "tense": tag,
                "person": person, "neg": neg}

    @classmethod
    def _finite_index(cls) -> dict[str, tuple]:
        """Ekleri son harfe göre indeksler: her sözcükte 50 ek denemek yerine ~6 aday denenir (O(1) hız)."""
        if not cls._FINITE_BY_LAST:
            idx: dict[str, list] = {}
            for item in cls._FINITE_SUFFIXES:
                idx.setdefault(item[0][-1], []).append(item)
            cls._FINITE_BY_LAST = {k: tuple(v) for k, v in idx.items()}
        return cls._FINITE_BY_LAST

    @classmethod
    def _finite_verb(cls, w: str) -> dict | None:
        """Bitmiş fiil biçimini (kök + zaman + kişi) çözer; yalnızca sözlükteki kökler kabul edilir."""
        # 1) SENTEZ İNDEKSİ — üretimin aynısı; üretilebilen her biçim çözülebilir.
        _hit = cls._index_lookup(w)
        if _hit is not None:
            # SIFAT KÖKLERİ zamanla fiil gibi öğrenilebiliyor ('güzel-di' -> kök 'güzel').
            # Yalnızca bu durumda koşaç okuması öne geçer (yoksa 'güzel-diğini' diye bozulur);
            # diğer bütün sözcüklerde indeks O(1) yolu korunur.
            try:
                from nora_knowledge import COMMON_ADJECTIVES as _CA, STATE_ADJECTIVES as _SA
                _st = _hit.get("stem")
                if _st in _CA or _st in _SA:
                    _nc0 = getattr(cls, "analyze_nominal_copula", None)
                    _r0 = _nc0(w, None) if _nc0 is not None else None
                    if _r0:
                        return _r0
            except Exception:
                pass
            return _hit
        # AD + KOŞAÇ: 'güzeldi', 'hastaydı', 'soğuktu' (indekste yoksa)
        _nc = getattr(cls, "analyze_nominal_copula", None)
        if _nc is not None:
            _ncr = _nc(w, None)
            if _ncr:
                return _ncr
        # 2) Birleşik zaman (hikâye/rivayet/şart): 'geliyordum', 'gelirmiş', 'gelecektim'
        if hasattr(cls, "analyze_compound_tense") and re.search(
                r"(du|dü|dı|di|tu|tü|tı|ti|muş|müş|mış|miş|sa|se)(m|n|k|niz|nız|nuz|nüz|ler|lar|um|üm|ım|im|sun|sün|sın|sin|uz|üz|ız|iz)?$", w):
            _cmp = cls.analyze_compound_tense(w)
            if _cmp:
                return _cmp
        # Koşaç -DIr ('öğretmendir', 'geliyordur') ve ilgi -ki ('evdeki') — nora_turkce
        for _fn in ("analyze_copula_dir", "analyze_rel_ki", "analyze_ca"):
            _f = getattr(cls, _fn, None)
            if _f is not None:
                _r = _f(w)
                if _r:
                    return _r
        # Yeterlilik (-(y)Abil-), emir kipi ve -mAktA sürekliliği (nora_turkce)
        for _fn in ("analyze_ability", "analyze_imperative", "analyze_progressive",
                    "analyze_verbal_noun"):
            _f = getattr(cls, _fn, None)
            if _f is not None:
                _r = _f(w)
                if _r:
                    return _r
        # ÇATI: 'yapılır', 'açılır', 'edilir', 'sürülür', 'gidilir' — edilgen/ettirgen gövde.
        # Kök sözlükte olduğu için uydurma yok; çatı bilgisi anlamı belirler (nasıl YAPILIR).
        _vv = cls.analyze_voice(w) if hasattr(cls, "analyze_voice") else {}
        if _vv.get("is_verb"):
            return {"stem": _vv.get("stem"), "case": "NOM", "is_verb": True,
                    "tense": _vv.get("tense"), "person": _vv.get("person"),
                    "voice": _vv.get("voice") or "", "voice_stem": _vv.get("voice_stem")}
        # Olumsuz geniş zaman: 'gelmem', 'gelmezsin', 'bitmez', 'atmazlar', 'okumayız'
        _na = re.match(r"^(.+?)m([ae])(z(sınız|siniz|sunuz|sünüz|lar|ler|sın|sin|sun|sün)?|y(ız|iz|uz|üz)|m|n)?$", w)
        if _na:
            _root = _na.group(1)
            _tail = "m" + _na.group(2) + (_na.group(3) or "")
            _psn = {"mam": "1SG", "mem": "1SG", "maz": "3SG", "mez": "3SG",
                    "mayız": "1PL", "meyiz": "1PL",
                    "mazsın": "2SG", "mezsin": "2SG", "mazsınız": "2PL", "mezsiniz": "2PL",
                    "mazlar": "3PL", "mezler": "3PL"}.get(_tail)
            if _psn and len(_root) >= 2:
                for base in (_root, cls._unsoften(_root), _root + "a", _root + "e",
                             _root + "ı", _root + "i", _root + "u", _root + "ü"):
                    if base in cls.KNOWN_VERB_ROOTS:
                        return {"stem": base, "case": "NOM", "is_verb": True, "neg": True,
                                "tense": "AOR", "person": _psn}

        for suf, tense, person in cls._finite_index().get(w[-1:], ()):
            if len(w) - len(suf) < 2 or not w.endswith(suf):
                continue
            cand = w[: -len(suf)]
            cands = [cand, cls._unsoften(cand)]
            if cand.endswith("y") and len(cand) > 2:      # 'okuyacağım' -> 'oku' (kaynaştırma y)
                cands += [cand[:-1], cls._unsoften(cand[:-1])]
            if cand.endswith("n") and len(cand) > 3:      # 'başlayacaksın' -> 'başlayacağ' -> 'başla'
                cands += [cand[:-1]]
            # Ünlüyle biten kök + ek: 'okursun' (ok+ur+sun değil, oku+r+sun), 'anlarım'
            for _v in ("a", "e", "ı", "i", "u", "ü"):
                cands.append(cand + _v)
            # Olumsuz -yor: 'istemiyorum' (iste+me+yor), 'görmüyorum' (gör+me+yor)
            if "yor" in suf and len(cand) >= 3 and cand[-2:-1] == "m" and cand[-1] in "ıiuü":
                for base in (cand[:-2], cls._unsoften(cand[:-2])):
                    if base in cls.KNOWN_VERB_ROOTS:
                        return {"stem": base, "case": "NOM", "is_verb": True, "neg": True,
                                "tense": tense, "person": person}
            # -yor çekiminde kök ile ek arasında dar ünlü türer ya da kökün son ünlüsü düşer:
            # 'anlatıyorum' -> anlat+ı, 'istiyorum' -> iste(->ist+i), 'başlıyorum' -> başla(->başl+ı).
            if "yor" in suf:
                _v, _s = cand, cls._unsoften(cand)
                if cand and cand[-1] in "ıiuü":
                    cands += [cand[:-1], cls._unsoften(cand[:-1])]
                for base in (_v, _s):
                    if base and len(base) >= 3:
                        cands += [base + "e", base + "a", base + "i", base + "ı"]
                if len(_v) >= 3 and _v[-1] in "ıiuü":
                    cands += [_v[:-1] + "e", _v[:-1] + "a"]
                # Olumsuz geniş zaman: 'istemiyorum' -> iste+me+yor; 'görmüyorum' -> gör+me+yor
                if len(_v) >= 3 and _v[-2:-1] == "m" and _v[-1] in "ıiuü":
                    cands += [_v[:-2], cls._unsoften(_v[:-2])]
            _uniq: list[str] = []
            for c_ in cands:
                if c_ and c_ not in _uniq:
                    _uniq.append(c_)
            for root in _uniq:
                if root in cls.KNOWN_VERB_ROOTS:
                    out = {"stem": root, "case": "NOM", "is_verb": True, "tense": tense, "person": person}
                    if suf.startswith(("ma", "me")) and not w.startswith(root + "m"):
                        pass
                    return out
            # olumsuzluk: 'uyuyamadım' -> 'uyu' + 'yama' + 'dım'; 'yapamadım' -> 'yap' + 'ama' + 'dım'
            if len(cand) > 3 and cand[-3:] in ("ama", "eme"):
                for base in (cand[:-3], cand[:-4], cand[:-3][:-1] if cand[:-3].endswith("y") else ""):
                    if not base or len(base) < 2:
                        continue
                    for root in (base, cls._unsoften(base)):
                        if root in cls.KNOWN_VERB_ROOTS:
                            return {"stem": root, "case": "NOM", "is_verb": True, "neg": True,
                                    "tense": tense, "person": person}
        return None

    @classmethod
    def _analyze_word_raw(cls, raw: str, lexicon: set[str] | None = None) -> dict:
        """Kelimeyi (kök, hal, çoğul, olumsuzluk, zaman, koşaç) bileşenlerine ayırır."""
        w = re.sub(r"[^\wçğıöşüÇĞİÖŞÜ-]", "", tr_lower(raw))
        info = {"surface": w, "stem": w, "case": "NOM", "plural": False, "neg": False, "copula": False, "is_verb": False}
        if not w or len(w) <= 2:
            return info
        known = cls._known_vocab(lexicon)

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
        # SIFAT SÖZLÜĞÜ: 'hasta' -> 'has'+LOC gibi savrulmaları önler (sıfatlar çekimsiz köklerdir).
        try:
            from nora_knowledge import COMMON_ADJECTIVES, STATE_ADJECTIVES
            if w in STATE_ADJECTIVES or w in COMMON_ADJECTIVES:
                return info
        except Exception:
            pass

        # 0) BİTMİŞ FİİL (finite verb) çözümlemesi — konuşma dilinin bel kemiği.
        #    'izledim', 'kalktım', 'uyuyamadım', 'geldi', 'konuştum'. Yalnızca kök SÖZLÜKTE varsa kabul
        #    edilir (tahmin yok): yanlış pozitif çıkmaz, gerçek kişi/zaman bilgisi kazanılır.
        fin = cls._finite_verb(w)
        if fin is not None:
            info.update(fin)
            return info

        cur = w
        # 0b) 1. tekil iyelik/koşaç (-Im): 'moralim', 'kafam', 'hastayım', 'öğretmenim'.
        #     Yalnızca SÖZLÜKTE olan bir köke indirgeniyorsa kabul edilir (uydurma kök yok).
        if len(w) >= 4 and w.endswith("m") and w not in known:
            _cands = [w[:-1], w[:-2], w[:-3]]
            for _c in list(_cands):
                if _c.endswith("y"):
                    _cands.append(_c[:-1])
                _cands.append(_c + "a")
                _cands.append(_c + "e")
            for _c in _cands:
                if _c and _c in known:
                    info["stem"] = _c
                    info["poss"] = 1
                    info["case"] = "NOM"
                    return info
                _u = cls._unsoften(_c)
                if _u != _c and _u and _u in known:
                    info["stem"] = _u
                    info["poss"] = 1
                    info["case"] = "NOM"
                    return info

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

        # 1b) 2. tekil kişi koşacı (-sIn): 'sessizsin', 'yorgunsun', 'hastasın' -> sıfat + 2SG.
        for _sf in ("sin", "sın", "sun", "sün"):
            if len(cur) > len(_sf) + 2 and cur.endswith(_sf) and cur[:-len(_sf)] in known:
                info["stem"] = cur[:-len(_sf)]
                info["copula"] = True
                info["person"] = "2SG"
                return info

        # 1c) -cA zarfı ('sence', 'bence', 'onca'): görüş/tarz belirtir, hal eki değildir.
        for _sf in ("ce", "ca", "çe", "ça"):
            if len(cur) > len(_sf) + 1 and cur.endswith(_sf) and cur[:-len(_sf)] in known:
                info["stem"] = cur[:-len(_sf)]
                info["adverb"] = "CA"
                return info

        # 2) Zarf-fiiller (-dıkça/-tikçe, -madan/-meden), yeterlilik (-(y)abilir), şart (-sa/-se, -ırsa/-irse),
        #    duyulan geçmiş (-mış/-miş) ve basit fiil çekim ekleri
        m_dikca = re.match(r"^(.+?)[dt][iıuü]kç[ae]$", cur)
        if m_dikca and len(m_dikca.group(1)) >= 2:
            cand, is_pas = cls.strip_passive_voice(m_dikca.group(1))
            hard_c = cls._unsoften(cand)
            info["stem"] = hard_c if hard_c in cls.KNOWN_VERB_ROOTS else cand
            info["is_verb"] = True
            info["passive"] = is_pas
            info["converb"] = "DIKCA"
            return info

        # 2b) ZARF-FİİL (ulaç) tanıma — SENTEZLE: aynı sesbilim fonksiyonu biçimi yeniden
        #     üretiyorsa kök gerçektir ('konuşurken' -> konuş, 'gittiğinde' -> git, 'koşalı' -> koş).
        m_madan = re.match(r"^(.+?)m[ae]d[ae]n$", cur)
        if m_madan and len(m_madan.group(1)) >= 2:
            cand, is_pas = cls.strip_passive_voice(m_madan.group(1))
            hard_c = cls._unsoften(cand)
            info["stem"] = hard_c if hard_c in cls.KNOWN_VERB_ROOTS else cand
            info["is_verb"] = True
            info["passive"] = is_pas
            info["neg"] = True
            info["converb"] = "MADAN"
            return info

        _conv_ends = ("ken", "ğında", "ğinde", "ğunda", "ğünde", "ğından", "ğinden",
                      "kça", "kçe", "madan", "meden", "alı", "eli", "maksızın", "meksizin",
                      "arak", "erek", "ınca", "ince", "unca", "ünce", "ıp", "ip", "up", "üp")
        if cur.endswith(_conv_ends) and cur not in known:
            matches: list[tuple[bool, int, str, str, bool]] = []
            for cut in range(1, min(10, len(cur) - 2) + 1):
                base = cur[:-cut]
                if len(base) < 2:
                    continue
                hard = cls._unsoften(base)
                known_root = base in cls.KNOWN_VERB_ROOTS or hard in cls.KNOWN_VERB_ROOTS
                for kind in cls.CONVERB_KINDS:
                    for neg in (False, True):
                        if cls.converb_form(base, kind, neg=neg) == cur:
                            matches.append((known_root, not neg, -cut, base, kind, neg))
            if matches:
                # En kısa gövde + bilinen kök yeğlenir: 'konuşurken' -> konuş (konuşu değil)
                matches.sort(reverse=True)
                _kr, _pos, _negcut, base, kind, neg = matches[0]
                hard = cls._unsoften(base)
                info["stem"] = hard if hard in cls.KNOWN_VERB_ROOTS else base
                info["is_verb"] = True
                info["converb"] = kind.upper()
                if neg or kind in ("madan", "maksizin"):
                    info["neg"] = True
                return info

        m_abil = re.match(r"^(.+?)y?[ae]bil(?:ir|iyor|[ae]c[ae]k|m[iıuü]ş|[dt][iıuü])?(?:im|ım|sin|sın|iz|ız|ler|lar)?$", cur)
        if m_abil and len(m_abil.group(1)) >= 2:
            cand, is_pas = cls.strip_passive_voice(m_abil.group(1))
            hard_c = cls._unsoften(cand)
            info["stem"] = hard_c if hard_c in cls.KNOWN_VERB_ROOTS else cand
            info["is_verb"] = True
            info["passive"] = is_pas
            info["modal"] = "ABIL"
            return info

        m_cond_v = re.match(r"^(.+?)(?:(?:[aeiıuü]r|r|[iıuü]yor|m[iıuü]ş)|m[ae]z)?s[ae](?:m|n|k|nız|niz|lar|ler)?$", cur)
        if m_cond_v and len(m_cond_v.group(1)) >= 2 and cur not in {"kısa", "masa", "kasa", "yasa", "tasa", "oysa", "yoksa", "ise", "kimse", "neyse"}:
            raw_c = m_cond_v.group(1)
            cand, is_pas = cls.strip_passive_voice(raw_c)
            hard_c = cls._unsoften(cand)
            if cand in cls.KNOWN_VERB_ROOTS or hard_c in cls.KNOWN_VERB_ROOTS or is_pas or cur.endswith(("ırsa", "irse", "ursa", "ürse", "arsa", "erse", "mezse", "mazsa")):
                info["stem"] = hard_c if hard_c in cls.KNOWN_VERB_ROOTS else cand
                info["is_verb"] = True
                info["passive"] = is_pas
                info["cond"] = True
                info["neg"] = cur.endswith(("mazsa", "mezse", "masa", "mese"))
                return info

        verb_pats = [
            (r"^(.+?)y?[ae](m[ae])z$", True),
            (r"^(.+?)(m[ae])z$", True),
            (r"^(.+?)y?[ae](m[iıuü])yor(um|sun|uz)?$", True),
            (r"^(.+?)(m[iıuü])yor(um|sun|uz)?$", True),
            (r"^(.+?)[iıuü]yor(um|sun|uz)?$", False),
            (r"^(.+?)y?[ae](m[ae])[dt][iıuü](m|n|k)?$", True),
            (r"^(.+?)(m[ae])[dt][iıuü](m|n|k)?$", True),
            (r"^(.+?)(m[ae])m[iıuü]ş(im|ım|um|üm|sin|sın)?$", True),
            (r"^(.+?)m[iıuü]ş(im|ım|um|üm|sin|sın|tır|tir|tur|tür)?$", False),
            (r"^(.+?)(m[ae])k$", False),
            (r"^(.+?)[ae]c[ae](k|ğim|ğım|ksin|ksın)$", False),
        ]
        for pat, is_neg in verb_pats:
            m = re.match(pat, cur)
            if m and len(m.group(1)) >= 2:
                # 'gümüş', 'yemiş', 'geçmiş' gibi sözlük isimlerini koru
                if cur in known and pat.startswith(r"^(.+?)m[iıuü]ş"):
                    continue
                cand, is_pas = cls.strip_passive_voice(m.group(1))
                hard_cand = cls._unsoften(cand)
                info["stem"] = hard_cand if hard_cand in cls.KNOWN_VERB_ROOTS or hard_cand in known else cand
                info["neg"] = is_neg
                info["passive"] = is_pas
                info["is_verb"] = True
                return info

        # Geniş zaman: önce ünlüyle biten fiil kökleri + '-r' ('yürür'->'yürü', 'uyur'->'uyu'), sonra ünsüz + '-ar/-er/-ir' (edilgen dahil!)
        m_aor_v = re.match(r"^(.+[aeıioöuü])r(im|ım|um|üm|sin|sın)?$", cur)
        if m_aor_v and m_aor_v.group(1) in cls.KNOWN_VERB_ROOTS:
            info["stem"] = m_aor_v.group(1)
            info["is_verb"] = True
            return info
        m_aor = re.match(r"^(.+?)([aeiıuü]r)(im|ım|um|üm|sin|sın|sun|sün|iz|ız|uz|üz|ler|lar|sınız|siniz|sunuz|sünüz)?$", cur)
        if m_aor and cur not in known and not cur.endswith(("lar", "ler")):
            raw_cand = m_aor.group(1)
            cand, is_pas = cls.strip_passive_voice(raw_cand)
            hard_cand = cls._unsoften(cand)
            if cand in cls.KNOWN_VERB_ROOTS or hard_cand in cls.KNOWN_VERB_ROOTS or is_pas:
                info["stem"] = hard_cand if hard_cand in cls.KNOWN_VERB_ROOTS else cand
                info["passive"] = is_pas
                info["is_verb"] = True
                info["tense"] = "AOR"
                info["person"] = {"im": "1SG", "ım": "1SG", "um": "1SG", "üm": "1SG",
                                  "sin": "2SG", "sın": "2SG", "sun": "2SG", "sün": "2SG",
                                  "iz": "1PL", "ız": "1PL", "uz": "1PL", "üz": "1PL",
                                  "ler": "3PL", "lar": "3PL", "sınız": "2PL", "siniz": "2PL",
                                  "sunuz": "2PL", "sünüz": "2PL"}.get(m_aor.group(3) or "", "3SG")
                return info
            # Ettirgen/işteş çatılı fiiller ('başlatır', 'çökertir', 'tanışır', 'sentezlenir')
            deep = cls.deep_verb_root(cand)
            if deep is not None:
                info["stem"] = cand          # Yüzeydeki türemiş gövde çekim için korunur!
                info["derived_root"] = deep  # Kavramsal kök ise ayrıca kaydedilir
                info["is_verb"] = True
                return info

        # 3A-0) 3. tekil iyelik YALIN hali: ünlü + '-sı/-si/-su/-sü' ('ifade-si'->'ifade', 'entropi-si'->'entropi', 'daralma-sı'->'daralma')
        if cur not in known and info["case"] == "NOM" and len(cur) - 2 >= 3 and cur.endswith(("sı", "si", "su", "sü")) and cur[-3] in ALL_VOWELS:
            cur = cur[:-2]
            info["poss3"] = True

        # 3A) 3. tekil iyelik + kaynaştırma 'n' + hal ekleri ('fonksiyonunu'->'fonksiyon'+ACC, 'seviyesine'->'seviye'+DAT, 'durumunda'->'durum'+LOC)
        poss3_case_rules = [
            (("sından", "sinden", "sundan", "sünden", "ından", "inden", "undan", "ünden"), "ABL"),
            (("sında", "sinde", "sunda", "sünde", "ında", "inde", "unda", "ünde"), "LOC"),
            (("sına", "sine", "suna", "süne", "ına", "ine", "una", "üne"), "DAT"),
            (("sını", "sini", "sunu", "sünü", "ını", "ini", "unu", "ünü"), "ACC"),
        ]
        if cur not in known and not cur.endswith(("lar", "ler")):
            p3_matched = False
            for p3_sufs, cname in poss3_case_rules:
                for sf in p3_sufs:
                    if len(cur) - len(sf) >= 3 and cur.endswith(sf):
                        cand = cur[:-len(sf)]
                        # 's...' eki ünlüyle biten köke ('seviye-sini'), ünlülü ek ünsüzle biten köke ('fonksiyon-unu') gelir!
                        if sf.startswith("s") and cand[-1] not in ALL_VOWELS:
                            continue
                        if not sf.startswith("s") and cand[-1] in ALL_VOWELS:
                            continue
                        # Kısa gövdede yumuşama tuzağı: 'kadına' -> 'kad'(+iyelik) -> 'kat' savrulması.
                        # Uzun gövdelerde (fonksiyon, seviye, durum) sözlük dışı olmak doğaldır.
                        if len(cand) <= 3 and cand not in known and cls._unsoften(cand) in known:
                            continue
                        # Doğrudan hal eki okuyuşu sözlükte gerçek bir ada çıkıyorsa o okuma üstündür:
                        # 'kadına' = kadın+DAT (kad+iyelik değil).
                        _direct_known = False
                        for _ds in ("a", "e", "ı", "i", "u", "ü", "da", "de", "ta", "te",
                                    "dan", "den", "tan", "ten", "ın", "in", "un", "ün",
                                    "nı", "ni", "nu", "nü", "na", "ne"):
                            if cur.endswith(_ds) and len(cur) - len(_ds) >= 3:
                                _dc = cur[:-len(_ds)]
                                if _dc in known or cls._unsoften(_dc) in known:
                                    _direct_known = True
                                    break
                        if _direct_known:
                            continue
                        if cand in cls.REV_VOWEL_DROP:
                            cur = cls.REV_VOWEL_DROP[cand]
                            info["vowel_drop"] = True
                        else:
                            hard = cls._unsoften(cand)
                            cur = hard if cls._likely_softened(cand, sf[0]) else cand
                        info["case"] = cname
                        info["poss3"] = True
                        p3_matched = True
                        break
                if p3_matched:
                    break

        # 3B) İsim hal ekleri (ABL -> LOC -> GEN -> INS -> DAT -> ACC)
        case_rules = [
            (("dan", "den", "tan", "ten"), "ABL"),
            (("da", "de", "ta", "te"), "LOC"),
            (("nın", "nin", "nun", "nün", "ın", "in", "un", "ün"), "GEN"),
            (("yla", "yle", "la", "le"), "INS"),
            (("ya", "ye", "na", "ne", "a", "e"), "DAT"),
            (("yı", "yi", "yu", "yü", "nı", "ni", "nu", "nü", "ı", "i", "u", "ü"), "ACC"),
        ]
        if info["case"] == "NOM" and not info.get("poss3"):
            for suffixes, cname in case_rules:
                matched = False
                for sf in suffixes:
                    if len(cur) - len(sf) >= 2 and cur.endswith(sf):
                        cand = cur[:-len(sf)]
                        # Türkçe ünsüz benzeşmesi yasası: 't' ile başlayan hal eki (-ta/-te/-tan/-ten) YALNIZCA sert ünsüzden sonra gelebilir!
                        if sf[0] == "t" and (not cand or cand[-1] not in HARD_CONS):
                            continue
                        # Ünlüyle biten gövdeye 'la/le' gelemez: araç hali 'yla/yle' olur
                        # ('okula' -> 'oku'+INS değil, 'okul'+DAT!). Aynı kural tek harfli
                        # belirtme/yönelme ekleri için de geçerli: ünlü gövde 'y'/'n' ister.
                        if cand and cand[-1] in ALL_VOWELS and sf in ("la", "le", "a", "e", "ı", "i", "u", "ü"):
                            continue
                        # Kaynaştırma 'n'li ek yalnızca ünlüyle biten ve SÖZLÜKTE olan bir gövdeye gelir
                        # ('gökyüzü-nü'); yoksa 'kadına' -> 'kadı' gibi savrulma olur.
                        if len(sf) == 2 and sf[0] == "n" and cand not in known \
                                and cls._unsoften(cand) not in known and cand not in cls.REV_VOWEL_DROP:
                            continue
                        # 'a'/'e'/'ı'/'i' tek harfli eklerde kelimenin zaten sözlükte kök olup olmadığını koru;
                        # ancak ünsüz yumuşaması ('ğ/b/c/d' + ünlü) veya sert ünsüz + ünlü ('sepete' -> 'sepet') hal ekidir!
                        hard = cls._unsoften(cand)
                        if hard not in known and cand not in known:
                            _ud = cls._undouble(cand)
                            if _ud in known:
                                hard = cand = _ud
                        softened = cls._likely_softened(cand, sf)
                        vdrop = (cand in cls.REV_VOWEL_DROP)
                        hard_dat_acc = (cand[-1] in HARD_CONS and len(cand) >= 3)
                        if len(sf) == 1 and cur in known and not vdrop and cand not in known and hard not in known:
                            # Yüzey sözlükte, soyalacak kök sözlükte değil: çekim uydurma ('bilgi' -> 'bilg').
                            continue
                        if len(sf) == 1:
                            # Ünlü uyumu + kökte ünlü bulunması koşuluyla tek harfli -ı/-i/-u/-ü ekini yine de kabul et
                            # ('patojeni'->'patojen'+ACC, 'hipotezi'->'hipotez'+ACC) — 'dünya'->'düny' gibi köksüz kalıplar reddedilir!
                            base_vowels = [ch for ch in cand if ch in ALL_VOWELS]
                            _v4 = cls.vowel_4(cand) if base_vowels else ""
                            # Büyük ünlü uyumu: kalın {a,ı,o,u} -> a/ı, ince {e,i,ö,ü} -> e/i;
                            # dörtlü eklerde (ı/i/u/ü) sınıfın kendi harfi gerekir.
                            _ok_harm = (
                                (sf == "a" and _v4 in ("ı", "u")) or (sf == "ı" and _v4 == "ı")
                                or (sf == "e" and _v4 in ("i", "ü")) or (sf == "i" and _v4 == "i")
                                or (sf == "u" and _v4 == "u") or (sf == "ü" and _v4 == "ü")
                            )
                            # Sözlükte DOĞRULANMIŞ iki harflik gerçek kökler ('ev', 'su', 'at') korunur;
                            # sözlükte olmayan kısa gövdeler ise uydurma soyma sayılır.
                            _min_len = 2 if (cand in known or hard in known) else (3 if vdrop else 4)
                            if not (len(cand) >= _min_len and base_vowels and _ok_harm):
                                continue
                        cur = hard if softened else cand
                        info["case"] = cname
                        matched = True
                        break
                if matched:
                    break

        # 1d) İYELİK + HAL ZİNCİRİ: 'kucağıma' (kucak+ım+a), 'kardeşime', 'evime', 'elimde'.
        #     Hal eki tek başına soyulunca geriye sözlükte olmayan bir gövde kalıyorsa, gövdenin sonunda
        #     bir iyelik eki daha vardır; ikisini birlikte çözeriz (Türkçe ad çekiminin gerçek zinciri).
        _case_chain = ("ndan", "nden", "nın", "nin", "nun", "nün", "dan", "den", "tan", "ten",
                       "nda", "nde", "da", "de", "ta", "te", "yla", "yle", "la", "le",
                       "nı", "ni", "nu", "nü", "yı", "yi", "yu", "yü", "ya", "ye",
                       "ın", "in", "un", "ün", "ı", "i", "u", "ü", "a", "e")
        _poss_chain = ("ımız", "imiz", "umuz", "ümüz", "sı", "si", "su", "sü",
                       "ım", "im", "um", "üm", "ın", "in", "un", "ün", "ı", "i", "u", "ü", "m")
        if w not in known:
            for _cs in _case_chain:
                if len(w) - len(_cs) < 3 or not w.endswith(_cs):
                    continue
                _mid = w[:-len(_cs)]
                _case_name = None
                if _cs in ("ndan", "nden"):
                    _case_name = "ABL"
                elif _cs in ("nın", "nin", "nun", "nün"):
                    _case_name = "GEN"
                elif _cs in ("dan", "den", "tan", "ten"):
                    _case_name = "ABL"
                elif _cs in ("nda", "nde", "da", "de", "ta", "te"):
                    _case_name = "LOC"
                elif _cs in ("yla", "yle", "la", "le"):
                    _case_name = "INS"
                elif _cs in ("ya", "ye", "a", "e"):
                    _case_name = "DAT"
                elif _cs in ("nı", "ni", "nu", "nü", "yı", "yi", "yu", "yü", "ı", "i", "u", "ü"):
                    _case_name = "ACC"
                elif _cs in ("ın", "in", "un", "ün"):
                    _case_name = "GEN"
                for _ps in _poss_chain:
                    if len(_mid) - len(_ps) < 2 or not _mid.endswith(_ps):
                        continue
                    _stem_c = _mid[:-len(_ps)]
                    _res = None
                    _pl_hit = False
                    for _pl in ("lar", "ler"):
                        if _stem_c.endswith(_pl) and len(_stem_c) - 3 >= 2:
                            _base = _stem_c[:-3]
                            _b2 = _base if _base in known else cls._unsoften(_base)
                            if _b2 in known:
                                _stem_c = _b2
                                _pl_hit = True
                                break
                    if _stem_c in known:
                        _res = _stem_c
                    elif _stem_c in cls.REV_VOWEL_DROP:
                        _res = cls.REV_VOWEL_DROP[_stem_c]
                        info["vowel_drop"] = True
                    else:
                        _h = cls._unsoften(_stem_c)
                        if _h in known:
                            _res = _h
                        elif cls._undouble(_stem_c) in known:
                            _res = cls._undouble(_stem_c)
                    if _res is None and _ps in ("ımız", "imiz", "umuz", "ümüz") \
                            and len(_stem_c) >= 2 and not _stem_c.endswith(("diğ", "dığ", "duğ", "düğ",
                                                                        "tiğ", "tığ", "tuğ", "tüğ", "y")):
                        # 'evimizde' -> 'ev' (sözlükte 'ev' yokken bile): -ImIz eki
                        # ad üzerinde TEK anlamlıdır, uydurma kök riski yok.
                        _res = _stem_c
                        info["poss_unverified"] = True
                    if _res is not None:
                        if _pl_hit:
                            info["plural"] = True
                        info["stem"] = _res
                        info["poss"] = 1
                        info["poss1"] = True
                        info["case"] = _case_name or "NOM"
                        if len(_ps) > 2:
                            info["poss"] = 2
                            info["poss1"] = False
                            info["poss_person"] = ("1PL" if _ps in ("ımız", "imiz", "umuz", "ümüz")
                                                   else "2PL")
                        return info

        # 3C) YALIN 1. ÇOĞUL İYELİK: 'evimiz' -> 'ev', 'hepimiz' -> 'hep', 'içimiz' -> 'iç'.
        #     -ImIz ad üzerinde tek anlamlıdır; kök sözlükte olmasa da (henüz duyulmamışsa)
        #     iki harflik gövde kabul edilir. -DIk sıfat-fiili ('geldiğimiz') dışarıda tutulur.
        for _p1pl in ("ımız", "imiz", "umuz", "ümüz"):
            if len(w) - len(_p1pl) >= 2 and w.endswith(_p1pl) and w not in known:
                _cand = w[:-len(_p1pl)]
                if _cand.endswith(("diğ", "dığ", "duğ", "düğ", "tiğ", "tığ", "tuğ", "tüğ")):
                    break
                _hard = cls._unsoften(_cand)
                _base = _hard if _hard in known else _cand
                info["stem"] = _base
                info["poss"] = 2
                info["poss_person"] = "1PL"
                info["poss_unverified"] = _base not in known
                return info

        # 4) Çoğul eki (-lar/-ler) ve iyelik (-ım/-im/-um/-üm / -ı/-i/-u/-ü) + Ünlü Düşmesi geri kazanımı
        for pl in ("lar", "ler"):
            if len(cur) > 4 and cur.endswith(pl) and cur not in known:
                cur = cur[:-3]
                info["plural"] = True
                break
        for poss in ("ımız", "imiz", "umuz", "ümüz", "ım", "im", "um", "üm", "ın", "in", "un", "ün", "m"):
            if info.get("poss3"):
                break   # 3. kişi iyeliği çözüldüyse üstüne 1. kişi iyeliği bindirme ('durumunda' -> 'dur')
            if len(cur) - len(poss) >= 3 and cur.endswith(poss) and cur not in known:
                cand = cur[:-len(poss)]
                if cand in cls.REV_VOWEL_DROP:
                    cur = cls.REV_VOWEL_DROP[cand]
                    info["poss1"] = True
                    info["poss"] = 1
                    info["vowel_drop"] = True
                    break
                hard = cls._unsoften(cand)
                if cand in known or hard in known:
                    cur = hard if hard in known else cand
                    info["poss1"] = True
                    info["poss"] = 1
                    info["poss_person"] = "1SG"
                    break
                if poss in ("ımız", "imiz", "umuz", "ümüz") and len(cand) >= 2 \
                        and not cand.endswith(("diğ", "dığ", "duğ", "düğ", "tiğ", "tığ", "tuğ", "tüğ")):
                    # 'evimiz' -> 'ev': 1. çoğul iyelik eki ad üzerinde tek anlamlı
                    cur = cand
                    info["poss"] = 2
                    info["poss_person"] = "1PL"
                    info["poss_unverified"] = True
                    break
                # 'yollarım' -> 'yol' + lar (çoğul) + ım (1. tekil iyelik)
                for _pl in ("lar", "ler"):
                    if cand.endswith(_pl) and len(cand) - len(_pl) >= 2:
                        _b = cand[:-len(_pl)]
                        _hb = cls._unsoften(_b)
                        if _b in known or _hb in known:
                            cur = _hb if _hb in known else _b
                            info["plural"] = True
                            info["poss1"] = True
                            info["poss"] = 1
                            info["poss_person"] = "1SG"
                            return cls._finish_noun(info, cur)

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
    def _undouble(cand: str) -> str:
        """Ünsüz ikizleşmesi geri çözümü: 'zıtt' -> 'zıt', 'hiss' -> 'his', 'ret' korunur."""
        if len(cand) >= 3 and cand[-1] == cand[-2] and cand[-1] in "bcçdfgğhjklmnprsştvyz":
            return cand[:-1]
        return cand

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
    rel: str                  # bilinen ilişki veya öğrenilmiş YENI_OPERATOR_* kimliği
    obj: str
    weight: float = 1.0
    source: str = "duyum"     # deneyim | duyum | doğuştan
    speaker: str = "insan"
    status: str = "accepted" # accepted | provisional | probation | rejected
    verb: str = ""           # icat edilen ilişkinin duyulan fiil kökü
    obj_surface: str = ""    # artikülasyon için öğrenilen nesne yüzeyi
    obj_case: str = "NOM"
    obj_plural: bool = False
    sentence: str = ""       # kanıt/kaynak; cevap şablonu olarak kullanılmaz


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
    verb: str = ""
    obj_surface: str = ""
    obj_case: str = "NOM"
    obj_plural: bool = False


GRAPH_STOP_WORDS = frozenset({
    "en", "bir", "bu", "şu", "o", "ve", "ile", "de", "da", "ki", "için", "gibi",
    "kadar", "daha", "çok", "her", "ne", "hangi", "kim", "nasıl", "nerede", "kaç",
    "ise", "mi", "mı", "mu", "mü", "ama", "fakat", "lakin", "çünkü", "ya", "hem",
})

class GroundedConceptGraph:
    """Bedenlenmiş kavram uzayı + 1-atış ilişkisel sinaps ağı + yayılan aktivasyon çıkarımı."""

    def __init__(self, chem: NeuroChemistry, seed: int = 0):
        self.chem = chem
        self.rng = np.random.default_rng(seed)
        self.lexicon: set[str] = set()
        # Kavram anahtarı küçük harfli kalır; taksonomik/özel adın kullanıcıdan
        # duyulan doğru yüzeyi ayrı tutulur ('aves' anahtarı, 'Aves' görünen ad).
        self.display_names: dict[str, str] = {}
        self.edges: list[RelationEdge] = []
        # Bilme/bilmeme durumu dünya-olgusu değildir. Olay Graph'ta kalıcıdır
        # fakat ``query`` bunu kavram kanıtı sayıp “artık biliyorum”a çevirmez.
        self.epistemic_events: list[RelationEdge] = []
        # O(1) yerel sinaptik komşuluk indeksleri (veri 100.000 kenara çıksa da sorgu gecikmesi ~0.05 ms kalır!)
        self._out_edges: dict[str, list[RelationEdge]] = {}
        self._in_edges: dict[str, list[RelationEdge]] = {}
        self._edge_map: dict[tuple[str, str, str], RelationEdge] = {}
        self.concept_vecs: dict[str, np.ndarray] = {}
        self.concept_id: dict[str, int] = {}
        self.id_concept: list[str] = []
        # Karakter-üçlüsü (trigram) -> seyrek Kenyon hücresi ile fonolojik-kavramsal çağrışım!
        self.word_mb = ChemMushroomBody(n_in=128, n_out=32, n_kc=2048, seed=seed + 11)
        # ANLAM KARARI: önerme adayları HDC/VSA bağlı ~%6 seyrek KC kodundan ve
        # W_fast/W_slow aktivasyonundan çıkar. Graf bu adayları sonradan doğrular.
        self.relation_mb = HDCMushroomBody(dim=256, n_kc=2048, sparsity=0.06, seed=seed + 71)
        self.relation_ledger: dict[str, dict] = {}
        self.relation_by_verb: dict[str, str] = {}
        self.last_decision: dict = {}
        self.last_invention: dict = {}
        self.last_vector_analogy: dict = {}
        self.last_disambiguation: dict = {}
        self.neural_proposer = None
        # Yeni ilişki operatörleri mevcut apply/forge/synth organlarında yaşar;
        # hedef cümlelere özel şablon tablosu değildir.
        self.relation_interpreter = Interpreter()
        self.operator_forge = OperatorForge(interpreter=self.relation_interpreter, ledger_path=None)
        self.relation_synth = Synthesizer()
        self._relation_morph_examples: list[tuple[str, str]] = []
        self._bootstrap_embodied_primitives()

    def clear_all_edges(self):
        self.edges.clear()
        self.epistemic_events.clear()
        self._out_edges.clear()
        self._in_edges.clear()
        self._edge_map.clear()
        self.relation_mb.reset()
        self.relation_ledger.clear()
        self.relation_by_verb.clear()
        self.relation_interpreter.relation_rules.clear()
        self.operator_forge.relation_examples.clear()
        self.relation_synth = Synthesizer()
        self._relation_morph_examples.clear()
        self.last_decision = {}
        self.last_invention = {}

    def record_epistemic_event(self, subj: str, state: str, context: str = "") -> RelationEdge:
        """Bilme durumunu dünya-olgularından ayrı, kalıcı bir Graph olayı yap."""
        s = tr_lower(subj).strip()
        state = tr_lower(state).strip()
        for event in reversed(self.epistemic_events):
            if event.subj == s and event.obj == state and event.status == "epistemic":
                event.weight = float(min(2.0, event.weight + 0.1))
                if context:
                    event.sentence = str(context)[:500]
                return event
        event = RelationEdge(
            subj=s, rel="HAS_STATE", obj=state, weight=1.0,
            source="öz-izleme", speaker="nora", status="epistemic",
            sentence=str(context)[:500],
        )
        self.epistemic_events.append(event)
        return event

    def epistemic_state(self, subj: str) -> RelationEdge | None:
        s = tr_lower(subj).strip()
        return next((event for event in reversed(self.epistemic_events) if event.subj == s), None)

    def _trigram_vec(self, text: str) -> np.ndarray:
        v = np.zeros(128, dtype=np.float32)
        s = f"<{tr_lower(text)}>"
        for i in range(max(1, len(s) - 2)):
            tri = s[i:i + 3]
            h = (_stable_hash(tri) & 0x7FFFFFFF) % 128
            h2 = ((_stable_hash(tri + "!") & 0x7FFFFFFF) % 128)
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

    def register_display_name(self, name: str, surface: str):
        """Kavramın anlam anahtarını değiştirmeden özel ad yüzeyini öğren."""
        c = tr_lower(name).strip()
        surf = str(surface or "").strip().strip(".,;:!?")
        if c and surf and surf != c and surf[:1].isupper():
            self.display_names[c] = surf

    def surface_name(self, name: str) -> str:
        c = tr_lower(name).strip()
        if c in self.display_names:
            return self.display_names[c]
        parts = c.split()
        return " ".join(self.display_names.get(p, p) for p in parts)

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
            self.add_edge(s, r, o, source="doğuştan", weight=0.9, bind_mb=False)
        for s, r, o in FOUNDATIONAL_PROPOSITIONS:
            self.add_edge(s, r, o, source="doğuştan", weight=0.88, bind_mb=False)
        for k_ant, v_ant in ANTONYMS.items():
            self.add_edge(k_ant, "OPPOSITE", v_ant, source="doğuştan", weight=0.85, bind_mb=False)

    @staticmethod
    def _relation_id(verb: str) -> str:
        root = tr_lower(verb).strip()
        surface = re.sub(r"[^\wçğıöşü]+", "_", root.upper(), flags=re.UNICODE).strip("_")
        return f"YENI_OPERATOR_{surface}"

    @staticmethod
    def _aor_root(word: str) -> str:
        """Sözlükte olmayan son sözcükte geniş-zaman ekini sözdizimi için ayır."""
        w = tr_lower(word).strip().strip(".!?…")
        for suffix in ("ar", "er", "ır", "ir", "ur", "ür"):
            if w.endswith(suffix) and len(w) - len(suffix) >= 2:
                return w[:-len(suffix)]
        return ""

    @staticmethod
    def _dynamic_noun(word: str) -> str:
        """Bilinmeyen adda yalnız açık çoğul/belirtme yüzeyini kavram gövdesine indir."""
        w = tr_lower(word).strip().strip(".!?…")
        for suffix in ("ları", "leri"):
            if w.endswith(suffix) and len(w) > len(suffix) + 1:
                return w[:-len(suffix)]
        if w.endswith(("lar", "ler")) and len(w) > 4:
            return w[:-3]
        if len(w) > 3 and w.endswith(("ı", "i", "u", "ü")):
            return w[:-1]
        return w

    def retract_edges(self, predicate) -> int:
        """Koşulu sağlayan kenarları sorgu indekslerinden çıkarır; kayıt silinmez.

        HDC bağları kenarın listedeki yerine (slot) göre tutulduğu için liste kısaltılmaz:
        kenar ağırlığı sıfırlanır, durumu 'retracted' olur ve hiçbir komşuluk onu döndürmez."""
        n = 0
        for edge in self.edges:
            if edge.status == "retracted" or not predicate(edge):
                continue
            edge.weight = 0.0
            edge.status = "retracted"
            for index, key in ((self._out_edges, edge.subj), (self._in_edges, edge.obj)):
                bucket = index.get(key)
                if bucket is not None:
                    bucket[:] = [e for e in bucket if e is not edge]
                    if not bucket:
                        index.pop(key, None)
            if self._edge_map.get((edge.subj, edge.rel, edge.obj)) is edge:
                del self._edge_map[(edge.subj, edge.rel, edge.obj)]
            n += 1
        return n

    def add_edge(self, subj: str, rel: str, obj: str, source: str = "duyum",
                 speaker: str = "insan", weight: float = 1.0, bind_mb: bool = True,
                 status: str = "accepted", verb: str = "", obj_surface: str = "",
                 obj_case: str = "NOM", obj_plural: bool = False,
                 sentence: str = "") -> RelationEdge:
        s = tr_lower(subj).strip()
        o = tr_lower(obj).strip()
        if s in GRAPH_STOP_WORDS or len(s) <= 1:
            return None
        rel = str(rel).strip()
        self.register_concept(s, bind_mb=bind_mb)
        self.register_concept(o, bind_mb=bind_mb)
        existing = self._edge_map.get((s, rel, o))
        if existing is not None:
            existing.weight = float(min(2.0, existing.weight + 0.4 * weight))
            if source == "deneyim":
                existing.source = "deneyim"
                existing.status = "accepted"
            return existing
        # Zıt kanıt eski kenarı ezmez. CAN_DO/CANNOT_DO birlikte kalır; sorguda
        # doğrudan ve daha özgül kanıt üstün gelir. Ağırlık silmek çözüm değildir.

        edge = RelationEdge(
            subj=s, rel=rel, obj=o, weight=float(weight), source=source, speaker=speaker,
            status=str(status), verb=tr_lower(verb).strip(), obj_surface=str(obj_surface),
            obj_case=str(obj_case or "NOM"), obj_plural=bool(obj_plural), sentence=str(sentence),
        )
        self.edges.append(edge)
        slot = len(self.edges) - 1
        self._out_edges.setdefault(s, []).append(edge)
        self._in_edges.setdefault(o, []).append(edge)
        self._edge_map[(s, rel, o)] = edge
        # Eski bilinmezlik yaşantısını silme; öğrenme gerçekleştiyse güncel
        # durumu yeni bir olayla kapat. Epistemik olay HDC sunumu değildir.
        if (status == "accepted" and weight > 0.25
                and any(event.subj == s and event.obj in {"unknown", "insufficient"}
                        for event in self.epistemic_events)):
            self.record_epistemic_event(s, "known", f"{s} {rel} {o}")
        # Tek olay -> HDC/VSA rol bağlama; gradyan ve replay yok.
        if bind_mb:
            self.relation_mb.bind_relation(s, rel, o, slot)
        if source != "doğuştan" and self.chem is not None:
            self.chem.pulse_learning(novelty=0.80, valence=0.35)
        return edge

    def invent_relation(self, source: str, target: str, sentence: str, verb: str,
                        *, obj_surface: str = "", obj_case: str = "NOM",
                        obj_plural: bool = False) -> dict:
        """Bilinmeyen yüklemi ISA'ya çevirmeden probation ilişki olarak icat et.

        Nöral önerici yalnız aday üretir. Yön/çelişki denetimi ve varsa kanıt
        sembolik graftan gelir; hiçbir aday otomatik olarak kabul edilmiş genel
        yasaya dönüşmez. Operatör nesnesi mevcut forge/apply hattına yazılır.
        """
        s, o = tr_lower(source).strip(), tr_lower(target).strip()
        root = tr_lower(verb).strip()
        if not s or not o or not root:
            result = {"status": "invalid", "mode": "none", "edge": None,
                      "relation": "", "candidates": []}
            self.last_invention = result
            return result
        rel = self._relation_id(root)
        reverse_edge = self._edge_map.get((o, rel, s))
        if reverse_edge is not None and reverse_edge.weight > 0.25:
            conflict = {"source": s, "target": o, "sentence": sentence,
                        "against": [reverse_edge.subj, reverse_edge.rel, reverse_edge.obj]}
            rec = self.relation_ledger.setdefault(rel, {
                "relation": rel, "verb": root, "status": "probation",
                "observations": [], "conflicts": [], "candidates": [],
            })
            rec.setdefault("conflicts", []).append(conflict)
            result = {"status": "conflict", "mode": "rejected", "edge": None,
                      "relation": rel, "verb": root, "conflict": conflict,
                      "candidates": []}
            self.last_invention = result
            return result

        # A142 hattı: nöral top-k aday, ardından bağımsız graf kanıtı.
        candidate_rows = []
        proposer = getattr(self, "neural_proposer", None)
        if proposer is not None:
            try:
                proposals = proposer.propose(s, k=5)
            except Exception as exc:
                proposals = []
                candidate_rows.append({"status": "proposer_error", "error": type(exc).__name__})
            allowed = {"ISA", "CAUSES", "USED_FOR", "LOCATED_AT", "YENI_OPERATOR"}
            for proposal in proposals:
                if tr_lower(str(proposal.get("obj", ""))) != o:
                    continue
                proposed_rel = str(proposal.get("rel", ""))
                category = "YENI_OPERATOR" if proposed_rel.startswith("YENI_OPERATOR") else proposed_rel
                if category not in allowed:
                    continue
                proof = self.prove_candidate(s, proposed_rel, o)
                candidate_rows.append({
                    **{k: proposal.get(k) for k in ("subj", "rel", "obj", "score", "neural_pair")},
                    "category": category, "validation": proof.get("status"),
                    "rule": proof.get("rule"), "accepted": proof.get("status") == "proved",
                })

        edge = self.add_edge(
            s, rel, o, source="duyum", weight=0.62, status="probation", verb=root,
            obj_surface=obj_surface or target, obj_case=obj_case,
            obj_plural=obj_plural, sentence=sentence,
        )
        if proposer is not None:
            # Yalnız gözlem belleğine işaretçi eklenir; burada gradyan/epoch
            # yoktur. Yeni operatör sonraki nöral top-k önerilerde de seçilebilir.
            proposer.observe(s, rel, o)
        forged = self.operator_forge.observe_relation(
            source=s, target=o, sentence=sentence, verb=root, relation=rel,
            status="probation",
        )
        self.relation_by_verb[root] = rel
        surface_verb = tr_lower(sentence).strip().rstrip(".!?…").split()[-1]
        pair = (surface_verb, root)
        if pair not in self._relation_morph_examples:
            self._relation_morph_examples.append(pair)
        morphology = {"status": "insufficient", "why": "tek-biçim"}
        if len(self._relation_morph_examples) >= 2:
            morphology = self.relation_synth.learn(list(self._relation_morph_examples))
            morphology = {k: v for k, v in morphology.items() if k != "program"}
        rec = self.relation_ledger.setdefault(rel, {
            "relation": rel, "verb": root, "status": "probation",
            "observations": [], "conflicts": [], "candidates": [],
        })
        observation = {"source": s, "target": o, "sentence": sentence,
                       "status": "probation"}
        if observation not in rec["observations"]:
            rec["observations"].append(observation)
        rec["candidates"] = candidate_rows
        rec["morphology"] = morphology
        rec["operator"] = forged["rule"].rid
        rec["mode"] = forged["mode"]
        result = {
            "status": "probation", "mode": forged["mode"], "edge": edge,
            "relation": rel, "verb": root, "candidates": candidate_rows,
            "operator": forged["rule"].rid, "morphology": morphology,
            "new_binding": forged["new_binding"],
        }
        self.last_invention = result
        return result

    def prove_candidate(self, subj: str, rel: str, obj: str) -> dict:
        """Nöral öneriyi SONRADAN denetle; aday veya dikkat zinciri üretme.

        Doğrudan kenar ``recall`` olup yeni çıkarım sayılmaz. İki dayanaklı
        kanıt yalnız genel ilişki anlamlarından kurulur: ISA ile özellik mirası,
        aynı geçişli ilişkinin kompozisyonu, parçada bulunanın bütünde de
        bulunması ve konumun alt türe özelleşmesi. Bu yöntem çağrılmadan önce
        hangi adayın deneneceğini bilmez; kapanış listesi üretmez.
        """
        q, r, o = (tr_lower(subj).strip(), str(rel).strip(), tr_lower(obj).strip())
        direct = self._edge_map.get((q, r, o))
        if direct is not None and direct.weight > 0.25:
            return {"status": "recall", "novel": False, "proof": [direct], "rule": "direct"}
        # Verilen adayın çok-basamaklı ISA mirasını doğrula. Bu yol aday
        # üretmez; yalnız dışarıdan gelen (q,r,o) üçlüsünün yönünü denetler.
        for ancestor, isa_path in self.ancestors(q):
            if not isa_path:
                continue
            inherited = self._edge_map.get((ancestor, r, o))
            if inherited is not None and inherited.weight > 0.25:
                return {"status": "proved", "novel": True,
                        "proof": list(isa_path) + [inherited], "rule": "isa_inheritance"}
        edges = [edge for edge in self.edges if edge.weight > 0.25]
        transitive = {"ISA", "PART_OF", "CAUSES", "IF_THEN"}
        for first in edges:
            for second in edges:
                if first is second:
                    continue
                rule = ""
                # q ISA a, a r o  =>  q r o (aynı genel miras semantiği query'de de var)
                if (first.subj == q and first.rel == "ISA" and second.subj == first.obj
                        and second.rel == r and second.obj == o):
                    rule = "isa_inheritance"
                # q r a, a r o  =>  q r o; yalnız anlamı geçişli ilişkiler.
                elif (r in transitive and first.subj == q and first.rel == r
                      and second.subj == first.obj and second.rel == r and second.obj == o):
                    rule = "transitive_composition"
                # q LOCATED_AT a, o COMPOSED_OF/HAS_PART a => q LOCATED_AT o
                # q LOCATED_AT a, a PART_OF o             => q LOCATED_AT o
                elif (q == first.subj and r == "LOCATED_AT" and first.rel == "LOCATED_AT"
                      and ((second.subj == o and second.obj == first.obj
                            and second.rel in {"COMPOSED_OF", "HAS_PART"})
                           or (second.subj == first.obj and second.obj == o
                               and second.rel == "PART_OF"))):
                    rule = "location_part_to_whole"
                # q LOCATED_AT a, o ISA a => q LOCATED_AT o (evrendeki alt tür)
                elif (q == first.subj and r == "LOCATED_AT" and first.rel == "LOCATED_AT"
                      and second.subj == o and second.rel == "ISA"
                      and second.obj == first.obj):
                    rule = "location_isa_specialization"
                if rule:
                    return {"status": "proved", "novel": True,
                            "proof": [first, second], "rule": rule}
        return {"status": "unresolved", "novel": False, "proof": [], "rule": ""}

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
        """MB aktivasyonuyla aday getir; graf ile kenar/yön kanıtını doğrula.

        ``ancestors`` burada aday üretmek için kullanılmaz. MB'den gelen ISA
        önermesi doğrulandığında bir üst kavrama yayılınır. Böylece sembolik
        kapanış listesini okuyup ona "nöral karar" demek mümkün değildir.
        """
        start = tr_lower(subj).strip()
        if start in GRAPH_STOP_WORDS or len(start) <= 1:
            return []
        frontier: list[tuple[str, list[RelationEdge]]] = [(start, [])]
        visited: set[str] = set()
        blocked_actions: set[str] = set()
        results: list[tuple[RelationEdge, list[RelationEdge]]] = []
        reads = []
        validated = 0
        rejected = 0
        for _hop in range(5):
            next_frontier: list[tuple[str, list[RelationEdge]]] = []
            for current, isa_path in frontier:
                if current in visited:
                    continue
                visited.add(current)
                recalled = self.relation_mb.recall_subject(current)
                read = dict(self.relation_mb.last_recall)
                read["concept"] = current
                reads.append(read)
                if not recalled and current in self._out_edges:
                    for edge in self._out_edges[current]:
                        if edge.weight <= 0.25 or self._edge_map.get((edge.subj, edge.rel, edge.obj)) is not edge:
                            continue
                        validated += 1
                        edge._last_mb_activation = 1.0
                        if edge.rel == "CANNOT_DO":
                            blocked_actions.add(edge.obj)
                        if edge.rel == "ISA" and edge.obj not in visited:
                            next_frontier.append((edge.obj, list(isa_path) + [edge]))
                        if rels is not None and edge.rel not in rels:
                            continue
                        if edge.rel == "CAN_DO" and edge.obj in blocked_actions and isa_path:
                            continue
                        results.append((edge, list(isa_path)))
                else:
                    for slot, activation in recalled:
                        if not 0 <= slot < len(self.edges):
                            rejected += 1
                            continue
                        edge = self.edges[slot]
                        # Bağımsız sembolik doğrulama: MB aktivasyonu tek başına
                        # dünya bilgisi değildir; indeks, özne ve canlı kenar tutmalı.
                        if (edge.subj != current or edge.weight <= 0.25
                                or self._edge_map.get((edge.subj, edge.rel, edge.obj)) is not edge):
                            rejected += 1
                            continue
                        validated += 1
                        edge._last_mb_activation = float(activation)
                        if edge.rel == "CANNOT_DO":
                            blocked_actions.add(edge.obj)
                        if edge.rel == "ISA" and edge.obj not in visited:
                            next_frontier.append((edge.obj, list(isa_path) + [edge]))
                        if rels is not None and edge.rel not in rels:
                            continue
                        if edge.rel == "CAN_DO" and edge.obj in blocked_actions and isa_path:
                            continue
                        results.append((edge, list(isa_path)))
            frontier = next_frontier
            if not frontier:
                break
        self.last_decision = {
            "engine": "hdc_mb", "awake": True, "replay": 0,
            "graph_validated": True, "query": start,
            "validated_candidates": validated, "rejected_candidates": rejected,
            "results": len(results), "reads": reads,
            "active_kc": reads[-1].get("active_kc", 0) if reads else 0,
            "actual_sparsity": reads[-1].get("actual_sparsity", 0.0) if reads else 0.0,
            "target_sparsity": 0.06,
            "working_set_size": reads[-1].get("working_set_size", 0) if reads else 0,
            "active_slots": reads[-1].get("active_slots", 0) if reads else 0,
            "passive_slots": reads[-1].get("passive_slots", 0) if reads else 0,
            "passive_reactivations": reads[-1].get("passive_reactivations", 0) if reads else 0,
            "scanned_slots": reads[-1].get("scanned_slots", 0) if reads else 0,
        }
        return results

    def relation_object_candidates(self, relation: str, obj: str) -> list[tuple[RelationEdge, float]]:
        """HDC ilişki+nesne okumasını canlı graf kenarlarıyla doğrula."""
        rel = str(relation).strip()
        obj = tr_lower(obj).strip()
        recalled = self.relation_mb.recall_relation_object(rel, obj)
        read = dict(self.relation_mb.last_recall)
        validated: list[tuple[RelationEdge, float]] = []
        rejected = 0
        for slot, activation in recalled:
            if not 0 <= slot < len(self.edges):
                rejected += 1
                continue
            edge = self.edges[slot]
            if (edge.rel != rel or edge.obj != obj or edge.weight <= 0.25
                    or self._edge_map.get((edge.subj, edge.rel, edge.obj)) is not edge):
                rejected += 1
                continue
            edge._last_mb_activation = float(activation)
            validated.append((edge, float(activation)))
        self.last_decision = {
            "engine": "hdc_mb", "mode": "relation_object", "awake": True,
            "replay": 0, "graph_validated": True, "query": f"{rel}:{obj}",
            "validated_candidates": len(validated), "rejected_candidates": rejected,
            "results": len(validated), "reads": [read],
            "working_set_size": read.get("working_set_size", 0),
            "active_slots": read.get("active_slots", 0),
            "passive_slots": read.get("passive_slots", 0),
            "passive_reactivations": read.get("passive_reactivations", 0),
        }
        return validated

    def find_by_requirements(self, requirements: list[tuple[str, str]], *,
                             head: str = "") -> dict:
        """Birleşik göreli cümleyi HDC aday kümelerinin kesişimiyle çöz.

        Örneğin HAS_PART(kanat) ve CAN_DO(uç) iki ayrı tek-sunum olayından
        gelir. Hiçbir sonuç önceden ``kuş`` diye yazılmaz; her koşulun adayları
        MB'den gelir ve kesişimdeki her kenar yeniden doğrulanır.
        """
        normalized = [(str(rel).strip(), tr_lower(obj).strip()) for rel, obj in requirements
                      if str(rel).strip() and tr_lower(obj).strip()]
        if not normalized:
            return {"status": "invalid", "candidates": [], "requirements": []}
        per_requirement: list[dict[str, list[tuple[RelationEdge, float]]]] = []
        for rel, obj in normalized:
            by_subject: dict[str, list[tuple[RelationEdge, float]]] = {}
            for edge, activation in self.relation_object_candidates(rel, obj):
                by_subject.setdefault(edge.subj, []).append((edge, activation))
            per_requirement.append(by_subject)
        common = set(per_requirement[0])
        for rows in per_requirement[1:]:
            common.intersection_update(rows)
        head = tr_lower(head).strip()
        candidates = []
        for subject in sorted(common):
            # Pozitif koşul, daha özgül olumsuz kanıtın üstüne çıkarılamaz.
            blocked = False
            for rel, obj in normalized:
                if rel != "CAN_DO":
                    continue
                if any(edge.obj == obj for edge, _path in self.query(subject, ("CANNOT_DO",))):
                    blocked = True
                    break
            if blocked:
                continue
            if head and head not in {"şey", "kavram"}:
                if not any(edge.obj == head for edge, _path in self.query(subject, ("ISA",))):
                    continue
            proof = []
            activations = []
            for (rel, obj), rows in zip(normalized, per_requirement):
                edge, activation = max(rows[subject], key=lambda row: row[1])
                proof.append(edge)
                activations.append(float(activation))
            candidates.append({"subject": subject, "proof": proof,
                               "score": min(activations) if activations else 0.0})
        candidates.sort(key=lambda row: (-row["score"], row["subject"]))
        return {
            "status": "ok" if candidates else "unresolved",
            "engine": "hdc_mb_relation_object_conjunction", "graph_validated": True,
            "requirements": normalized, "head": head, "candidates": candidates,
            "presentations": int(self.relation_mb.presentations),
        }

    def compare_concepts(self, left: str, right: str) -> dict:
        """İki kavramın HDC önerili, graf-doğrulamalı anlam kesişimini çıkar."""
        a, b = tr_lower(left).strip(), tr_lower(right).strip()
        left_rows = self.query(a, None) if a else []
        right_rows = self.query(b, None) if b else []
        missing = [name for name, rows in ((a, left_rows), (b, right_rows)) if name and not rows]

        def features(rows):
            out = {}
            for edge, path in rows:
                key = (edge.rel, edge.obj)
                old = out.get(key)
                if old is None or (len(path), -edge.weight) < (len(old[1]), -old[0].weight):
                    out[key] = (edge, path)
            return out

        left_features, right_features = features(left_rows), features(right_rows)
        shared = []
        hdc_proposals = 0
        # Aday özellik HDC'den geri gelmezse sembolik küme kesişimi cevabı
        # tek başına üretemez. Bu, karşılaştırmada hesaplamanın nedensel izidir.
        for key in dict.fromkeys([*left_features, *right_features]):
            recalled = self.relation_object_candidates(*key)
            if not recalled:
                continue
            hdc_proposals += 1
            if key not in left_features or key not in right_features:
                continue
            left_edge, left_path = left_features[key]
            right_edge, right_path = right_features[key]
            activations = {id(edge): float(activation) for edge, activation in recalled}
            # Doğrudan kenarlar farklıysa iki tarafın da HDC okuması gerekir;
            # tek taraflı aktivasyon + sembolik kesişim ortaklık sayılmaz.
            # Miras alınan aynı kenar ise iki yol için de aynı kanıtı taşır.
            if id(left_edge) not in activations or id(right_edge) not in activations:
                continue
            shared.append({
                "rel": key[0], "obj": key[1],
                "left": (left_edge, left_path), "right": (right_edge, right_path),
                "activation": min(activations[id(left_edge)], activations[id(right_edge)]),
            })
        common_keys = {(row["rel"], row["obj"]) for row in shared}
        return {
            "status": "missing" if missing else ("ok" if shared else "unresolved"),
            "engine": "hdc_mb_relation_object_intersection", "graph_validated": True,
            "left": a, "right": b, "missing": missing, "shared": shared,
            "left_only": sorted(set(left_features) - common_keys),
            "right_only": sorted(set(right_features) - common_keys),
            "hdc_proposals": hdc_proposals,
            "presentations": int(self.relation_mb.presentations),
        }

    def causal_chain(self, subj: str, max_hops: int = 4) -> list[tuple[str, list[RelationEdge]]]:
        """MB'den doğrulanan CAUSES/IF_THEN kenarlarıyla ileri yayılım.

        Doğrudan graf komşuluğu burada da cevap kaynağı değildir; her düğümün
        giden nedensel adayı ``query`` içindeki HDC/MB okumasından gelir.
        """
        s = tr_lower(subj).strip()
        visited = {s}
        frontier: list[tuple[str, list[RelationEdge]]] = [(s, [])]
        out: list[tuple[str, list[RelationEdge]]] = []
        for _ in range(max_hops):
            nxt = []
            for cur, path in frontier:
                for e, _isa_path in self.query(cur, ("CAUSES", "IF_THEN")):
                    if e.obj in visited:
                        continue
                    visited.add(e.obj)
                    npath = path + [e]
                    out.append((e.obj, npath))
                    nxt.append((e.obj, npath))
            frontier = nxt
            if not frontier:
                break
        return out

    def do(self, node: str, value: str | None = None, max_hops: int = 3,
           _disabled: set[tuple[str, str, str]] | None = None) -> dict:
        """Yapısal ``do(X)``: girenleri kes, yalnız HDC-doğrulamalı çıkanları izle.

        Graf indeksi tek başına cevap kaynağı değildir. Her halkada adaylar MB'nin
        bağlı önerme kodundan gelir; ``query`` canlı kenar/yön kanıtını ayrıca
        doğrular. Böylece MB ablasyonunda etiketleri gezip aynı cevabı üretmek
        mümkün olmaz. ``_disabled`` yalnız karşıolgusal modelin kapattığı yapısal
        kenarlardır; bellekten bilgi silinmez.
        """
        s = tr_lower(node).strip()
        disabled = set(_disabled or ())
        incoming = list(self._in_edges.get(s, ()))
        backdoor = [e for e in incoming
                    if e.rel in ("CAUSES", "IF_THEN") and e.weight > 0.25]
        effects: list[str] = []
        paths: list[list[RelationEdge]] = []
        visited = {s}
        frontier: list[tuple[str, list[RelationEdge]]] = [(s, [])]
        reads: list[dict] = []
        validated = 0
        for _ in range(max(1, int(max_hops))):
            nxt: list[tuple[str, list[RelationEdge]]] = []
            for cur, path in frontier:
                # ISA üstünden miras alınan nedensellik burada yapısal bir X→Y
                # kenarı değildir; yalnız cur'nin kendi giden kenarı izlenir.
                candidates = self.query(cur, ("CAUSES", "IF_THEN"))
                decision = dict(self.last_decision or {})
                reads.extend(decision.get("reads") or [])
                for e, isa_path in candidates:
                    key = (e.subj, e.rel, e.obj)
                    if isa_path or e.subj != cur or key in disabled:
                        continue
                    validated += 1
                    if e.obj in visited:
                        continue
                    visited.add(e.obj)
                    npath = path + [e]
                    paths.append(npath)
                    effects.append(e.obj)
                    nxt.append((e.obj, npath))
            frontier = nxt
            if not frontier:
                break
        result = {
            "intervened": s,
            "value": value,
            "effects": effects,
            "paths": paths,
            "confounders": sorted({e.subj for e in backdoor}),
            "identified": not backdoor,
            "observed_edges": [e.obj for e in self._out_edges.get(s, ())],
            "in_edges": [(e.subj, e.rel) for e in incoming],
            "cut_in_edges": [(e.subj, e.rel, e.obj) for e in backdoor],
            "disabled_edges": sorted(disabled),
            "engine": "hdc_mb+graph",
            "hdc_required": True,
        }
        self.last_decision = {
            "engine": "hdc_mb+graph", "mode": "do", "query": s,
            "graph_validated": True, "validated_candidates": validated,
            "results": len(effects), "reads": reads,
            "cut_in_edges": len(backdoor), "disabled_edges": len(disabled),
            "active_kc": reads[-1].get("active_kc", 0) if reads else 0,
            "actual_sparsity": reads[-1].get("actual_sparsity", 0.0) if reads else 0.0,
        }
        return result

    def counterfactual(self, node: str, max_hops: int = 3) -> dict:
        """X olmasaydı: X'in çıkan yapısal kenarlarını silmeden devre dışı bırak.

        ``do`` müdahalesi X'e girenleri kesip çıkanları izler; karşıolgusal
        karşılaştırma ise aynı HDC/Graf modelinde X'in çıkanlarını kapatır.
        Geçici indeks ameliyatı yapılmadığı için pasif MB kayıtlarıyla canlı graf
        arasında tutarsızlık ve bilgi kaybı oluşmaz.
        """
        s = tr_lower(node).strip()
        base = self.do(s, max_hops=max_hops)
        actual = list(base.get("effects") or [])
        disabled = {(e.subj, e.rel, e.obj) for e in self._out_edges.get(s, ())
                    if e.rel in ("CAUSES", "IF_THEN") and e.weight > 0.25}
        # X'siz modelde alternatif köklerden erişimi tara. Yalnız X'ten başlamak
        # “B→Y varken de olurdu” yolunu yanlışlıkla kayıp sayardı. Ham indeks
        # yalnız kök adayı üretir; her nedensel halka yine HDC/MB + Graph ile
        # doğrulanır ve X'e giren/yeniden X'e çıkan yol kullanılmaz.
        causal_nodes = {e.subj for e in self.edges
                        if e.rel in ("CAUSES", "IF_THEN") and e.weight > 0.25}
        causal_nodes.update(e.obj for e in self.edges
                            if e.rel in ("CAUSES", "IF_THEN") and e.weight > 0.25)
        roots = sorted(node for node in causal_nodes if node != s and not any(
            e.rel in ("CAUSES", "IF_THEN") and e.weight > 0.25
            for e in self._in_edges.get(node, ())))
        reachable: set[str] = set()
        visited = set(roots)
        frontier = list(roots)
        for _ in range(max(1, int(max_hops))):
            nxt = []
            for cur in frontier:
                for edge, isa_path in self.query(cur, ("CAUSES", "IF_THEN")):
                    key = (edge.subj, edge.rel, edge.obj)
                    if (isa_path or edge.subj != cur or key in disabled
                            or edge.obj == s or edge.obj in visited):
                        continue
                    visited.add(edge.obj)
                    reachable.add(edge.obj)
                    nxt.append(edge.obj)
            frontier = nxt
            if not frontier:
                break
        # Rapor kapsamı X'in gerçek etkileriyle sınırlıdır; dünyadaki ilgisiz
        # bütün kök etkileri “kept” diye şişirilmez.
        without = [x for x in actual if x in reachable]
        lost = [x for x in actual if x not in without]
        kept = list(without)
        return {"node": s, "effects": actual, "effects_without": without,
                "lost": lost, "kept": kept, "paths": base.get("paths") or [],
                "confounders": base.get("confounders") or [],
                "identified": bool(base.get("identified")), "structural": True,
                "disabled_edges": sorted(disabled), "engine": "hdc_mb+graph",
                "counterfactual_roots": roots}

    def check_relation(self, subj: str, target_pred: str) -> dict:
        """'X Y yapar mı / X Y midir / X Y'ye yol açar mı?' sorusu için geçişli kanıt zinciri arar."""
        s = tr_lower(subj).strip()
        t = tr_lower(target_pred).strip()
        t_clean = re.sub(r"\s*(yol açar|neden olur)$", "", t).strip()
        t_parts = t_clean.split() if t_clean else []
        t_stem = TurkishMorphology.analyze_word(t_parts[-1], self.lexicon)["stem"] if t_parts else ""
        t_phrase_stem = (" ".join(t_parts[:-1] + [t_stem])).strip() if len(t_parts) > 1 else t_stem

        # Öğrenilmiş yeni operatör: yüklem artık sözlükte olmasa bile aynı
        # sözdizimsel rol çözümlemesiyle sorgulanır. ISA'ya veya CAN_DO'ya
        # sessizce çevrilmez; kendi probation ilişki kimliği aranır.
        dynamic_root = self._aor_root(t_parts[-1]) if t_parts else ""
        dynamic_rel = self.relation_by_verb.get(dynamic_root, "")
        if dynamic_rel:
            obj_parts = t_parts[:-1]
            if obj_parts:
                dynamic_obj = " ".join(obj_parts[:-1] + [self._dynamic_noun(obj_parts[-1])]).strip()
                for edge, isa_path in self.query(s, (dynamic_rel,)):
                    if edge.obj == dynamic_obj:
                        return {"answer": True, "edge": edge, "isa_path": isa_path,
                                "invented_relation": dynamic_rel}
            return {"answer": None, "edge": None, "isa_path": []}

        # 1) Doğrudan veya kalıtsal negatif istisna var mı?
        for e, isa_path in self.query(s, ("CANNOT_DO",)):
            e_last = TurkishMorphology.analyze_word(e.obj.split()[-1], self.lexicon)["stem"] if e.obj else ""
            if e.obj in (t, t_stem, t_phrase_stem) or (len(t_parts) >= 1 and e_last == t_stem and e.obj.split()[0] == t_parts[0]):
                return {"answer": False, "edge": e, "isa_path": isa_path}

        # 2) Doğrudan veya ISA-kalıtsal pozitif bağ var mı?
        for e, isa_path in self.query(s, ("ISA", "HAS_PROP", "HAS_PART", "COMPOSED_OF", "CAN_DO", "CAUSES", "USED_FOR", "LOCATED_AT")):
            e_stem = TurkishMorphology.analyze_word(e.obj.split()[-1], self.lexicon)["stem"] if e.obj else ""
            if e.obj == t or e.obj == t_clean or e.obj == t_stem or e_stem == t_stem or t_clean in e.obj:
                return {"answer": True, "edge": e, "isa_path": isa_path}

        # 3) Çok-adımlı nedensel zincir (A -> B -> C): ör. 'kuraklık güçsüzlüğe yol açar mı?'
        for eff, cpath in self.causal_chain(s):
            eff_stem = TurkishMorphology.analyze_word(eff.split()[-1], self.lexicon)["stem"] if eff else ""
            if eff == t_clean or eff == t_stem or eff_stem == t_stem:
                return {"answer": True, "edge": cpath[-1], "isa_path": [], "causal_path": cpath}

        # 4) Zıt özellik kontrolü (ör. 'ateş soğuk mudur?' -> ateş sıcaktır, sıcak != soğuk -> Hayır!)
        opposites = {e.obj for e, _path in self.query(t_stem, ("OPPOSITE",))}
        if opposites:
            for e, isa_path in self.query(s, ("HAS_PROP",)):
                if e.obj in opposites:
                    return {"answer": False, "edge": e, "isa_path": isa_path, "contrast": True}

        return {"answer": None, "edge": None, "isa_path": []}

    # -------------------------------------------------- HDC vektör cebiri + bağlam
    def _hdc_role_filler(self, relation: str, obj: str) -> np.ndarray:
        """Rol ve dolguyu gerçek bipolar bağlama ile tek vektörde birleştir."""
        r = self.relation_mb._symbol("role", str(relation))
        f = self.relation_mb._symbol("filler", tr_lower(obj).strip())
        # Zayıf dolgu yolu, HDCMushroomBody.encode ile aynı kod ailesindedir;
        # asıl anlam rol*dolgu bağlı teriminden gelir.
        return (r * f + 0.20 * f).astype(np.float32)

    @staticmethod
    def _analogy_role(relation: str) -> str:
        """Copula/taksonomi ile özellik önermesini aynı analoji rolünde buluştur.

        ``Kral erkektir`` ve ``Kraliçe kadındır`` yüzeyde farklı ayrıştırıcı
        dallarına düşebilir; bu, anlam vektöründe cinsiyet köprüsünü bozmamalı.
        Özgün Graph kenarı silinmez/değiştirilmez, yalnız vektör rolü normalize
        edilir. Nedensel ve işlevsel roller ayrı kalır.
        """
        return "ATTRIBUTE" if relation in {"ISA", "HAS_PROP"} else relation

    def _semantic_bundle(self, concept: str) -> tuple[np.ndarray, list[RelationEdge]]:
        """Kavramın öğrenilmiş doğrudan önermelerini bağlı vektörler olarak topla.

        Düz ``concept_vecs`` veya ilişki etiketi aktarımı kullanılmaz. Önermeler
        önce MB'den okunur ve graf yönüyle doğrulanır; vektör toplamı yalnız bu
        kanıtlı olaylardan kurulur. Normalize etmeden döndürmek, çıkarma sırasında
        terimlerin gerçekten birbirini götürebilmesini sağlar.
        """
        name = tr_lower(concept).strip()
        terms: list[np.ndarray] = []
        used: list[RelationEdge] = []
        for edge, isa_path in self.query(name, None):
            if isa_path or edge.subj != name:
                continue
            terms.append(self._hdc_role_filler(self._analogy_role(edge.rel), edge.obj))
            used.append(edge)
        if terms:
            return np.sum(terms, axis=0, dtype=np.float32), used
        return np.zeros(self.relation_mb.dim, dtype=np.float32), used

    def vector_analogy(self, a: str, b: str, c: str) -> dict:
        """``a - b + c`` analojisini bağlı HDC vektörleriyle çöz.

        Önce A→B önermesinin *ilişki rolü* öğrenilmiş kenardan bulunur. Sonra
        ``v(A) - bind(rol,B) + bind(rol,C)`` hesaplanır. Adaylar ortak önermelerin
        ters HDC/MB okumasından gelir; sıralama vektör kosinüsüyle yapılır, Graph'ta
        ``C`` ilişkisini arayıp nesneyi kopyalamak yoktur.
        """
        a, b, c = (tr_lower(x).strip() for x in (a, b, c))
        empty = {"status": "unresolved", "answer": None, "a": a, "b": b, "c": c,
                 "engine": "hdc_vector_binding+mb", "candidates": []}
        if not a or not b or not c or len({a, b, c}) < 3:
            empty["reason"] = "üç ayrı kavram gerekli"
            self.last_vector_analogy = empty
            return empty
        a_rows = [(e, p) for e, p in self.query(a, None) if not p and e.subj == a]
        ab = [e for e, _p in a_rows if e.obj == b]
        if not ab:
            empty["reason"] = "çıkarılacak rolün öğrenilmiş A→B bağı yok"
            self.last_vector_analogy = empty
            return empty
        priority = {"HAS_PROP": 0, "ISA": 1, "HAS_PART": 2, "COMPOSED_OF": 3,
                    "USED_FOR": 4, "CAN_DO": 5, "LOCATED_AT": 6}
        ab.sort(key=lambda e: (priority.get(e.rel, 20), -e.weight))
        chosen = ab[0]
        role = chosen.rel
        binding_role = self._analogy_role(role)
        va, a_used = self._semantic_bundle(a)
        subtract = self._hdc_role_filler(binding_role, b)
        add = self._hdc_role_filler(binding_role, c)
        query_vec = (va - subtract + add).astype(np.float32)
        qnorm = float(np.linalg.norm(query_vec))
        if qnorm < 1e-8:
            empty["reason"] = "bağlı vektör boş"
            self.last_vector_analogy = empty
            return empty

        # Ortak özellik, arama uzayını daraltır; cevap değildir. Örneğin aynı
        # sınıfı taşıyan bütün kavramlar gelir, hangisinin kaldığına vektör karar verir.
        candidate_names: set[str] = set()
        for edge in a_used:
            if edge is chosen:
                continue
            for other, _activation in self.relation_object_candidates(edge.rel, edge.obj):
                if other.subj not in {a, b, c} and other.subj != edge.obj:
                    candidate_names.add(other.subj)
        scored: list[dict] = []
        qk = set(int(x) for x in self.relation_mb.mb.encode_kc(query_vec))
        for name in sorted(candidate_names):
            vec, used = self._semantic_bundle(name)
            norm = float(np.linalg.norm(vec))
            if norm < 1e-8:
                continue
            cosine = float(np.dot(query_vec, vec) / (qnorm * norm))
            tk = set(int(x) for x in self.relation_mb.mb.encode_kc(vec))
            sparse = len(qk & tk) / max(1, min(len(qk), len(tk)))
            scored.append({"concept": name, "score": cosine, "sparse_overlap": sparse,
                           "evidence": [[e.subj, e.rel, e.obj] for e in used]})
        scored.sort(key=lambda row: (-row["score"], -row["sparse_overlap"], row["concept"]))
        top = scored[0] if scored else None
        second = float(scored[1]["score"]) if len(scored) > 1 else -1.0
        margin = float(top["score"] - second) if top else 0.0
        resolved = bool(top and top["score"] >= 0.20 and margin >= 0.03)
        result = {
            "status": "resolved" if resolved else "unresolved",
            "answer": top["concept"] if resolved else None,
            "a": a, "b": b, "c": c, "relation": role,
            "score": float(top["score"]) if top else 0.0, "margin": margin,
            "candidates": scored[:12], "engine": "hdc_vector_binding+mb",
            "operation": "v(A)-bind(role,B)+bind(role,C)",
            "query_vector": query_vec, "active_kc": len(qk),
            "reason": "" if resolved else "vektör benzerliği veya aday marjı yetersiz",
        }
        self.last_vector_analogy = {k: v for k, v in result.items() if k != "query_vector"}
        self.last_decision = {
            "engine": "hdc_vector_binding+mb", "mode": "vector_analogy",
            "query": f"{a}-{b}+{c}", "graph_validated": True,
            "results": int(resolved), "active_kc": len(qk),
            "actual_sparsity": len(qk) / max(1, self.relation_mb.mb.n_kc),
            "score": result["score"], "margin": margin,
        }
        return result

    def disambiguate(self, concept: str, context: str) -> dict:
        """Çoklu anlamı, bağlam ve anlam önermelerinin seyrek KC örtüşmesiyle seç.

        ISA adayları ve her anlamın kanıtı MB'den okunur. Bağlam sözcükleri rol-
        dolgu bağlamasıyla kodlanır; yalnız hazır ``yüz→organ`` kuralı yoktur.
        Ortak bağlam kanıtı ve yeterli skor marjı yoksa seçim yapılmaz.
        """
        name = tr_lower(concept).strip()
        stop = {"hangi", "anlamda", "anlamı", "anlamını", "nedir", "ne", "bir", "bu",
                "şu", "o", "ve", "veya", "ile", "için", "çok", "daha", "mi", "mı", "mu", "mü"}

        def tokens(text: str) -> list[str]:
            out = []
            for word in re.findall(r"[^\W\d_]+", tr_lower(text), flags=re.UNICODE):
                an = TurkishMorphology.analyze_word(word, self.lexicon)
                stem = tr_lower(an.get("stem") or word).strip()
                if stem and stem not in stop and stem != name and len(stem) > 1:
                    out.append(stem)
            return list(dict.fromkeys(out))

        context_tokens = tokens(context)
        context_terms = [self._hdc_role_filler("context", tok) for tok in context_tokens]
        context_vec = (np.sum(context_terms, axis=0, dtype=np.float32) if context_terms
                       else np.zeros(self.relation_mb.dim, dtype=np.float32))
        context_norm = float(np.linalg.norm(context_vec))
        context_kc = set(int(x) for x in self.relation_mb.mb.encode_kc(context_vec))
        candidates = []
        for isa, path in self.query(name, ("ISA",)):
            if path or isa.subj != name:
                continue
            sense = isa.obj
            rows = [(e, p) for e, p in self.query(sense, None) if not p and e.subj == sense]
            sense_tokens = [sense]
            evidence = []
            for edge, _p in rows:
                edge_tokens = tokens(f"{edge.subj} {edge.obj}")
                sense_tokens.extend(edge_tokens)
                if set(edge_tokens) & set(context_tokens):
                    evidence.append([edge.subj, edge.rel, edge.obj])
            sense_tokens = list(dict.fromkeys(sense_tokens))
            terms = [self._hdc_role_filler("context", tok) for tok in sense_tokens]
            vec = (np.sum(terms, axis=0, dtype=np.float32) if terms
                   else np.zeros(self.relation_mb.dim, dtype=np.float32))
            norm = float(np.linalg.norm(vec))
            sense_kc = set(int(x) for x in self.relation_mb.mb.encode_kc(vec))
            cosine = (float(np.dot(context_vec, vec) / (context_norm * norm))
                      if context_norm > 1e-8 and norm > 1e-8 else 0.0)
            sparse = (len(context_kc & sense_kc) / max(1, min(len(context_kc), len(sense_kc)))
                      if context_kc and sense_kc else 0.0)
            shared = sorted(set(context_tokens) & set(sense_tokens))
            score = 0.65 * sparse + 0.35 * cosine
            candidates.append({"sense": sense, "score": float(score), "cosine": cosine,
                               "sparse_overlap": float(sparse), "shared": shared,
                               "evidence": evidence, "active_kc": len(sense_kc)})
        candidates.sort(key=lambda row: (-row["score"], row["sense"]))
        top = candidates[0] if candidates else None
        second = float(candidates[1]["score"]) if len(candidates) > 1 else 0.0
        margin = float(top["score"] - second) if top else 0.0
        resolved = bool(top and top["shared"] and top["sparse_overlap"] > 0.0
                        and top["score"] >= 0.15 and margin >= 0.04)
        result = {"status": "resolved" if resolved else "ambiguous", "concept": name,
                  "sense": top["sense"] if resolved else None, "score": float(top["score"]) if top else 0.0,
                  "margin": margin, "context_tokens": context_tokens, "candidates": candidates,
                  "engine": "hdc_mb_sparse_context", "active_kc": len(context_kc),
                  "actual_sparsity": len(context_kc) / max(1, self.relation_mb.mb.n_kc)}
        self.last_disambiguation = result
        self.last_decision = {"engine": "hdc_mb_sparse_context", "mode": "disambiguation",
                              "query": name, "graph_validated": True, "results": int(resolved),
                              "active_kc": len(context_kc), "actual_sparsity": result["actual_sparsity"],
                              "score": result["score"], "margin": margin}
        return result

    def solve_analogy(self, a: str, b: str, c: str) -> tuple[str | None, str]:
        """Eski ``a:b::c:?`` API'sini koru; aritmetik analojide HDC'ye düş.

        Klasik ilişki aktarımı mümkünse eski çağrıların anlamı değişmez. Fakat
        ``Kral − Erkek + Kadın`` gibi c'de aynı etiketle cevap bulunamayan
        durumda artık ``None`` dönüp durmak yerine bağlı vektör cebiri çalışır;
        sonuç yalnız Graph etiketi kopyalanarak üretilmez.
        """
        a_l, b_l, c_l = tr_lower(a), tr_lower(b), tr_lower(c)
        # 1) Doğrudan ilişki türü eşleşmesi (eski API'nin klasik biçimi)
        for e_ab, _ in self.query(a_l):
            if e_ab.obj == b_l:
                for e_c, _ in self.query(c_l, (e_ab.rel,)):
                    return e_c.obj, e_ab.rel
        # 2) Zıtlık ilişkisi (adaylar da MB aktivasyonundan gelir)
        for e, _path in self.query(a_l, ("OPPOSITE",)):
            if e.obj == b_l:
                for e2, _path in self.query(c_l, ("OPPOSITE",)):
                    return e2.obj, "OPPOSITE"
        # 3) Vektör aritmetiği: v(a)-bind(rol,b)+bind(rol,c).
        vector = self.vector_analogy(a_l, b_l, c_l)
        if vector.get("status") == "resolved":
            return vector.get("answer"), vector.get("relation", "HDC_VECTOR")
        return None, ""


class BrocaGenerator:
    """Önerme ağaçlarını (Proposition) üretken Türkçe morfolojiyle doğal cümlelere dönüştürür."""

    def __init__(self):
        self.morph = TurkishMorphology()
        self.connective_counts: dict[str, int] = {"çünkü": 1, "dolayısıyla": 1, "ancak": 1}

    def _copula_phrase(self, phrase: str) -> str:
        """Çok sözcüklü yüklem öbeğinde koşaç YALNIZCA son sözcüğe gelir ('çok sıcaktır',
        'bir değerli metaldir'); tam dizgeye uygulanınca 'sıcakır' gibi bozuk biçimler çıkıyordu."""
        # Clean up detached Turkish suffixes: e.g. "devletleri nin" -> "devletleri'nin"
        phrase = re.sub(r"(\w+)\s+(nın|nin|nun|nün|ın|in|un|ün)\b", r"\1'\2", phrase)
        clean = phrase.strip().rstrip(".!?,")
        parts = clean.split()
        if not parts:
            return phrase
        # Kullanıcıdan öğrenilen özel/taksonomik ad: anlam anahtarı küçük harfli
        # olsa da yüzeyde 'Aves'tir' biçimi ve kesme işareti korunur.
        if parts[-1][:1].isupper():
            proper = parts[-1]
            low = tr_lower(proper)
            suffix = self.morph.dental_d_t(low) + self.morph.vowel_4(low) + "r"
            return " ".join(parts[:-1] + [proper + "'" + suffix])
        # Yüklem zaten çekimli bir fiilse veya zaten ek-fiil (koşaç) taşıyorsa yeniden koşaç EKLENMEZ:
        last_an = self.morph.analyze_word(parts[-1])
        if (last_an.get("is_verb") and str(last_an.get("person", "")) in ("3SG", "3PL")) or last_an.get("copula"):
            return clean
        return " ".join(parts[:-1] + [self.morph.inflect_noun(parts[-1], "NOM", copula=True)])

    def _gen_phrase(self, phrase: str) -> str:
        """Çok sözcüklü öznede tamlayan eki son sözcüğe ('yer kabuğunun', 'basınç farkının')."""
        parts = phrase.split()
        if not parts:
            return phrase
        return " ".join(parts[:-1] + [self.morph.inflect_noun(parts[-1], "GEN")])

    def _head_case(self, head: str, case: str, compound: bool) -> str:
        """Ad tamlamasının BAŞINA hâl ekini doğru takar ('besin üretimi' -> 'üretimine')."""
        an = self.morph.analyze_word(head)
        if an.get("poss3") and not an.get("is_verb"):
            return self.morph.inflect_noun(an.get("stem") or head, case, poss3=True)
        # Belirtisiz ad tamlamasında baş iyelik taşır: 'oksijen ve besin üretimi' içindeki
        # 'üretimi' = üretim+iyelik, 'üretim+i(hâl)' değil. Ayrıştırıcı ikisini de geçerli
        # sayabildiği için (ACC okuması) tamlama bağlamında iyelik okuması seçilir.
        if (compound and case != "NOM" and len(head) >= 5 and head[-1] in "ıiuü"
                and head[-2] not in "aeıioöuü"):
            stem = head[:-1]
            if stem.endswith("ğ"):
                stem = stem[:-1] + "k"
            if len(stem) >= 3 and stem[-1] not in "aeıioöuü":
                return self.morph.inflect_noun(stem, case, poss3=True)
        return self.morph.inflect_noun(head, case)

    def _case_phrase(self, phrase: str, case: str, poss: int = 0) -> str:
        """Çok sözcüklü öbekte hâl eki YALNIZCA son sözcüğe gelir ('oksijen ve besin üretimine')."""
        parts = phrase.split()
        if not parts:
            return phrase
        head = parts[-1]
        if poss == 3:
            tail = self.morph.inflect_noun(head, case, poss3=True)
        elif poss == 0:
            tail = self._head_case(head, case, compound=len(parts) > 1)
        else:
            tail = self.morph.inflect_noun(head, case)
        return " ".join(parts[:-1] + [tail])

    def _surface_case(self, phrase: str, case: str) -> str:
        """Tek sözcüklü özel adı hâl ekiyle yaz; diğer öbeklerde mevcut motoru kullan."""
        if phrase and " " not in phrase and phrase[:1].isupper():
            low = tr_lower(phrase)
            inflected = self.morph.inflect_noun(low, case)
            suffix = inflected[len(low):] if inflected.startswith(low) else inflected
            return phrase + "'" + suffix
        return self.morph.inflect_noun(phrase, case)

    def realize_prop(self, p: Proposition) -> str:
        s = p.subj.strip()
        o = p.obj.strip()
        # Normalise detached suffixes in raw text: e.g. "Devletleri nin" -> "Devletleri'nin"
        o = re.sub(r"(\w+)\s+(nın|nin|nun|nün|ın|in|un|ün)\b", r"\1'\2", o)
        # Normalise screaming caps grammatical words from noisy QA datasets
        for word in ("BİR", "VE", "GENEL", "EKONOMİK", "İFADE", "EDER", "OLARAK"):
            if word in o:
                o = o.replace(word, tr_lower(word))
        r = p.rel

        if r == "CAPITAL":
            clean_o = o.rstrip(".!?, ")
            s_gen = self._surface_case(s, "GEN")
            pred = self._copula_phrase(clean_o)
            return f"{s_gen} başkenti {clean_o} değildir" if p.neg else f"{s_gen} başkenti {pred}"

        if r == "ISA":
            clean_o = o.rstrip(".!?, ")
            # Belirtili tamlamada belgisiz artikel tamlayanın önüne gelmez:
            # 'yaşamın temel birimidir', 'bir yaşamın temel birimidir' değil.
            has_genitive = bool(re.search(r"(?:^|\s)\S+(?:nın|nin|nun|nün|ın|in|un|ün)(?:\s|$)", clean_o))
            art = "" if clean_o.startswith("bir ") or has_genitive else "bir "
            pred = self._copula_phrase(art + clean_o)
            return f"{s} {art}{clean_o} değildir" if p.neg else f"{s} {pred}"

        if r == "HAS_PROP":
            clean_o = o.rstrip(".!?, ")
            if clean_o.startswith("başkenti "):
                s_gen = self._surface_case(s, "GEN")
                return f"{s_gen} {self._copula_phrase(clean_o)}"
            if p.neg:
                return f"{s} {clean_o} değildir"
            return f"{s} {self._copula_phrase(clean_o)}"

        if r == "COMPOSED_OF":
            # Bileşim yalnız 'bir parçası var' değildir; çoğul ablatif bunu açıkça
            # korur: 'ökaryot hücre' -> 'ökaryot hücrelerden oluşur'.
            parts = o.split()
            if parts:
                abl = self.morph.inflect_noun(parts[-1], "ABL", plural=True)
                material = " ".join(parts[:-1] + [abl])
            else:
                material = o
            return f"{s} {material} oluşmaz" if p.neg else f"{s} {material} oluşur"

        if r == "HAS_PART":
            # Sahiplik: 'zürafanın uzun boynu var' (parça/özellik adı iyelikli kurulur)
            parts = o.split()
            head = parts[-1] if parts else o
            # Öğe zaten 3. çoğul iyelik taşıyorsa ('duyu organları') yeniden iyeliklenmez.
            if head.endswith(("ları", "leri")):
                part = o
            else:
                part = " ".join(parts[:-1] + [self.morph.inflect_noun(head, "NOM", poss3=True)]).strip()
            if p.neg:
                return f"{self._surface_case(s, 'GEN')} {part} yok"
            return f"{self._surface_case(s, 'GEN')} {part} var"

        if r in ("CAN_DO", "CANNOT_DO"):
            is_neg = p.neg or (r == "CANNOT_DO")
            parts = o.split()
            # Bazı öğretmen kayıtları eylemi mastar olarak taşır ('besin aramak').
            # Kök doğrulanabiliyorsa CAN_DO bunu geniş zamana çeker; aksi hâlde
            # yüklem yarım 'Hayvan besin aramak.' diye kalır.
            if parts and parts[-1].endswith(("mak", "mek")):
                _inf = parts[-1]
                _root = _inf[:-3]
                if _root in self.morph.KNOWN_VERB_ROOTS or self.morph._unsoften(_root) in self.morph.KNOWN_VERB_ROOTS:
                    parts[-1] = _root if _root in self.morph.KNOWN_VERB_ROOTS else self.morph._unsoften(_root)
                    o = " ".join(parts)
            # Öğrenilmiş tamamlanmış fiil ('belleği pekiştirir') yeniden çekimlenmez: çift ek olurdu.
            if parts and parts[-1].endswith(("ir", "ır", "ur", "ür", "yor", "di", "dı", "ti", "tı",
                                             "ecek", "acak", "meli", "malı", "maz", "mez")):
                return f"{s} {' '.join(parts)}"
            # Yüklem bir FİİL değilse (sıfat/ad) çekimlenmez, koşaç alır:
            # 'hava çok sıcakmış' -> 'hava çok sıcaktır' ('sıcakır' değil).
            try:
                _la = self.morph.analyze_word(parts[-1] if parts else o)
            except Exception:
                _la = {}
            _last = parts[-1] if parts else o
            _is_root = _last in self.morph.KNOWN_VERB_ROOTS
            if not _is_root and parts and not _la.get("is_verb") and not _la.get("tense"):
                if is_neg:
                    return f"{s} {o} değildir".strip()
                return f"{s} {self._copula_phrase(o)}".strip()
            if not _is_root and _la.get("is_verb") and _la.get("tense"):
                # Yüklem ZATEN çekimli ('uçar', 'yaşar'): ikinci kez çekimlenmez.
                return f"{s} {o}".strip()
            if len(parts) == 1:
                v_form = self.morph.conjugate_verb(parts[0], tense=p.tense, neg=is_neg, person=p.person)
                return f"{s} {v_form}".strip()
            v_form = self.morph.conjugate_verb(parts[-1], tense=p.tense, neg=is_neg, person=p.person)
            return f"{s} {' '.join(parts[:-1])} {v_form}".strip()

        if r in ("CAUSES", "USED_FOR") and s.split() and s.split()[-1].endswith(("mak", "mek")):
            # 'rüya görmek' -> 'rüya görmenin nedeni' (mastar, isim gibi çekilmez)
            try:
                from nora_broca import TurkishSyntaxEngine as _SYN
                s_gen = _SYN.realize_verbal_noun(s, "GEN")
            except Exception:
                s_gen = s
            if r == "CAUSES":
                tail = self._copula_phrase(o)
                return f"{s_gen} nedeni {tail}" if not p.neg else f"{s_gen} nedeni {o} değildir"
        if r == "CAUSES":
            # DİKKAT: "X'in nedeni Y'dir" YÖNÜ TERS çevirir — kenar X->Y iken "X'in nedeni Y"
            # demek, Y'yi neden yapar. Doğru kuruluş: neden özne, etki yönelme hâlinde.
            # Çok sözcüklü öbekte eki YALNIZ son sözcüğe ver ('basınç farkı' -> 'basınç farkına').
            if len(o.split()) > 1:
                # _case_phrase eki ZATEN yalnız son sözcüğe takar; tamlama bağlamını
                # korumak için öbeğin tamamı verilir ('besin üretimine', 'üretimiye' değil).
                try:
                    dat = self._case_phrase(o, "DAT")
                except Exception:
                    dat = o
                return (f"{s} {dat} yol açmaz" if p.neg else f"{s} {dat} yol açar")
            if o.endswith(("ı", "i", "u", "ü")):
                gen = self._gen_phrase(s)
                return f"{gen} nedeni {self._copula_phrase(o)}" if not p.neg else f"{gen} nedeni {o} değildir"
            if p.neg:
                return f"{s} {self._case_phrase(o, 'DAT')} yol açmaz"
            return f"{s} {self._case_phrase(o, 'DAT')} yol açar"

        if r == "USED_FOR":
            # Eylemlikle kurulmuş amaç ("enerji kazanma") doğal Türkçede "-mak için" ister:
            # "yem, enerji kazanmak için kullanılır" (kazanma için değil).
            if " " in o and o.endswith(("ma", "me")) and len(o.rstrip()) > 4:
                o_aim = o[:-2] + ("mak" if o.endswith("ma") else "mek")
                return f"{s} {o_aim} için kullanılır"
            return f"{s} {o} için kullanılır"

        if r == "LOCATED_AT":
            # Sözlükte önceden bulunmayan özel/taksonomik adlar da öğrenilebilir.
            # Yalnız donmuş PROPER_NOUNS listesine bakmak, kullanıcıdan öğrenilen
            # 'Aves'i yüzeyde sıradan ada çevirip 'aveste' üretiyordu. Büyük harfli
            # yeni adlar da diğer özel adlarla aynı kesme ve hâl kuralını kullanır.
            learned_proper = bool(o.split() and o.split()[-1][:1].isupper())
            if o in PROPER_NOUNS or learned_proper:
                # Özel ad kesme işaretiyle çekimlenir: "Himalaya'da", "Aves'te".
                loc = tr_capitalize(o) + "'" + self._case_phrase(o, "LOC")[len(o):]
            else:
                loc = self._case_phrase(o, "LOC")
            return f"{s} {loc} bulunur"

        if r == "OPPOSITE":
            # "Sıcak, soğuğun zıttıdır." — özne tekrarlanmaz, tamlayan kurulur.
            gen = self._gen_phrase(o)
            if p.neg:
                return f"{s} {gen} zıttı değildir"
            return f"{s} {gen} zıttıdır"

        if r == "IF_THEN":
            return f"{s} olduğunda {o}"

        if r == "ACTION_SELF":
            v_form = self.morph.conjugate_verb(o, tense=p.tense, neg=p.neg, person="1SG")
            return f"{s} {v_form}".strip() if s else v_form

        if r.startswith("YENI_OPERATOR_"):
            # Yeni ilişkinin operatörü koda gömülü değildir: duyulan fiil kökü
            # ve nesne hâl/çoğul bilgisi önermeyle Broca'ya gelir.
            root = p.verb or tr_lower(r[len("YENI_OPERATOR_"):])
            obj = p.obj_surface
            if not obj and o:
                obj = self.morph.inflect_noun(o, p.obj_case, plural=p.obj_plural)
            v_form = self.morph.conjugate_verb(root, tense=p.tense, neg=p.neg, person=p.person)
            return f"{s} {obj} {v_form}".strip()

        return f"{s} {o}".strip()

    def realize_epistemic(self, event: RelationEdge, *, concept_surface: str = "",
                          known: list[Proposition] | None = None,
                          missing: list[str] | None = None) -> str:
        """Graph epistemik olayını hazır cevap değil, çekimli düşünce olarak söyle."""
        surface = str(concept_surface or event.subj).strip()
        state = tr_lower(event.obj).strip()
        knowledge = self.morph.inflect_noun("bilgi", "NOM", poss1=True)
        definition = self.morph.inflect_noun("tanım", "NOM", poss1=True)
        if state == "unknown":
            word_acc = self.morph.inflect_noun("kelime", "ACC", poss3=True)
            heard = self.morph.conjugate_verb("duy", tense="PAST", person="1SG")
            # Bilinmezlik cümlesinin kalıbı donanım entropisiyle seçilir: her seferinde aynı
            # 'kavram bilgim yok' cümlesi dönmez; anlam aynı, söyleyiş değişir.
            frames = (
                f"{surface} {word_acc} {heard}; kavram {knowledge} yok ve doğrulanmış {definition} yok.",
                f"{surface} hakkında {knowledge} yok; doğrulanmış bir {definition} da yok.",
                f"{surface} {word_acc} {heard}, ama onun için doğrulanmış bir {definition} bende yok.",
                f"{surface} için doğrulanmış bir {definition} yok; {knowledge} de henüz eksik.",
            )
            text = frames[int.from_bytes(os.urandom(2), "big") % len(frames)]
            return tr_capitalize(text)
        clauses = []
        seen_content: list[set[str]] = []
        for proposition in (known or [])[:(6 if state == "insufficient" else 4)]:
            obj_clean = proposition.obj.strip().rstrip(".!?")
            # Gürültü filtresi: HAS_PROP içindeki tek sözcüklü izole fiil kökleri ('yap', 'art', 'netice')
            if proposition.rel == "HAS_PROP" and len(obj_clean.split()) == 1 and tr_lower(obj_clean) in {
                "yap", "et", "ol", "art", "git", "gel", "netice", "dur"
            }:
                continue
            realized = self.realize_prop(proposition).strip().rstrip(".!?")
            if not realized:
                continue
            words = set(re.findall(r"\w+", tr_lower(realized)))
            content_sig = {w for w in words if len(w) > 2 and w not in {"bir", "ve", "ile", "için", "olan", "gibi", "kadar"}}
            is_dup = False
            for prev_sig in seen_content:
                overlap = len(content_sig & prev_sig)
                if content_sig and (overlap / max(len(content_sig), 1) >= 0.65 or overlap / max(len(prev_sig), 1) >= 0.65):
                    is_dup = True
                    break
            if is_dup:
                continue
            seen_content.append(content_sig)
            clauses.append(tr_capitalize(realized) + ".")
        if state == "insufficient":
            gap = ", ".join(str(item) for item in (missing or []) if str(item).strip())
            if not gap:
                gap = "bağlam"
            clauses.append(f"Tanım için {gap} kanıtı eksik; {knowledge} yetersiz.")
            return " ".join(clauses)
        learned = self.morph.conjugate_verb("öğren", tense="PAST", person="1SG")
        clauses.append(tr_capitalize(f"{surface} için yeni kanıt {learned}."))
        return " ".join(clauses)

    def realize_shared_features(self, left: str, right: str,
                                shared: list[tuple[str, str]]) -> str:
        """Ortaklık çıkarımının yalnız doğrulanmış özelliklerini çekimle."""
        subject = f"{left} ve {right}"
        if not shared:
            not_found = self.morph.conjugate_verb("bul", tense="PAST", neg=True, person="1SG")
            return tr_capitalize(f"{subject} için ortak doğrulanmış anlam özelliği {not_found}.")
        clauses = []
        for rel, obj in shared[:4]:
            text = self.realize_prop(Proposition(subj=subject, rel=rel, obj=obj)).strip()
            if text:
                clauses.append(text.rstrip(".!?"))
        if not clauses:
            not_found = self.morph.conjugate_verb("bul", tense="PAST", neg=True, person="1SG")
            return tr_capitalize(f"{subject} için ortak doğrulanmış anlam özelliği {not_found}.")
        return tr_capitalize("; ".join(clauses)) + "."

    def realize_inverse_identification(self, candidates: list[dict]) -> str:
        """Birleşik koşul sonucunu ve onu zorunlu kılan kanıtı birlikte söyle."""
        if not candidates:
            not_found = self.morph.conjugate_verb("bul", tense="PAST", neg=True, person="1SG")
            return tr_capitalize(f"Bu ölçütleri birlikte karşılayan kavram {not_found}.")
        names = [str(row["subject"]) for row in candidates[:4]]
        if len(candidates) > 1:
            return tr_capitalize(f"{', '.join(names)} bu ölçütleri birlikte karşılıyor.")
        row = candidates[0]
        proof = []
        for proposition in row.get("proof") or []:
            text = self.realize_prop(proposition).strip().rstrip(".!?")
            if text:
                proof.append(text)
        suffix = f"; çünkü {' ve '.join(proof)}" if proof else ""
        return tr_capitalize(f"{row['subject']}{suffix}.")

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
            # Doğrudan bilgi; yeni ilişkinin fiil ve hâl bilgisi de MB'nin
            # doğrulanmış önermesinden Broca'ya taşınır.
            main = Proposition(
                subj=subj, rel=edge.rel, obj=edge.obj,
                neg=(not answer and edge.rel != "CANNOT_DO"), verb=edge.verb,
                obj_surface=edge.obj_surface, obj_case=edge.obj_case, obj_plural=edge.obj_plural,
            )
            prefix = "Evet" if answer else "Hayır"
            ep = "deneyimime göre" if edge.source == "deneyim" else ""
            return self.compose(main, prefix=prefix, epistemic_note=ep)

        # Geçişli çıkarım: 'Evet, karga bir kuştur ve kuş uçar; dolayısıyla karga uçar.'
        first_isa = isa_path[0]
        p1 = self.realize_prop(Proposition(subj=first_isa.subj, rel="ISA", obj=first_isa.obj))
        p2 = self.realize_prop(Proposition(
            subj=edge.subj, rel=edge.rel, obj=edge.obj, verb=edge.verb,
            obj_surface=edge.obj_surface, obj_case=edge.obj_case, obj_plural=edge.obj_plural,
        ))
        concl = self.realize_prop(Proposition(
            subj=subj, rel=edge.rel if answer else "CANNOT_DO", obj=edge.obj,
            verb=edge.verb, obj_surface=edge.obj_surface,
            obj_case=edge.obj_case, obj_plural=edge.obj_plural,
        ))
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
    math_expr: tuple[int, str, int] | str | None = None
    analogy: tuple[str, str, str] | None = None
    sub_kind: str = ""
    wh: str = ""              # soru sözcüğü: nasıl | neden | nerede | kim | ne ...
    obj_surface: str = ""     # özel/taksonomik adın duyulan yüzeyi; anlam anahtarı obj'de kalır
    verb: str = ""            # bilinmeyen yüklem için sözdiziminden çıkarılan kök
    obj_case: str = "NOM"
    obj_plural: bool = False


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

    # ------------------------------------------------------------------ AD ÖBEĞİ
    #: Sıfat işlevli sözcükler: ad öbeğinde ÖNEK olarak kalır, ayrı kavram olmaz.
    ADJ_SEED = frozenset((
        "aşırı", "yetersiz", "büyük", "küçük", "kara", "ak", "yapay", "sanal", "hızlı", "yavaş",
        "sıcak", "soğuk", "kuru", "ıslak", "temiz", "kirli", "açık", "kapalı", "yeni", "eski",
        "zor", "kolay", "önemli", "basit", "karmaşık", "erken", "geç", "uzun", "kısa",
    ))
    #: Ad öbeğinde yüzey eki alabilen hâller: nesne sınırını bunlar belirler.
    OBLIQUE = frozenset(("ACC", "DAT", "ABL", "LOC", "INS"))
    UNIVERSAL_QUANT = frozenset(("bütün", "tüm"))

    #: Sözcük sonunda sıyrılabilen hâl ekleri (uzundan kısaya).
    _CASE_SUF = ("ndan", "nden", "ndan", "nın", "nin", "nun", "nün",
                 "dan", "den", "tan", "ten", "na", "ne", "ya", "ye",
                 "da", "de", "ta", "te", "ın", "in", "un", "ün",
                 "ı", "i", "u", "ü", "a", "e")

    def _base_of(self, w: str, an: dict | None = None) -> str:
        """Sözlükte olmayan (doğrulanmamış) yüzeyin KAVRAM GÖVDESİ.

        "kıtlığa" -> 'kıtlığ' (yanlış) değil, "kıtlık". Sıra: ek sıyır → yumuşamayı
        geri al (ğ->k) → sözlükte olan ilk adayı seç. Sözlükte hiçbiri yoksa
        yumuşatılmış biçim kullanılır; uydurma bir kök ağa yazılmaz.
        """
        an = an or self.morph.analyze_word(w, self.graph.lexicon)
        if not an.get("unverified_stem"):
            return an.get("stem", w)
        stems = getattr(self.morph, "KNOWN_VERB_ROOTS", set())
        for suf in self._CASE_SUF:
            if len(w) - len(suf) >= 3 and w.endswith(suf):
                base = w[: -len(suf)]
                # "-ma/-me" ile türemiş AD ("sulama", "birikme"): son ünlüyü sıyırmak
                # onu bozar. Fiil kökü tanınıyorsa yüzey olduğu gibi kalır.
                if suf in ("a", "e") and base in stems:
                    return w
                cands = []
                if base.endswith("ğ"):
                    cands.append(base[:-1] + "k")
                cands.extend([base + "k", base])
                for c in cands:
                    if c in self.graph.lexicon:
                        return c
                # Sözlükte yok. Yalnız YUMUŞAMA GERİ ALINABİLİYORSA gövdeye in
                # ("kıtlığa" -> "kıtlık"); aksi halde yüzey korunur ("sulama"), çünkü
                # uydurma kök ağa yazılırsa aynı kavram iki düğüme bölünür.
                if base.endswith("ğ"):
                    return base[:-1] + "k"
                # ÇÖZÜMLEYİCİ EKİ ZATEN OKUDU: hâl eki tanındıysa ve gövde ünsüzle
                # kapanıyorsa, UYDURMADAN gövdeye inmek doğrudur ("erozyona" -> "erozyon";
                # yoksa ağda "erozyona" diye bir düğüm açılıyor ve zincir kopuyordu).
                # "-ma/-me" adlaştırması hariç: orada son ünlü gövdenin parçasıdır.
                # Kök, TEK ünsüzle kapanmalı (ünlü+ünsüz). "gübr"/"kumany" gibi
                # çift-ünsüz kuyruklar Türkçe gövde değildir; orada yüzey korunur.
                _st = str(an.get("stem") or "")
                _coda_ok = (len(_st) >= 3 and _st[-1] not in "aeıioöuü"
                            and (len(_st) < 2 or _st[-2] in "aeıioöuü"))
                if (_st and _st != w and _coda_ok and not w.endswith(("ma", "me"))):
                    return _st
                return w
        return w

    def _tok(self, w: str) -> dict:
        an = dict(self.morph.analyze_word(w, self.graph.lexicon))
        an["surface"] = w
        an["head"] = self._base_of(w, an)
        return an

    @staticmethod
    def _np_name(tokens: list[str], head: dict | None) -> str:
        """Öbeğin KAVRAM ADI: niteleyiciler yüzeyiyle, baş gövdesiyle.

        Neden gövde? "kök çürümesi / kök çürümesine / kök çürümeleri" aynı düğüme
        bağlanmalı; yüzey eki düğüm adına girerse ağ parçalanır.
        """
        if not tokens:
            return ""
        if head is None:
            return " ".join(tr_lower(t) for t in tokens)
        return " ".join([tr_lower(t) for t in tokens[:-1]] + [tr_lower(str(head.get("head") or tokens[-1]))]).strip()

    @staticmethod
    def _poss_head(tok: dict) -> bool:
        """İYELİKLİ BAŞ: "kirliliği" (kirlilik+i), "kovanı" (kovan+ı), "çürümesi".

        Ayrıştırıcı bunları çoğu zaman ACC sanır (yalın okuma da geçerli olduğu için).
        Öbek İÇİNDE hâl ekli bir ad BAŞ OLAMAZ; demek ki o ek iyeliktir.
        Ölçüt: iyelik bayrağı VEYA yüksek ünlüyle biten, gövdesi ünsüzle kapanan biçim.
        """
        if tok.get("poss3"):
            return True
        surf = str(tok.get("surface") or "")
        stem = str(tok.get("stem") or "")
        return bool(tok.get("case") in ("ACC", "NOM") and len(surf) >= 4
                    and surf[-1] in "ıiuü" and len(stem) >= 3
                    and stem[-1] not in "aeıioöuü")

    def split_svo(self, words: list[str]) -> tuple[dict | None, dict | None]:
        """Cümleyi (özne öbeği, nesne öbeği) olarak ayır. Yüklem çağıranda kalır.

        Türkçe ad öbeği BAŞ-SONALIDIR. Nesne sınırı, hâl eki taşıyan SON baştır:
            "aşırı sulama | kök çürümesine"  -> özne "aşırı sulama", nesne "kök çürüme"
            "kara delik | ışığı"             -> özne "kara delik",  nesne "ışık"
        Baş iyelikliyse (belirtisiz ad tamlaması) hemen solundaki ad TAMAMLAYANDIR;
        böylece "kök çürümesi" tek kavram olur, "kök" ile "çürüme" iki ayrı düğüm olmaz.
        """
        words = [w for w in words if w]
        if not words:
            return None, None
        toks = [self._tok(w) for w in words]
        # DOĞRULANMAMIŞ KÖK HÂL EKİ TAŞIMAZ ("serçe" -> "serç"+DAT sanılıyordu).
        # Sözlükte doğrulanmamış bir kökten tek harflik ek soymak EK UYDURMAKTIR;
        # burada yalnız NESNE SINIRI için yalın kabul edilir (çözümlemenin geri kalanı
        # dokunulmaz — ad öbeği sınırı testleri A83 bunu ölçer).
        for t in toks:
            if t.get("unverified_stem") and t.get("case") in self.OBLIQUE:
                t["case"] = "NOM"
        obj_end = None
        for i, t in enumerate(toks):
            if t.get("is_verb"):
                break
            if t["case"] in self.OBLIQUE and not t.get("is_verb"):
                obj_end = i
        if obj_end is None:
            name = self._np_name(words, toks[-1])
            return ({"canonical": name, "surface": " ".join(words),
                     "case": toks[-1]["case"], "head": toks[-1]["head"]} if name else None), None
        head = toks[obj_end]
        start = obj_end
        if head.get("poss3") and obj_end >= 1:
            start = obj_end - 1           # belirtisiz ad tamlaması: tamamlayan + baş
        else:
            # NESNE TAMLAYANI: baş iyelik taşımıyorsa da nesne çok sözcüklü olabilir
            # ("deniz kirliliği | plastik atıktan"). ÖZNENİN başı iyelikliyse (belirtisiz
            # ad tamlaması) sınır ORADADIR: tamamlayanın solu özne, sağı nesne.
            # Bu kural yalnız sınırı SAĞA taşır; iyelik yoksa hiçbir şey değişmez.
            for j in range(obj_end - 1, 0, -1):
                if toks[j].get("is_verb"):
                    break
                if self._poss_head(toks[j]):
                    start = j + 1
                    break
        # "unverified_stem" tamamlayan ekleme gerekçesi DEĞİLDİR: "kuraklık kıtlığa yol
        # açar"da baş doğrulanmamış; tamamlayan sanılıp ÖZNE yutuluyordu.
        obj = {"canonical": self._np_name(words[start:obj_end + 1], head),
               "surface": " ".join(words[start:obj_end + 1]),
               "case": head["case"], "head": head["head"]}
        subj = None
        if start > 0:
            subj_toks = toks[:start]
            subj = {"canonical": self._np_name(words[:start], subj_toks[-1]),
                    "surface": " ".join(words[:start]),
                    "case": subj_toks[-1]["case"], "head": subj_toks[-1]["head"]}
        return subj, obj

    #: Soru sözcükleri: özne öbeğinin BİTTİĞİ yer.
    Q_MARKS = ("neden", "niçin", "niye", "neyden", "nedeni", "ne", "neye", "neyi",
               "nasıl", "kaç", "kim", "hangisi", "hangi", "sebep", "sebebi", "olur",
               "oldu", "olmuş", "hakkında", "biliyorsun", "mı", "mi", "mu", "mü",
               "mıdır", "midir", "musun", "misin", "var", "yok")

    def _query_subject_np(self, words: list[str]) -> str:
        """Soru öznesini AD ÖBEĞİ olarak kur ("kök çürümesi" -> "kök çürüme").

        Tek sözcüklü kök aramak bileşik kavramı kaçırıyordu: "Kök çürümesi neyden
        olur?" sorusu 'çürüme' diye aranıp ağdaki 'kök çürüme' bulunamıyordu.
        Baş-sonda kuralı: öbek, soru sözcüğünden ÖNCEKİ son addır.
        """
        pre: list[str] = []
        for w in words:
            if w in self.Q_MARKS:
                break
            pre.append(w)
        if not pre:
            return ""
        toks = [self._tok(w) for w in pre]
        parts = []
        for i, (w, t) in enumerate(zip(pre, toks)):
            if i == len(pre) - 1 or self._poss_head(t):
                parts.append(str(t.get("head") or w))
            else:
                parts.append(w)
        return " ".join(parts).strip()

    def _sent_vec(self, text: str) -> np.ndarray:
        v = np.zeros(128, dtype=np.float32)
        s = tr_lower(text)
        for tok in s.split():
            for i in range(max(1, len(tok) - 1)):
                bg = tok[i:i + 2]
                v[(_stable_hash(bg) & 0x7FFFFFFF) % 128] += 1.0
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

    def _extract_math(self, s: str) -> tuple[int, str, int] | str | None:
        def number_token(token: str) -> str | None:
            normalized = TurkishMorphology.normalize_num_token(token)
            if normalized is not None:
                return normalized
            # Aynı kavram ağında öğrenilmiş sembol-değer bilgisi, matematikte de
            # kullanılır. Kalıtsal sınıf özelliği değil, doğrudan ve tekil bağ gerekir.
            values = {
                int(edge.obj) for edge, path in self.graph.query(tr_lower(token), ("NUMERIC_VALUE",))
                if not path and re.fullmatch(r"-?\d+", str(edge.obj))
            }
            return str(next(iter(values))) if len(values) == 1 else None

        # 0) Özel 'ile' kalıbı: '125 ile 47 toplamı', '100 ile 20 farkı'
        m_ile = re.search(r"(-?\d+)\s+ile\s+(-?\d+)\s+(toplam[ıi]|fark[ıi]|çarpım[ıi]|bölüm[üu]|çıkar|eksi|artı|topla)", s)
        if m_ile:
            a, b, w = int(m_ile.group(1)), int(m_ile.group(2)), m_ile.group(3)
            op = "-" if ("fark" in w or "çıkar" in w or "eksi" in w) else ("*" if "çarp" in w else ("/" if "böl" in w else "+"))
            return (a, op, b)

        # 1) Rakamlı veya sembollü matematiksel ifade (tekil veya çoklu işlem):
        # '20 x 20 + 50 kaç', '247 + 389', '84 eksi 59', '5x5', '5*5', '10/2', '(20 + 5) * 4'
        op_map = [
            (r"\bçarpı\b|\bçarp\b|\bkere\b|[xX×]", "*"),
            (r"\bbölü\b|\bböl\b|÷", "/"),
            (r"\bartı\b|\btopla\b", "+"),
            (r"\beksi\b|\bçıkar\b", "-"),
        ]
        expr_candidate = s
        for pattern, sym in op_map:
            expr_candidate = re.sub(pattern, f" {sym} ", expr_candidate)

        m = re.search(r"((?:(?:\(?\s*-?\d+\s*|\))\s*[\+\-\*\/]\s*)+\(?\s*-?\d+\s*\)?)", expr_candidate)
        if m:
            rest = expr_candidate[m.start():]
            full_m = re.match(r"[\d\s\+\-\*\/\(\)]+", rest)
            if full_m:
                candidate = full_m.group(0).strip()
                candidate = re.sub(r"[\+\-\*\/\(\s]+$", "", candidate).strip()
                # Tekil ikili işlem mi?
                m_single = re.fullmatch(r"(-?\d+)\s*([\+\-\*\/])\s*(-?\d+)", candidate)
                if m_single:
                    return (int(m_single.group(1)), m_single.group(2), int(m_single.group(3)))
                # Çoklu / bileşik işlem: doğruluğunu safe_eval ile test et
                try:
                    from tools.tool_calc import safe_eval
                    _, err = safe_eval(candidate)
                    if err is None:
                        return candidate
                except Exception:
                    pass

        # 2) Yazıyla Türkçe sayı ifadesi: 'iki yüz kırk yedi artı üç yüz seksen dokuz kaç eder' / 'beş çarpı beş'
        for kw, op in (
            (" artı ", "+"), (" eksi ", "-"), (" ile ", "+" if "fark" not in s else "-"),
            (" çarpı ", "*"), (" kere ", "*"), (" bölü ", "/")
        ):
            if kw in s and ("kaç" in s or "eder" in s or "toplam" in s or "fark" in s or "çarpım" in s or "bölüm" in s or "sonuç" in s or "?" in s):
                left, right = s.split(kw, 1)
                ltoks = [nt for t in re.findall(r"[\wçğıöşü]+", left)
                         if (nt := number_token(t)) is not None]
                rtoks = []
                for t in re.findall(r"[\wçğıöşü]+", right):
                    nt = number_token(t)
                    if nt is not None:
                        rtoks.append(nt)
                    elif rtoks:
                        break
                a = TurkishMorphology.words_to_int(ltoks)
                b = TurkishMorphology.words_to_int(rtoks)
                if a is not None and b is not None:
                    return (a, op, b)
        return None

    #: "Tahmin et: …", "Söyle: …", "Düşün: …" — bunlar SORU DEĞİL, soruyu SARMALAYAN
    #: buyruklardır. Sarmalayıcı soYULMAZSA özne sanılır ("Tahm hakkında bilgim yok").
    #: Yalnız İKİ NOKTA ile biten biçimde soyarız: "bana bir hikâye söyle" bir istektir,
    #: buyruk sanılıp yutulmamalı.
    _META_PREFIX = None

    @classmethod
    def _strip_meta_prefix(cls, s: str) -> str:
        if cls._META_PREFIX is None:
            cls._META_PREFIX = re.compile(
                r"^\s*(?:tahmin\s+et|tahmin|s[oö]yle|d[uü][sş][uü]n|cevapla|"
                r"bir\s+tahmin\s+yap|farz\s+et|diyelim\s+ki)\s*:\s*", re.I)
        return cls._META_PREFIX.sub("", s, count=1)

    def parse(self, raw_text: str) -> SemanticFrame:
        self.turn_count += 1
        # Sarmalayıcı buyruğu soy; kalan cümle neyse o.
        raw_text = self._strip_meta_prefix(str(raw_text or ""))
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

        # 2a) Sayısal sembol bağlama: 'alfa bir demektir' gibi genel bir olgu.
        # Sağ tarafın tamamı sayı olarak çözümlenebiliyorsa özel isim/taksonomi değil,
        # diğer alanların da okuyabileceği paylaşılan NUMERIC_VALUE bilgisidir.
        statement = s.strip().rstrip(".!?")
        numeric_definition = re.fullmatch(
            r"([\wçğıöşü+\-×÷=<>%√²³π∞]{1,32})\s+(.+?)\s+demektir", statement,
        )
        if numeric_definition:
            number_tokens = re.findall(r"[\wçğıöşü]+", numeric_definition.group(2))
            if (number_tokens and all(TurkishMorphology.normalize_num_token(token) is not None
                                      for token in number_tokens)):
                value = TurkishMorphology.words_to_int(number_tokens)
                if value is not None:
                    return SemanticFrame(
                        kind="teach_fact", raw=s_raw, subj=numeric_definition.group(1),
                        rel="NUMERIC_VALUE", obj=str(value),
                    )

        # 2) Matematik / Aritmetik sorgusu
        math_ex = self._extract_math(s)
        if math_ex is not None:
            return SemanticFrame(kind="math_query", raw=s_raw, math_expr=math_ex)

        # 3) Selamlama / Hal-hatır (başka bir soru içermiyorsa)
        if not has_q and words and words[0] in {"merhaba", "selam", "günaydın", "iyi"} and len(words) <= 4:
            return SemanticFrame(kind="greeting", raw=s_raw)
        if set(words) <= {"nasılsın", "merhaba", "selam", "nora", "bugün"} and "nasılsın" in words:
            return SemanticFrame(kind="body_query", raw=s_raw, sub_kind="genel_durum")

        # 3b) '<X> neden olur / neye yol açar / sebep olur?' -> X'in NEDENİ (dünya bilgisi), içebakış değil
        m_cause = re.match(r"^(?:eğer\s+)?(.+?)\s+(?:neden|niçin|niye)\s+olur\??$", s)
        if m_cause:
            tgt = m_cause.group(1).strip()
            tgt = re.sub(r"^(?:bir|bu|şu)\s+", "", tgt)
            self.last_topic = tgt
            return SemanticFrame(kind="wh_query", raw=s_raw, subj=tgt, rel="CAUSES")

        # 4) 'Neden / Niçin': DÜNYA sorusu mu, kendi davranışımın açıklaması mı?
        #    "Neden gökyüzü mavi?" bir dünya sorusudur (içebakış değil); "Neden böyle yaptın?" kendime dair.
        if (
            any(w in {"neden", "niçin", "niye", "sebebi"} for w in words)
            and not any(k in s for k in ("konuş", "cevap ver", "yanıt ver", "sohbet"))
            # Evet/hayır sorusu ('Erozyon kıtlığa neden olur mu?') kendi yolundan gider.
            and not any(w in {"mı", "mi", "mu", "mü", "mıdır", "midir", "mudur", "müdür"} for w in words)
            # 'X kıtlığa neden olur' bir NEDENSELLİK BİLDİRİMİdir ('neden ol-' yüklemi), soru değil.
            and not re.search(r"\b(?:neden|sebep)\s+ol(?:ur|du|uyor|acak)\b|\byol\s+aç", s)
        ):
            _self_words = {"sen", "seni", "sana", "senin", "böyle", "öyle", "davran", "davrandın", "davranıyorsun",
                           "yaptın", "yapıyorsun", "reddettin", "reddediyorsun", "kabul", "ettin", "dedin",
                           "söyledin", "sus", "sustun", "korkuyorsun", "üzülüyorsun"}
            _about_self = any(w in _self_words for w in words)
            _content = [w for w in words if w not in {"neden", "niçin", "niye", "sebebi", "bir", "bu", "şu", "çok", "kadar", "acaba"}]
            if not _about_self and _content:
                _subj = self._clean_np(_content[0])
                self.last_topic = _subj
                return SemanticFrame(kind="wh_query", raw=s_raw, subj=_subj, wh="neden")
            if len(words) <= 3 or _about_self or any(k in s for k in ("reddet", "yapmadın", "kabul etmedin", "erteledin", "dedin")):
                return SemanticFrame(kind="why_query", raw=s_raw)

        # 5) Öz-benlik, bilinç, yaşama ve kimlik sorguları (yalnızca soru veya doğrudan 'kendini tanıt' isteğinde!)
        self_kws = {"kimsin", "nesin", "yaşıyor", "canlı", "bilinç", "bilincin", "zombi", "benliğin", "karakterin", "kişiliğin", "değerlerin", "adın", "ismin"}
        # 'canlı'/'yaşıyor' yalnız muhatap hakkında sorulduğunda öz-sorgudur ('Ali canlı mıdır?' değildir).
        _second = bool({"sen", "senin", "seni", "sana", "kendini", "kendin"} & set(words)) \
            or any(w.endswith(("sın", "sin", "sun", "sün")) for w in words)
        _self_hit = any(w in self_kws and (_second or w not in {"canlı", "yaşıyor"}) for w in words)
        if (has_q and _self_hit) or ("kendini" in words and ("tanıt" in words or "anlat" in words or "ayırırsın" in words)):
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
            # 8a) Ortak istek kipi ('biraz sohbet edelim', 'gidelim', 'konuşalım'): davet.
            _inv = self.morph.analyze_word(last_w, self.graph.lexicon)
            if _inv.get("tense") == "OPT" and _inv.get("is_verb"):
                tgt = ""
                for w in words[:-1]:
                    if w in spatial_postpos or w in {"bana", "sen", "sana", "ben", "biraz", "hadi"}:
                        continue
                    an = self.morph.analyze_word(w, self.graph.lexicon)
                    if an["stem"] not in spatial_postpos and not an.get("is_verb") and an["case"] == "NOM":
                        tgt = an["stem"]
                if tgt:
                    self.last_topic = tgt
                return SemanticFrame(kind="command", raw=s_raw, rel=_inv["stem"], obj=tgt, sub_kind="invite")

            # 8b) Genel emir kipi: yalın fiil köküyle biten cümle ('bana bir hikâye anlat', 'şunu söyle').
            #     Bu bir OLGU değil İSTEKtir; olgu diye öğrenilirse Nora kendi sözünü papağan gibi tekrarlar.
            # 'var'/'yok' VARLIK yüklemidir, emir değildir: 'çok işim var' bir istek değil durumdur.
            if (last_w in self.morph.KNOWN_VERB_ROOTS and last_w not in {"var", "yok", "ol", "et"}
                    and not any(w in words for w in ("ise", "olursa", "eder", "mı", "mi"))):
                tgt = ""
                for w in words[:-1]:
                    if w in spatial_postpos or w in {"kendini", "kendine", "bana", "sen", "sana", "ben"}:
                        continue
                    an = self.morph.analyze_word(w, self.graph.lexicon)
                    if an["stem"] not in spatial_postpos and not an.get("is_verb"):
                        tgt = an["stem"]
                if tgt:
                    self.last_topic = tgt
                return SemanticFrame(kind="command", raw=s_raw, rel=last_w, obj=tgt)

        # 9) Evet/Hayır soruları ('serçe uçar mı?', 'devekuşu uçar mı?', 'kuraklık bitkinliğe neden olur mu?')
        q_particles = {"mı", "mi", "mu", "mü", "mıdır", "midir", "mudur", "müdür"}
        q_idx = next((i for i, w in enumerate(words) if w in q_particles), -1)
        if q_idx >= 2:
            # ÖZNE SEÇİMİ: baştaki BELİRTEÇ/sıfat (sıradaki, bu, her) özne DEĞİLDİR;
            # ilk ADI al. Ve sözlükte doğrulanmamış kökü ad gibi kullanma
            # ('serçe' -> 'serç' değil 'serçe'; 'sıradaki' -> 'sıra' değil).
            _si = 0
            while _si < q_idx - 1 and self._is_determiner(words[_si]):
                _si += 1
            _a0 = self.morph.analyze_word(words[_si], self.graph.lexicon)
            # Kök sözlükte doğrulanmadıysa duyulan biçim kullanılır ('serçe', 'serç' değil).
            subj = words[_si] if _a0.get("unverified_stem") else _a0["stem"]
            # ÖNEMLİ: sözcük listesi DÖNDÜRÜLMEZ (q_idx kayardı); özne yalnız _si'den
            # okunur, yüklem _si+1..q_idx arasıdır.
            pred_tokens = [w for w in words[_si + 1:q_idx] if w != "bir"]
            # 'bitkinliğe / heyelana yol açar mı / neden olur mu / sebep olur mu'
            # ve 'kıtlığa sebebiyet verir mi' (+DAT yöneten nedensel fiiller!)
            if len(pred_tokens) >= 3 and (
                pred_tokens[-2:] in (["yol", "açar"], ["neden", "olur"], ["sebep", "olur"],
                                      ["sebebiyet", "verir"])
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
            # Çok sözcüklü kavram ('yapay zekâ', 'kara delik'): en uzun bilinen öneki özne say.
            # Bilinen önek yoksa Türkçenin baş-sonda kuralı: öbek başı SON sözcüktür ('kuantum bilgisayar').
            _an0 = self.morph.analyze_word(words[0], self.graph.lexicon)
            # Kök sözlükle DOĞRULANMADIYSA duyulan biçim kullanılır: 'Pusula ne işe yarar?'
            # sorusunda 'pusul' (uydurma kök) değil 'pusula' konuşulur.
            subj = words[0] if _an0.get("unverified_stem") else _an0["stem"]
            _found = False
            for n_w in (3, 2):
                if len(words) > n_w:
                    cand = " ".join(words[:n_w])
                    if cand in self.graph.lexicon:
                        subj = cand
                        _found = True
                        break
            if not _found:
                _qmarks = {"nedir", "ne", "nasıl", "nerede", "kim", "neye", "niye", "neden", "niçin",
                           "hakkında", "biliyorsun", "mı", "mi", "mu", "mü", "ise", "bir", "en", "çok", "daha"}
                for w_tok in reversed(words):
                    if w_tok in _qmarks:
                        continue
                    a_tok = self.morph.analyze_word(w_tok, self.graph.lexicon)
                    if a_tok["stem"] in self.graph.lexicon and not a_tok.get("is_verb"):
                        subj = a_tok["stem"]
                        break
            _wh = ""
            for _wq in ("nasıl", "neden", "niye", "niçin", "nerede", "nereye", "nereden", "kim", "hangisi"):
                if _wq in words:
                    _wh = _wq
                    break
            # AĞA DOĞRULANMIŞ AD ÖBEĞİ: "kök çürümesi" -> "kök çürüme". Tek kök aramak
            # bileşik kavramı kaçırıyor ("çürüme hakkında bilgim yok" gibi sahte bilgisizlik).
            _np_cand = self._query_subject_np(words)
            if _np_cand and (_np_cand in self.graph._out_edges or _np_cand in self.graph._in_edges):
                subj = _np_cand
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
                return SemanticFrame(kind="wh_query", raw=s_raw, subj=subj, rel=qtype, wh=_wh)

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

    @staticmethod
    def _strip_nominal_copula(word: str) -> str:
        """Koşacı yalnız son addan sıyır ('omurgalıdır' -> 'omurgalı').

        Bunu fiil çözümleyicisinden ayrı tutmak gerekir: sözlükte yeni bir sıfat
        olduğunda ``analyze_word`` -dIr ekini her zaman koşaç olarak işaretleyemez;
        yine de sözcüğün yüzeyindeki açık ek bilgiyi kaybetmemelidir.
        """
        w = tr_lower(word).strip()
        return w[:-3] if len(w) > 4 and re.search(r"[dt][ıiuü]r$", w) else w

    def _noun_head(self, word: str, *, force_poss: bool = False,
                   force_case: bool = False) -> str:
        """Bir ad öbeğinin başını, kanıtlanmış ek bağlamında kavram gövdesine indir.

        ``kalbe`` gibi sözlük-dışı çekimlerde genel çözümleyici yüzeyi yalın
        bırakabilir. Fiilin yönelme istemesi veya tamlayanın bulunması eki
        bağımsız olarak doğrular; bu durumda b/p, c/ç, d/t yumuşaması da geri
        alınır. Bağlam yoksa kök uydurulmaz.
        """
        w = tr_lower(word).strip()
        an = self._tok(w)
        head = str(an.get("head") or an.get("stem") or w)
        # Tamlayanda -nIn/-In adaylarından sözlükte gerçekten bulunan gövdeyi
        # seç. 'insanın' -> 'insa' değil 'insan'; uzun eki körü körüne seçmek
        # gövdenin son n'sini ek sanabiliyordu.
        if an.get("case") == "GEN":
            known_candidates = []
            for suf in ("nın", "nin", "nun", "nün", "ın", "in", "un", "ün"):
                if len(w) - len(suf) >= 2 and w.endswith(suf):
                    cand = w[:-len(suf)]
                    for option in (cand, self.morph._unsoften(cand)):
                        if option in self.graph.lexicon:
                            known_candidates.append(option)
            if known_candidates:
                return max(known_candidates, key=len)
        if head != w and not force_poss and (
                head in self.graph.lexicon or an.get("plural") or an.get("poss3")
                or (not force_case or len(w) - len(head) >= 2)):
            return head
        suffixes = (("ları", "leri", "sı", "si", "su", "sü", "ı", "i", "u", "ü")
                    if force_poss else
                    ("ndan", "nden", "dan", "den", "tan", "ten", "ına", "ine",
                     "una", "üne", "na", "ne", "ya", "ye", "a", "e", "ı", "i", "u", "ü"))
        for suf in suffixes:
            if len(w) - len(suf) < 2 or not w.endswith(suf):
                continue
            cand = w[:-len(suf)]
            hard = self.morph._unsoften(cand)
            if hard in self.graph.lexicon or hard != cand:
                return hard
            if cand in self.graph.lexicon or force_poss or force_case:
                return cand
        return head

    def _np_from_tokens(self, tokens: list[str], *, head_poss: bool = False,
                        head_case: bool = False) -> str:
        if not tokens:
            return ""
        head = self._noun_head(tokens[-1], force_poss=head_poss, force_case=head_case)
        return " ".join([tr_lower(t) for t in tokens[:-1]] + [head]).strip()

    def _take_leading_subject(self, tokens: list[str]) -> tuple[str, list[str]]:
        """Belirtisiz nesneli SOV cümlede özne ile kalan tümleci ayır.

        Bilinen çok sözcüklü kavramlar önceliklidir ('kara delik ışık yutar').
        Yeni kavramda tek baş sözcük güvenli alt sınırdır; kalan ad öbeği özneye
        yapıştırılmaz ('mitokondri ATP üretir' -> mitokondri | ATP).
        """
        for n in range(min(3, len(tokens) - 1), 1, -1):
            cand = self._np_from_tokens(tokens[:n])
            if cand in self.graph._out_edges or cand in self.graph.lexicon:
                return cand, tokens[n:]
        return self._np_from_tokens(tokens[:1]), tokens[1:]

    def _parse_invented_svo(self, s_raw: str, words: list[str]) -> SemanticFrame | None:
        """Sözlük dışı geniş-zaman yüklemde özne/nesne/yüklem rollerini çıkar.

        Yüzeyin sonundaki -Ar ailesi tek başına anlam kanıtı değildir; en az bir
        nesne/tümleç taşıyan SOV yapısı da gerekir. Böylece çoğul bir ad yüklem
        diye yeni operatöre dönüştürülmez. Kök ve hâl bilgisi öğrenilir, ilişki
        türü hazır ISA/CAN_DO sınıfına zorlanmaz.
        """
        if len(words) < 3:
            return None
        surface_verb = words[-1]
        known = self.morph.analyze_word(surface_verb, self.graph.lexicon)
        if known.get("is_verb") or known.get("copula"):
            return None
        root = GroundedConceptGraph._aor_root(surface_verb)
        if not root or surface_verb in self.graph.lexicon:
            return None
        subject, object_tokens = self._take_leading_subject(words[:-1])
        if not subject or not object_tokens:
            return None
        surface_obj = object_tokens[-1]
        an = self.morph.analyze_word(surface_obj, self.graph.lexicon)
        if (an.get("stem") and an.get("stem") != surface_obj
                and (an.get("case") != "NOM" or an.get("plural"))):
            head = str(an["stem"])
        else:
            head = GroundedConceptGraph._dynamic_noun(surface_obj)
        obj = " ".join(list(object_tokens[:-1]) + [head]).strip()
        if not obj or obj == subject:
            return None
        obj_case = str(an.get("case") or "NOM")
        if obj_case == "NOM" and surface_obj.endswith(("ları", "leri")):
            obj_case = "ACC"
        elif obj_case == "NOM" and surface_obj.endswith(("da", "de", "ta", "te")):
            obj_case = "LOC"
        obj_plural = bool(an.get("plural") or surface_obj.endswith(("lar", "ler", "ları", "leri")))
        return SemanticFrame(
            kind="invent_relation", raw=s_raw, subj=subject, rel="YENI_OPERATOR",
            obj=obj, verb=root, obj_surface=" ".join(object_tokens),
            obj_case=obj_case, obj_plural=obj_plural, sub_kind="SVO",
        )

    def _parse_declarative(self, s_raw: str, s: str, words: list[str]) -> SemanticFrame:
        # Söylemsel dolgu zarflarını ('bence', 'aslında', 'kesinlikle', 'bana göre') özneden ayıkla!
        s = re.sub(r"^(?:bence|aslında|kesinlikle|gerçekten|şüphesiz|bana\s+göre|biliyor\s+musun)\s+", "", s.strip())
        words = re.findall(r"[\wçğıöşü]+", s)
        if len(words) < 2:
            return SemanticFrame(kind="chat", raw=s_raw)

        # EMİR: "bana yardım et", "kapıyı aç", "bunu yap" — öğretilecek olgu değil, istektir.
        _last_w = words[-1]
        _IMP = {"et", "yap", "gel", "git", "bak", "aç", "kapat", "söyle", "anlat", "ver", "al",
                "dur", "başla", "kalk", "otur", "dinle", "oku", "yaz", "çiz", "göster", "bul"}
        if _last_w in _IMP and len(words) >= 2:
            return SemanticFrame(kind="command", raw=s_raw, rel=_last_w, obj=" ".join(words[:-1]))

        # İSTEK: "bana bir zımbırtı uyarla ve sonra tarat" — 'bana' ile kurulan söz
        # dünya olgusu değildir. Son sözcük çekimli fiil/koşaç ya da sözlükteki AD değilse
        # (üretken emir biçimi: 'uyarla', 'tarat'), cümle İSTEKtir. Olgu diye öğrenilirse
        # Nora kullanıcının kendi isteğini papağan gibi tekrarlar (yankı).
        if "bana" in words:
            _la = self.morph.analyze_word(_last_w, self.graph.lexicon)
            if (not _la.get("is_verb") and not _la.get("copula")
                    and _last_w not in self.graph.lexicon and len(_last_w) >= 4):
                _fill = {"bana", "bir", "sonra", "ve", "lütfen", "şimdi", "hemen", "biraz",
                         "ardından", "bunu", "şunu"}
                _tgt = ""
                for _w in words[:-1]:
                    if _w in _fill:
                        continue
                    _an = self.morph.analyze_word(_w, self.graph.lexicon)
                    if not _an.get("is_verb"):
                        _tgt = _an["stem"] if _an.get("unverified_stem") else _w
                if _tgt:
                    self.last_topic = _tgt
                return SemanticFrame(kind="command", raw=s_raw, rel=_last_w, obj=_tgt)

        # KENDİ BEDENİ: "canım sıkılıyor", "karnım ağrıyor", "başım dönüyor".
        # Bu bir dünya olgusu değil, o anki hâldir; "canım sıkılır" diye genellenemez.
        if words[0].endswith(("m", "ım", "im", "um", "üm")) and tr_lower(words[0]) in {
                "canım", "karnım", "başım", "içim", "gözüm", "kolum", "bacağım", "dişim",
                "boğazım", "midem", "sırtım", "elim", "ayağım", "yüreğim", "kalbim"}:
            return SemanticFrame(kind="chat", raw=s_raw, subj=words[0], sub_kind="own_state")

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

        # Taksonomik üyelik: 'X, Y sınıfındandır / takımına aittir'.
        # Rank adı yükleme yapışık da olsa sınıf üyesini 'HAS_PROP sınıf' diye
        # silmek yanlıştır; anlam X ISA Y'dir. Kural Aves'e değil bütün ranklara geneldir.
        ranks = ("sınıf", "familya", "takım", "cins", "tür", "şube", "alem")
        raw_words = re.findall(r"[\wçğıöşü]+", s_raw)
        if len(words) >= 3:
            tail_rank = next((r for r in ranks if words[-1].startswith(r) and words[-1] != r), "")
            tail = words[-1][len(tail_rank):] if tail_rank else ""
            valid_tail = bool(tail_rank and (
                "ndandır" in tail or "ndendir" in tail or tail.startswith(("ına", "ine", "na", "ne"))
                or re.fullmatch(r"[ıiuü][dt][ıiuü]r", tail or "")))
            # 'X bir alet türüdür' üyelik değil, belirtisiz ISA'dır. Rank
            # kuralı burada 'bir'i özneye katarak uydurma düğüm üretir; olağan
            # 'X bir Y-dir' çözümleyicisi 'alet türü' nesnesini kurmalıdır.
            if valid_tail and not (len(words) >= 4 and words[-3] == "bir"):
                subj_t = self._clean_np(" ".join(words[:-2]))
                class_t = self._clean_np(words[-2])
                if subj_t and class_t:
                    self.last_topic = subj_t
                    return SemanticFrame(kind="teach_fact", raw=s_raw, subj=subj_t,
                                         rel="ISA", obj=class_t,
                                         obj_surface=raw_words[-2] if len(raw_words) == len(words) else "")
            if (words[-1] in {"aittir", "mensuptur", "mensubudur"} and len(words) >= 4):
                rank_word = words[-2]
                rank = next((r for r in ranks if rank_word.startswith(r)), "")
                if rank:
                    subj_t = self._clean_np(" ".join(words[:-3]))
                    class_t = self._clean_np(words[-3])
                    if subj_t and class_t:
                        self.last_topic = subj_t
                        return SemanticFrame(kind="teach_fact", raw=s_raw, subj=subj_t,
                                             rel="ISA", obj=class_t)

        # Evrensel niceleyici: 'Bütün kuşlar hayvandır' bir sınıf kapsamasıdır (kuş ISA hayvan).
        # 'bütün' bir iyelik değildir; morfoloji onu 'büt' sahibine çevirip PART_OF uydurmasın.
        # Böylece 'Serçe bir kuştur' + 'Bütün kuşlar hayvandır' zinciri yanıtlanabilir olur.
        if (words[0] in self.UNIVERSAL_QUANT and len(words) == 3
                and re.search(r"[dt][ıiuü]r$", words[-1])
                and words[-1] not in {"vardır", "yoktur"}
                and not self.morph.analyze_word(words[-1], self.graph.lexicon).get("is_verb")):
            cls_t = self._noun_head(words[1])
            pred_t = self._strip_nominal_copula(words[-1])
            if cls_t and pred_t and cls_t != pred_t:
                self.last_topic = cls_t
                return SemanticFrame(kind="teach_fact", raw=s_raw, subj=cls_t,
                                     rel="ISA", obj=pred_t)

        # Nitelenmiş parça özne olduğunda sahibine genelleme yapma:
        # 'Kuşların olgun alyuvarları çekirdeklidir' bütün kuşların çekirdekli
        # olduğu anlamına gelmez. Kapsamlı ad öbeği ayrı düğüm, sahiplik ayrı bağdır.
        first_an = self._tok(words[0])
        if (first_an.get("case") == "GEN" and len(words) >= 3
                and words[0] not in {"benim", "senin", "onun", "bizim", "sizin", "onların"}
                and re.search(r"[dt][ıiuü]r$", words[-1])
                and words[-1] not in {"vardır", "yoktur"}
                and not self.morph.analyze_word(words[-1], self.graph.lexicon).get("is_verb")
                and "değil" not in words and not words[-1].startswith("değil")):
            owner = self._noun_head(words[0])
            part = self._noun_head(words[-2], force_poss=True)
            prop = self._strip_nominal_copula(words[-1])
            scoped = " ".join(words[:-1])
            if owner and part and prop:
                self.last_topic = owner
                return SemanticFrame(kind="teach_fact", raw=s_raw, subj=scoped,
                                     rel="SCOPED_HAS_PROP", obj=prop,
                                     sub_kind=f"{owner}|{part}")

        # Belirtili iyelik + varoluş: 'Ökaryot hücrelerin çekirdeği vardır'.
        # Burada var/yok bir özellik değil, sahibin parçasına işaret eder.
        if words[-1] == "vardır" and len(words) >= 3:
            gen_idx = next((i for i, w in enumerate(words[:-2])
                            if self._tok(w).get("case") == "GEN"), -1)
            if gen_idx >= 0 and gen_idx + 1 < len(words) - 1:
                owner_tokens = words[:gen_idx + 1]
                owner = self._np_from_tokens(owner_tokens)
                if owner not in {"ben", "sen", "o", "biz", "siz", "onlar"}:
                    part = self._np_from_tokens(words[gen_idx + 1:-1], head_poss=True)
                    if owner and part:
                        self.last_topic = owner
                        return SemanticFrame(kind="teach_fact", raw=s_raw, subj=owner,
                                             rel="HAS_PART", obj=part)

        # 'X, Y'ye sahiptir': sahip/ait gibi yardımcı adı özellik diye kaydetme.
        if words[-1] == "sahiptir" and len(words) >= 3:
            subj_t, obj_tokens = self._take_leading_subject(words[:-1])
            obj_t = self._np_from_tokens(obj_tokens, head_case=True)
            if subj_t and obj_t:
                self.last_topic = subj_t
                return SemanticFrame(kind="teach_fact", raw=s_raw, subj=subj_t,
                                     rel="HAS_PART", obj=obj_t)

        # 'X, Y'den oluşur': yön bütünden parçaya gider; CAN_DO 'oluş' değildir.
        if words[-1] in {"oluşur", "oluşmaktadır"} and len(words) >= 3:
            subj_t, obj_tokens = self._take_leading_subject(words[:-1])
            if obj_tokens:
                obj_t = self._np_from_tokens(obj_tokens)
                if subj_t and obj_t:
                    self.last_topic = subj_t
                    return SemanticFrame(kind="teach_fact", raw=s_raw, subj=subj_t,
                                         rel="COMPOSED_OF", obj=obj_t)

        # Yerli varoluş: 'Ökaryot hücrelerde mitokondri bulunur'. Yer bildiren
        # öbek cümlenin öznesi değildir; bulunan şey yüklem öncesindeki addır.
        if words[-1] == "bulunur" and len(words) >= 3:
            loc_idx = next((i for i, w in enumerate(words[:-1])
                            if self._tok(w).get("case") == "LOC"), -1)
            if 0 <= loc_idx < len(words) - 2:
                container = self._np_from_tokens(words[:loc_idx + 1])
                entity = self._np_from_tokens(words[loc_idx + 1:-1])
                if container and entity:
                    self.last_topic = entity
                    return SemanticFrame(kind="teach_fact", raw=s_raw, subj=entity,
                                         rel="LOCATED_AT", obj=container)

        # Hafif fiil yapılarında yer/amaç tümleci kaybolmaz:
        # 'Mitokondri hücrenin enerji üretiminde görev yapar'.
        if len(words) >= 4 and words[-2:] in (["görev", "yapar"], ["rol", "oynar"]):
            subj_t, adjunct = self._take_leading_subject(words[:-2])
            last_an0 = self.morph.analyze_word(words[-1], self.graph.lexicon)
            if subj_t and adjunct and last_an0.get("is_verb"):
                obj_t = " ".join(adjunct + [words[-2], last_an0["stem"]])
                self.last_topic = subj_t
                return SemanticFrame(kind="teach_fact", raw=s_raw, subj=subj_t,
                                     rel="CAN_DO", obj=obj_t)

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

        # "...-den olur": kaynak/neden yapısı ("kök çürümesi su birikmesinden olur").
        # Bu, "X olur" (varoluş) değil "X, Y'den olur" (neden) demektir; özne/nesne
        # ad öbeğiyle ayrılır, yoksa "kök" + "çürümesi su birikmesinden" gibi bölünür.
        if words and words[-1] == "olur" and len(words) >= 3:
            sj, ob = self.split_svo(words[:-1])
            if ob is not None and sj is not None and ob.get("case") in ("ABL", "INS"):
                # YÖN: "X, Y'den olur" demek Y'nin X'in NEDENİ olduğu demektir.
                # Çerçeve (özne, nesne) sırasını yüzeyden alırsa kenar ters kurulur:
                # "kök çürümesi su birikmesinden olur" -> (kök çürüme)CAUSES(su birikme)
                # YANLIŞ. Nedensel kenarın yönü daima NEDEN -> ETKİ'dir.
                self.last_topic = sj["canonical"]
                return SemanticFrame(kind="teach_fact", raw=s_raw, subj=ob["canonical"],
                                     rel="CAUSES", obj=sj["canonical"])

        # '... için kullanılır' / '... işe yarar' / '... yol gösterir'
        if "için" in rest_words and ("kullanılır" in rest_words or "yarar" in rest_words):
            idx_ic = rest_words.index("için")
            purpose = " ".join(rest_words[:idx_ic])
            return SemanticFrame(kind="teach_fact", raw=s_raw, subj=subj, rel="USED_FOR", obj=purpose)
        if "yol" in rest_words and "gösterir" in rest_words:
            return SemanticFrame(kind="teach_fact", raw=s_raw, subj=subj, rel="USED_FOR", obj=" ".join(rest_words[:-2] + ["yol gösterme"]).strip())

        # 'X, Y'ye sebebiyet verir' — bilinen nedensel yapının DAT nesneli
        # eşdeğeri. Bilinmeyen ilişkiyi ISA'ya zorlamaz; CAUSES olarak tek-atış
        # öğrenilir ve aynı Graph/HDC zincirine katılır.
        if "sebebiyet" in rest_words and "verir" in rest_words:
            cue = next((i for i, w in enumerate(words) if w == "sebebiyet"), -1)
            if cue >= 1:
                cause_obj = self._strip_governed_dative(words[cue - 1])
                if cue >= 2:
                    cause_subj = self._np_name(words[:cue - 1], self._tok(words[cue - 2])) or subj
                else:
                    cause_subj = subj
                if cause_subj and cause_obj:
                    self.last_topic = cause_subj
                    return SemanticFrame(kind="teach_fact", raw=s_raw, subj=cause_subj,
                                         rel="CAUSES", obj=cause_obj)

        # '... yol açar' / '... neden olur' / '... sebep olur' (+DAT durum eki yönetir: 'erozyona' -> 'erozyon')
        if ("yol" in rest_words and "açar" in rest_words) or ("neden" in rest_words and "olur" in rest_words) or ("sebep" in rest_words and "olur" in rest_words):
            # NEDENSELLİK YAPISI kendi içinde çözülür: nesne, "yol/neden/sebep" sözünden
            # ÖNCEKİ addır; özne ondan önceki TÜM addır.
            #   "aşırı sulama tuzlanmaya yol açar" -> özne "aşırı sulama", nesne "tuzlanma"
            # Yukarıda hesaplanan `subj` burada YETMEZ (tek sözcüğe kırpılmış olabilir).
            cue = next((i for i, w in enumerate(words) if w in ("yol", "neden", "sebep")), -1)
            if cue > 0:
                sj, ob = self.split_svo(words[:cue])
                if ob is not None:
                    cause_obj = ob.get("canonical")
                    cause_subj = (sj or {}).get("canonical") or subj
                elif cue >= 2:
                    # Hâl eki okunamadı ("göçe" yalın sanıldı) ama YAPI belli:
                    # nedensellikte son ad NESNE, öncesi ÖZNE. Ek sıyrılır, uydurulmaz.
                    cause_obj = self._strip_governed_dative(words[cue - 1])
                    cause_subj = self._np_name(words[: cue - 1], self._tok(words[cue - 2])) or subj
                else:
                    cause_obj = self._strip_governed_dative(words[cue - 1])
                    cause_subj = subj
            else:
                cause_subj = subj
                cause_obj = self._strip_governed_dative(rest_words[0]) if rest_words else ""
            if cause_subj:
                self.last_topic = cause_subj
            return SemanticFrame(kind="teach_fact", raw=s_raw, subj=cause_subj, rel="CAUSES", obj=cause_obj)

        # "X, Y'nin zıttıdır" / "X, Y'nin tam tersidir" -> OPPOSITE  (tamlayan + zıt adı + koşaç)
        for opp_word in ("zıttı", "zıddı", "tersi", "karşıtı"):
            if rest_words and rest_words[-1].startswith(opp_word):
                gen_part = " ".join(rest_words[:-1])
                if gen_part:
                    gen_an = self.morph.analyze_word(gen_part.split()[-1], self.graph.lexicon)
                    tail = rest_words[-1][len(opp_word):]
                    obj = gen_an["stem"] if gen_an["case"] == "GEN" or gen_an["stem"] != gen_part.split()[-1] else gen_part
                    neg = "değil" in tail
                    return SemanticFrame(kind="teach_fact", raw=s_raw, subj=subj, rel="OPPOSITE", obj=obj,
                                         valence=-1.0 if neg else 0.0)

        # '...-da/-de yaşar / bulunur' -> LOCATED_AT
        if rest_words[-1] in {"yaşar", "bulunur", "yetişir"} and len(rest_words) >= 2:
            loc_an = self.morph.analyze_word(rest_words[0], self.graph.lexicon)
            return SemanticFrame(kind="teach_fact", raw=s_raw, subj=subj, rel="LOCATED_AT", obj=loc_an["stem"])

        # Çok sözcüklü koşaçlı yüklem: yeni sıfatlarda -dIr biçimbirimini sözlük
        # tanımıyor diye özneye sıfatı yapıştırma. Tamlayanlı ad yüklem ISA,
        # niteleme yüklemi HAS_PROP olarak korunur.
        _copula_an = self.morph.analyze_word(rest_words[-1], self.graph.lexicon)
        if (re.search(r"[dt][ıiuü]r$", rest_words[-1]) and not _copula_an.get("is_verb")
                and "değil" not in rest_words and not rest_words[-1].startswith("değil")
                and "bir" not in rest_words and "ve" not in rest_words):
            pred_tokens = list(rest_words)
            pred_tokens[-1] = self._strip_nominal_copula(pred_tokens[-1])
            pred = " ".join(w for w in pred_tokens if w)
            if pred:
                has_gen_pred = any(self._tok(w).get("case") == "GEN" for w in pred_tokens)
                return SemanticFrame(kind="teach_fact", raw=s_raw, subj=subj,
                                     rel="ISA" if has_gen_pred else "HAS_PROP", obj=pred)

        # Bilinmeyen yüklem: nominal özellik/ISA diye yazmadan önce SOV
        # sözdizimiyle yeni ilişki adayını ayır.
        invented = self._parse_invented_svo(s_raw, words)
        if invented is not None:
            self.last_topic = invented.subj
            return invented

        # Fiil mi koşaçlı sıfat mı? ('kuşlar uçar', 'penguen uçmaz' vs 'ateş sıcaktır')
        last_an = dict(self.morph.analyze_word(rest_words[-1], self.graph.lexicon))
        if not last_an.get("is_verb") and not last_an.get("copula"):
            # Sözlükte olmayan çekimli fiil: "-ar/-er/-ır..." eki + tanınan kök.
            _w = rest_words[-1]
            _roots = getattr(self.morph, "KNOWN_VERB_ROOTS", set())
            for _k in (1, 2, 3):
                if len(_w) - _k >= 2 and _w[:-_k] in _roots:
                    last_an = dict(last_an, is_verb=True, stem=_w[:-_k], tense="AOR", person="3SG",
                                   neg=False)
                    break
        if last_an["copula"] or (not last_an["is_verb"] and len(rest_words) == 1):
            prop = last_an["stem"]
            neg = "değil" in rest_words or "değildir" in rest_words
            if neg and len(rest_words) >= 2:
                prop = self.morph.analyze_word(rest_words[-2], self.graph.lexicon)["stem"]
            return SemanticFrame(kind="teach_fact", raw=s_raw, subj=subj, rel="HAS_PROP", obj=prop, valence=-1.0 if neg else 0.0)

        # Sıfat yüklemi: 'matematik ödevim çok zor', 'bugün hava sıcak' (son sözcük fiil değil)
        if not last_an["is_verb"] and not last_an["copula"]:
            _advs = {"çok", "pek", "fena", "gerçekten", "biraz", "oldukça", "çokça", "hayli"}
            _sw = words[:-1]
            while len(_sw) > 1 and _sw[-1] in _advs:
                _sw = _sw[:-1]
            subj_adj = self._clean_np(" ".join(_sw)) if _sw else subj
            if subj_adj:
                self.last_topic = subj_adj
                return SemanticFrame(kind="teach_fact", raw=s_raw, subj=subj_adj, rel="HAS_PROP",
                                     obj=rest_words[-1])

        # Eylem: 'kuşlar uçar', 'penguen uçmaz', 'balıklar suda yüzer',
        #        'kara delik ışığı yutar' (çok sözcüklü özne + hâl ekli nesne)
        rel_type = "CANNOT_DO" if last_an["neg"] else "CAN_DO"
        _pre_verb = list(words[:-1])
        _sj, _ob = self.split_svo(_pre_verb)
        _subj = (_sj or {}).get("canonical") or subj
        # Nesne yalnız ACC/DAT ise fiilin nesnesidir; LOC/ABL/INS zarf tümlecidir
        # ("balıklar suda yüzer" -> eylem "yüz", yer "su"). Böylece özne, nesne
        # sanılıp yutulmaz. Belirtisiz nesnede hâl eki yoktur ('ATP üretir',
        # 'sert kabuklu yumurta bırakır'); o zaman ilk kavram özne, kalan öbek
        # yüklem tümlecidir. Bütün öbeği özneye katmak bilgiyi başka düğüme yazar.
        _indefinite = ""
        if _ob is None and len(_pre_verb) >= 2:
            _lead, _remainder = self._take_leading_subject(_pre_verb)
            if _lead and _remainder:
                _subj = _lead
                _indefinite = self._np_from_tokens(_remainder)
        if _ob is not None and _ob.get("canonical") and _ob.get("case") in ("ACC", "DAT"):
            act_obj = f"{_ob['canonical']} {last_an['stem']}".strip()
        elif _indefinite:
            act_obj = f"{_indefinite} {last_an['stem']}".strip()
        else:
            act_obj = last_an["stem"]
        if _subj:
            self.last_topic = _subj
        return SemanticFrame(kind="teach_fact", raw=s_raw, subj=_subj, rel=rel_type, obj=act_obj)

    #: ÖZNE OLMAYAN BAŞ SÖZCÜKLER: belirteçler ve "sıradaki/bir sonraki" gibi ortacıllar.
    DETERMINERS = frozenset((
        "bu", "şu", "o", "her", "bir", "bazı", "tüm", "bütün", "hiç", "çok", "az",
        "sıradaki", "önceki", "sonraki", "gelecek", "geçen", "ilk", "son",
    ))

    def _is_determiner(self, w: str) -> bool:
        return w in self.DETERMINERS or w in self.ADJ_SEED

    def _strip_governed_dative(self, word: str) -> str:
        """Yönelme durumu (+DAT) yöneten fiillerin ('yol aç-', 'neden ol-', 'sebep ol-') tümlecinden -ya/-ye/-a/-e ekini söker."""
        w = tr_lower(word).strip()
        if " " in w:
            # Çok sözcüklü öbek: eki YALNIZ baştan sıyır (öbeğin tamamına uygulanmaz).
            parts = w.split()
            return " ".join(parts[:-1] + [self._strip_governed_dative(parts[-1])])
        if len(w) > 4 and w.endswith(("ma", "me")):
            # "tuzlanma", "sulama": son ünlü hâl eki değil, türetim eki.
            return w
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
        # "X ... ve Y ..." birleşik YÜKLEM cümlesi iki ayrı iddiadır: her iki yarı da kendi başına
        # yüklem taşıyorsa bölünür. Böylece bütün bir yan cümle, ilişki nesnesi olarak kaydedilmez
        # ('sadıktır ve insana keyif ver' gibi). 'Köpek ve kedi evcildir' ise bölünmez (özne listesi).
        _expanded: list[str] = []
        for cl in clauses:
            parts = re.split(r"\s+ve\s+", cl.strip())
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
                # Sağ yarı öznesiz kaldıysa ('... ve insana keyif verir') sol yarının öznesi devredilir.
                _subj = getattr(_l, "subj", "") or ""
                if _subj and not getattr(_r, "subj", ""):
                    _cand = f"{_subj} {parts[1].strip()}"
                    try:
                        _c2 = self.wernicke.parse(_cand)
                    except Exception:
                        _c2 = None
                    if _c2 is not None and _c2.kind == "teach_fact" and getattr(_c2, "obj", ""):
                        _expanded.extend([parts[0].strip(), _cand])
                        continue
            # VİRGÜL: "Zorvath bir alet türüdür, kayaları keser." iki ayrı iddiadır.
            # Ama "elma, armut ve muz meyvedir" bir listedir: bölmek yanlış olur. Ölçüt
            # biçim değil ÇÖZÜMLEME: iki yarı da kendi yüklemini taşıyorsa bölünür.
            if "," in cl:
                _l2, _r2 = (x.strip() for x in cl.split(",", 1))
                try:
                    _pl = self.wernicke.parse(_l2)
                    _pr = self.wernicke.parse(_r2)
                    _ok2 = (_pl.kind == "teach_fact"
                            and _pr.kind in {"teach_fact", "invent_relation"}
                            and getattr(_pl, "obj", "") and getattr(_pr, "obj", ""))
                except Exception:
                    _ok2 = False
                if _ok2:
                    # "X bir Y'dir, kayaları keser"de sağ yarım yüklem
                    # kendi öznesini düşürmüştür. Nesne hâli/çokluk-belirtme
                    # kanıtı varsa özne soldan devralınır; "kaya" özne yapılmaz.
                    _rw = re.findall(r"[\wçğıöşü]+", tr_lower(_r2))
                    _an_r = self.wernicke._tok(_rw[0]) if _rw else {}
                    _object_marked = bool(
                        _an_r.get("case") in self.wernicke.OBLIQUE
                        or (_rw and _rw[0].endswith(("ları", "leri")))
                    )
                    if _object_marked and getattr(_pl, "subj", ""):
                        _candidate = f"{_pl.subj} {_r2}"
                        try:
                            _pc = self.wernicke.parse(_candidate)
                        except Exception:
                            _pc = None
                        if (_pc is not None and _pc.kind in {"teach_fact", "invent_relation"}
                                and _pc.subj == _pl.subj and getattr(_pc, "obj", "")):
                            _r2 = _candidate
                    _expanded.extend([_l2, _r2])
                    continue
            _expanded.append(cl)
        clauses = _expanded
        learned: list[RelationEdge] = []
        for cl in clauses:
            cl = cl.strip()
            if not cl:
                continue
            # Bilimsel kısaltmaların duyulan yüzeyini koru (ATP, DNA, RNA ...).
            # Cümle başındaki yalnız ilk harfi büyük Türkçe sözcükler buna girmez.
            for m in re.finditer(r"[\wçğıöşü]+", cl):
                token = m.group(0)
                if len(token) >= 2 and token.isupper() and any(ch.isalpha() for ch in token):
                    self.graph.register_display_name(token, token)
                    continue
                # Özel ad yüzeyi: ekli ad 'Elif'ten', 'Ayşe'den' biçiminde kesme işaretiyle yazılır;
                # ortada büyük harfle başlayan sözcük de (cümle başı dışında) özel addır.
                _ekli = cl[m.end():m.end() + 1] in ("'", "’")
                if token[:1].isupper() and (_ekli or m.start() > 0):
                    self.graph.register_display_name(token.lower(), token)
            fr = self.wernicke.parse(cl)
            # Zamir öznenin ('benim', 'sen', 'o' ...) göndergesi cümlenin dışında, konuşanda durur.
            # Kavram ağına 'benim ISA ...' diye yazılırsa okunan her metnin yazarı Nora'nın
            # kendisi olur ve sonraki her 'benim' sözünde o yabancı hayat anlatılır.
            if self.is_deictic(getattr(fr, "subj", "")):
                continue
            if fr.kind == "invent_relation":
                invented = self.graph.invent_relation(
                    fr.subj, fr.obj, cl, fr.verb,
                    obj_surface=fr.obj_surface, obj_case=fr.obj_case,
                    obj_plural=fr.obj_plural,
                )
                if invented.get("edge") is not None:
                    learned.append(invented["edge"])
            elif fr.kind == "teach_fact":
                if getattr(fr, "obj_surface", ""):
                    self.graph.register_display_name(fr.obj, fr.obj_surface)
                if not self._teachable_predicate(fr.obj):
                    continue
                if fr.rel == "SCOPED_HAS_PROP":
                    owner, _, part = str(fr.sub_kind).partition("|")
                    if not owner or not part:
                        continue
                    learned.append(self.graph.add_edge(fr.subj, "HAS_PROP", fr.obj, source="duyum", speaker=speaker))
                    learned.append(self.graph.add_edge(fr.subj, "PART_OF", owner, source="duyum", speaker=speaker))
                    learned.append(self.graph.add_edge(owner, "HAS_PART", part, source="duyum", speaker=speaker))
                elif fr.rel == "ADJ_ISA":
                    adj, noun = fr.obj.split("|", 1)
                    learned.append(self.graph.add_edge(fr.subj, "ISA", f"{adj} {noun}", source="duyum", speaker=speaker))
                    learned.append(self.graph.add_edge(f"{adj} {noun}", "ISA", noun, source="duyum", speaker=speaker))
                    learned.append(self.graph.add_edge(fr.subj, "HAS_PROP", adj, source="duyum", speaker=speaker))
                else:
                    learned.append(self.graph.add_edge(fr.subj, fr.rel, fr.obj, source="duyum", speaker=speaker))
                    # "X bir Y türüdür" ifadesinde tür kavramı ile üst kavram
                    # arasındaki genel köprü de öğrenilir. Böylece yüzeydeki
                    # "türü" sözcüğü ISA zincirini koparmaz; kural bütün tür
                    # adlarına uygulanır, belirli bir hedefe özel değildir.
                    if fr.rel == "ISA" and fr.obj.endswith((" türü", " tür")):
                        parent = fr.obj.rsplit(" ", 1)[0].strip()
                        if parent and parent != fr.subj:
                            learned.append(self.graph.add_edge(
                                fr.obj, "ISA", parent, source="duyum", speaker=speaker,
                            ))
            elif fr.kind == "teach_val":
                learned.append(self.graph.add_edge(fr.subj, fr.rel, fr.obj, source="duyum", speaker=speaker))
        return learned

    @staticmethod
    def is_deictic(subj: str) -> bool:
        """Özne bir kişi/gösterme zamiri mi (her hâl ekiyle)? Göndergesi bağlamdadır, dünyada değil."""
        from nora_broca import TurkishSyntaxEngine
        head = tr_lower(str(subj or "")).split()
        if not head:
            return False
        forms = {form for cases in TurkishSyntaxEngine.PRONOUN_CASES.values() for form in cases.values()}
        return head[0] in forms

    def _teachable_predicate(self, obj: str) -> bool:
        """Yüklem gerçekten tek bir özellik mi? 'organdır ve düşünmeyi sağlar' gibi bütün bir
        yan cümle ad yerine geçmişse kavram ağına YAZILMAZ (bozuk düğüm = bozuk dünya bilgisi)."""
        words = obj.split()
        if len(words) > 6:
            return False
        if any(w in {"ve", "ama", "ancak", "fakat", "çünkü", "ki"} for w in words):
            return False
        # Bütün bir yan cümle ad yerine geçmişse (çekimli yüklem eki taşıyorsa) kavram ağına yazılmaz.
        _finite_tails = ("dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür",
                         "mış", "miş", "muş", "müş", "yor", "yım", "yim", "yız", "yiz",
                         "sın", "sin", "sun", "sün")
        return not any(w.endswith(_finite_tails) for w in words)


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


# ============================================================================
# TÜRKÇENİN EKSİKSİZ ÇEKİM/TÜRETİM ÇEKİRDEĞİ (birleşik zaman, sıfat-fiil, yapım eki,
# çatı çözümleme, büyük/sıra/ondalık sayı) — sesbilim tek kaynaktan beslenir.
# ============================================================================
from nora_turkce import attach as _attach_turkish  # noqa: E402
from nora_turkce import COMPOUND_MAP  # noqa: E402
_attach_turkish(TurkishMorphology)

# Sentez indeksi hazır kurulur: ilk konuşma turu indeks kurma maliyetini ödemez.
TurkishMorphology._ensure_verb_index()

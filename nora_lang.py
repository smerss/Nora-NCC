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
        info = cls._analyze_word_raw(raw, lexicon)
        known = cls._known_vocab(lexicon)
        if not known:
            return info
        w, stem = info.get("surface", ""), info.get("stem", "")
        if w in known and (stem not in known or info.get("is_verb")) and not info.get("vowel_drop") and not info.get("poss"):
            # Sözlükte bütün hâliyle duran sözcük, kök soyma tahmininden üstündür:
            # 'yemek' (aş) 'ye+mek' diye fiil köküne indirgenmemeli; 'hikâye' 'hikâ'+DAT olmamalı.
            return {**info, "stem": w, "case": "NOM", "is_verb": False, "neg": False}
        if (w not in known and stem not in known and info.get("is_verb")
                and not info.get("voice") and stem not in cls.KNOWN_VERB_ROOTS):
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
        self._in_edges: dict[str, list[RelationEdge]] = {}
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
        self._in_edges.clear()
        self._edge_map.clear()

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
        self._in_edges.setdefault(o, []).append(edge)   # ters arama: "dünyanın en yüksek dağı" -> everest
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
        for e, isa_path in self.query(s, ("ISA", "HAS_PROP", "HAS_PART", "CAN_DO", "CAUSES", "USED_FOR", "LOCATED_AT")):
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

    def _copula_phrase(self, phrase: str) -> str:
        """Çok sözcüklü yüklem öbeğinde koşaç YALNIZCA son sözcüğe gelir ('çok sıcaktır',
        'bir değerli metaldir'); tam dizgeye uygulanınca 'sıcakır' gibi bozuk biçimler çıkıyordu."""
        parts = phrase.split()
        if not parts:
            return phrase
        # Yüklem zaten çekimli bir fiilse koşaç EKLENMEZ: 'kalp sağlığına iyi gelirdir' değil
        # 'iyi gelir'. (Geniş zaman 3. tekil fiil + -dIr = çifte çekim)
        last_an = self.morph.analyze_word(parts[-1])
        if last_an.get("is_verb") and str(last_an.get("person", "")) in ("3SG", "3PL"):
            return phrase
        return " ".join(parts[:-1] + [self.morph.inflect_noun(parts[-1], "NOM", copula=True)])

    def _gen_phrase(self, phrase: str) -> str:
        """Çok sözcüklü öznede tamlayan eki son sözcüğe ('yer kabuğunun', 'basınç farkının')."""
        parts = phrase.split()
        if not parts:
            return phrase
        return " ".join(parts[:-1] + [self.morph.inflect_noun(parts[-1], "GEN")])

    def _case_phrase(self, phrase: str, case: str, poss: int = 0) -> str:
        """Çok sözcüklü öbekte hâl eki YALNIZCA son sözcüğe gelir ('oksijen ve besin üretimine')."""
        parts = phrase.split()
        if not parts:
            return phrase
        head = parts[-1]
        tail = (self.morph.inflect_noun(head, case, poss3=(poss == 3)) if poss in (0, 3)
                else self.morph.inflect_noun(head, case))
        return " ".join(parts[:-1] + [tail])

    def realize_prop(self, p: Proposition) -> str:
        s = p.subj.strip()
        o = p.obj.strip()
        r = p.rel

        if r == "ISA":
            art = "" if o.startswith("bir ") else "bir "
            pred = self._copula_phrase(art + o)
            return f"{s} {art}{o} değildir" if p.neg else f"{s} {pred}"

        if r == "HAS_PROP":
            if p.neg:
                return f"{s} {o} değildir"
            return f"{s} {self._copula_phrase(o)}"

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
                return f"{s} {self.morph.inflect_noun(s, 'GEN')} {part} yok"
            return f"{self.morph.inflect_noun(s, 'GEN')} {part} var"

        if r in ("CAN_DO", "CANNOT_DO"):
            is_neg = p.neg or (r == "CANNOT_DO")
            parts = o.split()
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
            # Çok sözcüklü ve iyelikli öbeklerde yönelme eki güvenilir değil ('basınç farkıya' gibi).
            # Doğal ve her öbekte doğru kalan kuruluş: "X'in nedeni Y'dir."
            if o.endswith(("ı", "i", "u", "ü")) or len(o.split()) > 1:
                gen = self._gen_phrase(s)
                return f"{gen} nedeni {self._copula_phrase(o)}" if not p.neg else f"{gen} nedeni {o} değildir"
            if p.neg:
                return f"{s} {self._case_phrase(o, 'DAT')} yol açmaz"
            return f"{s} {self._case_phrase(o, 'DAT')} yol açar"

        if r == "USED_FOR":
            return f"{s} {o} için kullanılır"

        if r == "LOCATED_AT":
            if o in PROPER_NOUNS:
                # Özel ad kesme işaretiyle çekimlenir: "Himalaya'da", "Antalya'da".
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
    wh: str = ""              # soru sözcüğü: nasıl | neden | nerede | kim | ne ...


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
            # Çok sözcüklü kavram ('yapay zekâ', 'kara delik'): en uzun bilinen öneki özne say.
            # Bilinen önek yoksa Türkçenin baş-sonda kuralı: öbek başı SON sözcüktür ('kuantum bilgisayar').
            subj = self.morph.analyze_word(words[0], self.graph.lexicon)["stem"]
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

        # Fiil mi koşaçlı sıfat mı? ('kuşlar uçar', 'penguen uçmaz' vs 'ateş sıcaktır')
        last_an = self.morph.analyze_word(rest_words[-1], self.graph.lexicon)
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
            _expanded.append(cl)
        clauses = _expanded
        learned: list[RelationEdge] = []
        for cl in clauses:
            cl = cl.strip()
            if not cl:
                continue
            fr = self.wernicke.parse(cl)
            if fr.kind == "teach_fact":
                if not self._teachable_predicate(fr.obj):
                    continue
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

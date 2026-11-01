# -*- coding: utf-8 -*-
"""Çok cümleli kelime problemi çözücüsü (kural tabanlı, sembolik).

Yalnızca açıkça tanınan yapıyı çözer: bir başlangıç miktarı, ardından her cümlede tek bir
eklenme ya da eksilme olayı, en sonda 'kaç kaldı' ya da 'toplam kaç' sorusu. Tanınmayan bir
cümle ya da belirsiz bir sayı varsa None döner; Nora'nın olağan yanıt yollarına geçer.

Doğruluk sınırı: bu bir anlam çözümü değildir; fiil sınıflarına (ekleme/eksiltme) ve cümle
sırasına dayanır. Karşılaştırma ('kaç fazla') ve bölme soruları burada çözülmez.
"""

from __future__ import annotations

import random
import re

from nora.cognition.puzzles import ADD_WORDS, _is_removal, _numbers, _words
from nora.language.lang import tr_lower

# Geçmiş zaman eksiltme fiilleri: çekim ekleri değişken olduğu için tam sözcük listesi.
REMOVE_PAST = {"yedi", "yedim", "yedin", "yiyor", "yedik", "verdi", "verdim", "verdik", "attı", "attım",
               "kaybetti", "kaybettim", "sattı", "sattım", "harcadı", "harcadım", "kullandı", "kullandım",
               "gitti", "gittim", "kırdı", "sildi", "çıkardı", "çıkardım", "eksildi", "uçtu", "uçup",
               "dağıttı", "sunmadı", "giderken", "gitti", "indi", "çıktı", "bozuldu", "kırıldı",
               "verildi", "çıkarıldı", "satıldı", "kullanıldı", "yenildi", "harcandı", "kaybedildi"}
# Ekleme eylemleri (edilgen ve etken): sözlük ADD_WORDS'e ek olarak.
ADD_EXTRA = {"koydu", "ekledi", "koyuldu", "eklendi", "getirdi"}
LEFT_WORDS = {"kaldı", "kalır", "kalan", "geriye", "kalacak", "kalmış", "kalır mı", "kalsın"}
TOTAL_WORDS = {"toplam", "toplamda", "hepsi", "birlikte", "altogether"}


def _digits(sentence: str) -> list[int]:
    """Yalnız rakamlı sayılar: 'yedi' fiili ile 'yedi' sayısı karışmasın diye sözcük sayıları alınmaz."""
    return [int(x) for x in re.findall(r"\d+", sentence or "")]


def _sentences(raw: str) -> list[str]:
    s = re.sub(r"^[\s>]*\d+\s*[.)]\s+", "", (raw or "").strip())
    parts = re.split(r"(?<=[.!?])\s+", s)
    return [p.strip() for p in parts if p.strip()]


def _classify(words: list[str]) -> str:
    """'remove', 'add' ya da 'neutral' (olay yok)."""
    if "ona" in words and any(w in ("verdi", "veriyor", "verirse") for w in words):
        return "add"
    if "ondan" in words and any(w in ("aldı", "alırsa", "alıyor") for w in words):
        return "remove"
    if any(_is_removal(w) or w in REMOVE_PAST for w in words):
        return "remove"
    if any(w in ADD_WORDS or w in ADD_EXTRA for w in words):
        return "add"
    return "neutral"


_POSS = r"(?:nin|nın|nun|nün|in|ın|un|ün)"
_NAME_COUNT = re.compile(r"([\wçğıöşü]+)'" + _POSS + r"\s+(?:(?:ise|de|da)\s+)?(\d+)")
_COMPARE_Q = re.compile(
    r"([\wçğıöşü]+),\s*([\wçğıöşü]+)'(?:den|dan|ten|tan)\s+(?:ne\s+kadar\s+)?(?:kaç\s+)?"
    r"[\wçğıöşü]+\s+(?:daha\s+)?(fazla|çok|eksik|az)")
_LEAD_COUNT = re.compile(r"^\s*([\wçğıöşü]+)\s+(?:ise\s+)?(\d+)\s+[\wçğıöşü]+")


def _solve_compare(sentences: list[str]) -> tuple[int, str] | None:
    """'A'nın x var. B'nin y var. A, B'den kaç fazla?' → x - y (x > y ise)."""
    question = tr_lower(sentences[-1])
    match = _COMPARE_Q.search(question)
    if not match:
        return None
    a, b, polarity = match.group(1), match.group(2), match.group(3)
    counts: dict[str, int] = {}
    for sentence in sentences[:-1]:
        low = tr_lower(sentence)
        for name, value in _NAME_COUNT.findall(low):
            counts[name] = int(value)
        lead = _LEAD_COUNT.match(low)          # "ali 5 kart topladı." → ali: 5
        if lead:
            counts[lead.group(1)] = int(lead.group(2))
    if a not in counts or b not in counts:
        return None
    # 'fazla/çok': a daha çok olmalı; 'eksik/az': a daha az olmalı (b'nin eksiği).
    if polarity in {"fazla", "çok"} and counts[a] <= counts[b]:
        return None
    if polarity in {"eksik", "az"} and counts[a] >= counts[b]:
        return None
    diff = abs(counts[a] - counts[b])
    return diff, f"{a} {counts[a]}, {b} {counts[b]}; fark {diff}"


def _solve_divide(sentences: list[str]) -> tuple[int, str] | None:
    """'N tane var. Hepsini eşit olarak K kutuya koydu. Her kutuda kaç?' → N / K (tam bölünürse)."""
    question = tr_lower(sentences[-1])
    if "her" not in question or "kaç" not in question:
        return None
    body = sentences[:-1]
    share = [s for s in body if ("eşit" in tr_lower(s) or "paylaştır" in tr_lower(s)) and _digits(s)]
    if len(share) == 1 and len(_digits(share[0])) == 2 and share[0] == body[0]:
        total, parts = _digits(share[0])
        if parts == 0 or total % parts != 0:
            return None
        return total // parts, f"{total} {parts} kişiye eşit paylaşıldı: her biri {total // parts}"
    if len(share) != 1:
        return None
    total_nums = _digits(body[0])
    if len(total_nums) != 1:
        return None
    total = total_nums[0]
    parts = _digits(share[0])
    if len(parts) != 1 or parts[0] == 0 or total % parts[0] != 0:
        return None
    each = total // parts[0]
    return each, f"{total} tane {parts[0]} eşit parçaya bölündü: her biri {each}"


_KAT = re.compile(r"(\d+|iki|üç|dört|beş|altı)\s+kat")
_WORD_NUM = {"iki": 2, "üç": 3, "dört": 4, "beş": 5, "altı": 6}


def _solve_multiplier(sentences: list[str]) -> tuple[int, str] | None:
    """'A'nın N var. Arkadaşının bundan K kat fazla. Kaç?' → N × K (tek kat ifadesi, tek olay)."""
    body, question = sentences[:-1], tr_lower(sentences[-1])
    if len(body) < 2 or "kaç" not in question:
        return None
    base = _digits(body[0])
    if len(base) != 1:
        return None
    for sentence in body[1:]:
        match = _KAT.search(tr_lower(sentence))
        if match:
            raw = match.group(1)
            factor = int(raw) if raw.isdigit() else _WORD_NUM[raw]
            value = base[0] * factor
            return value, f"{base[0]} × {factor} = {value}"
    return None


_CONTAINER = r"(?:kutu|paket|torba|sepet|tabak|kavanoz|koli|raf|vazo|tepsi|şişe|kasa|grup|sıra|saksı)"


def _solve_per_unit(sentences: list[str]) -> tuple[int, str] | None:
    """'Her kutuya E konuyor / Bir kutuda E var. X B kutu doldurdu/aldı. Toplam kaç?' → E × B."""
    question = tr_lower(sentences[-1])
    if "kaç" not in question:
        return None
    body = sentences[:-1]
    if len(body) != 2:
        return None
    each = [s for s in body if (tr_lower(s).startswith("her ")
                                or re.search(r"\b(?:her|bir)\s+" + _CONTAINER + r"(?:da|de|ta|te|ya|ye|na|ne)?\b", tr_lower(s)))
            and len(_digits(s)) == 1]
    counts = [s for s in body if s not in each and len(_digits(s)) == 1]
    if len(each) != 1 or len(counts) != 1:
        return None
    # Eğer soru grup/kutu sayısını soruyorsa ('kaç kutu') ya da diğer cümlede zaten 'toplam' verilmişse bu çarpma değildir.
    if re.search(r"kaç\s+" + _CONTAINER, question) or "toplam" in tr_lower(counts[0]):
        return None
    per, units = _digits(each[0])[0], _digits(counts[0])[0]
    return per * units, f"{units} × {per} = {per * units}"


_ANIMAL_LEGS = {
    "tavuk": 2, "ördek": 2, "kaz": 2, "hindi": 2, "kuş": 2, "serçe": 2, "güvercin": 2, "papağan": 2,
    "koyun": 4, "inek": 4, "keçi": 4, "at": 4, "köpek": 4, "kedi": 4, "tavşan": 4, "eşek": 4, "kuzu": 4, "aslan": 4, "fil": 4,
}


def _solve_animal_parts(sentences: list[str]) -> tuple[int, str] | None:
    """'Bir çiftlikte 6 tavuk ve 4 koyun var. Hayvanların toplam kaç ayağı/bacağı vardır?'"""
    question = tr_lower(sentences[-1])
    if not re.search(r"\bkaç\b.*?\b(?:ayak|ayağı|bacak|bacağı)\b", question):
        return None
    body_text = tr_lower(" ".join(sentences[:-1]))
    pairs = re.findall(r"(\d+)\s+([a-zçğıöşü]+)", body_text)
    if not pairs:
        return None
    total = 0
    parts_str = []
    for num_s, noun in pairs:
        n = int(num_s)
        legs = _ANIMAL_LEGS.get(noun)
        if legs is None:
            # Çoğul eki varsa ('tavuklar') soy
            base = re.sub(r"(?:lar|ler)$", "", noun)
            legs = _ANIMAL_LEGS.get(base)
        if legs is None:
            return None
        total += n * legs
        parts_str.append(f"{n} × {legs} = {n * legs}")
    return total, "; ".join(parts_str) + f"; toplam {total}"


def _solve_counts_sum(sentences: list[str]) -> tuple[int, str] | None:
    """'A'nın 5 var. B'nin de 7 var. İkisinin toplam kaç?' → sahiplik sayılarının toplamı."""
    question = tr_lower(sentences[-1])
    if "toplam" not in question or "kaç" not in question:
        return None
    body = sentences[:-1]
    counts: dict[str, int] = {}
    for sentence in body:
        low = tr_lower(sentence)
        found = _NAME_COUNT.findall(low)
        if not found:
            return None
        for name, value in found:
            counts[name] = int(value)
    if len(counts) < 2:
        return None
    total = sum(counts.values())
    return total, " + ".join(str(v) for v in counts.values()) + f" = {total}"


# --- DÜŞÜNCE VE RASTGELELİK (RANDOMNESS) BİÇİMLENDİRİCİSİ ---
INTRO_CHOICES = [
    "Adım adım düşünelim:",
    "Bunu adım adım hesaplayalım:",
    "Birlikte adım adım bakalım:",
    "Aşama aşama hesaplayalım:",
    "Durumu adım adım inceleyelim:",
]

CONCLUSION_CHOICES = [
    "Sonuç olarak, {ans}.",
    "Böylece {ans}.",
    "Demek ki {ans}.",
    "Hesabımıza göre {ans}.",
    "Bu durumda {ans}.",
]

_WORD_NUM_EXT = {
    "bir": 1, "iki": 2, "üç": 3, "dört": 4, "beş": 5, "altı": 6,
    "yedi": 7, "sekiz": 8, "dokuz": 9, "on": 10
}


def _word_to_num(s: str) -> int:
    s = (s or "").strip()
    if s.isdigit():
        return int(s)
    return _WORD_NUM_EXT.get(s, 0)


def format_thought(steps: list[str], conclusion: str) -> str:
    intro = random.choice(INTRO_CHOICES)
    formatted_steps = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))
    conc = random.choice(CONCLUSION_CHOICES).format(ans=conclusion)
    return f"{intro}\n{formatted_steps}\n{conc}"


def _solve_division_remainder(raw: str) -> tuple[int, str] | None:
    """'53 kalemi 6 öğrenciye eşit sayıda dağıtıyor. Her öğrenci kaç alır ve kaç artar?'"""
    low = tr_lower(raw)
    if not ("eşit" in low or "paylaştır" in low or "dağıt" in low):
        return None
    if not (("kaç" in low and ("artar" in low or "kalır" in low)) or "kalan" in low):
        return None
    nums = [int(x) for x in re.findall(r"\b\d+\b", raw)]
    if len(nums) != 2:
        return None
    total, parts = nums[0], nums[1]
    if parts <= 0:
        return None
    q = total // parts
    r = total % parts
    steps = [
        f"Toplam {total} nesneyi {parts} kişiye eşit olarak paylaştıralım.",
        f"Bölme işlemini yaparsak: {total} ÷ {parts} = {q} (kalan {r})."
    ]
    conc = f"her öğrenci/kişi {q} kalem alır ve {r} kalem artar"
    return q, format_thought(steps, conc)


def _solve_algebraic_number_puzzle(raw: str) -> tuple[int, str] | None:
    """'Bir sayının 3 katına 9 eklenince 45 oluyor. Bu sayı kaçtır?'"""
    low = tr_lower(raw)
    m = re.search(r"bir\s+sayının\s+(\d+|[a-zçğıöşü]+)\s+katı(?:na|ndan)\s+(\d+)\s+(eklenince|çıkarılınca|fazlası|eksiği)\s+(?:sayı\s+)?(\d+)\s*(?:oluyor|etmektedir|ise)", low)
    if m:
        kat_s, n2_s, op, n3_s = m.groups()
        kat = _word_to_num(kat_s)
        n2 = int(n2_s)
        target = int(n3_s)
        if kat <= 0:
            return None
        if op in ("eklenince", "fazlası"):
            diff = target - n2
            val = diff // kat
            steps = [
                f"Ters işlem yaparak eklenen {n2} sayısını {target}'den çıkaralım: {target} - {n2} = {diff}.",
                f"Sayının {kat} katı {diff} olduğuna göre, sayıyı bulmak için {kat}'e bölelim: {diff} ÷ {kat} = {val}."
            ]
            conc = f"bu sayı {val}'dir"
            return val, format_thought(steps, conc)
        elif op in ("çıkarılınca", "eksiği"):
            diff = target + n2
            val = diff // kat
            steps = [
                f"Ters işlem yaparak çıkarılan {n2} sayısını {target}'e ekleyelim: {target} + {n2} = {diff}.",
                f"Sayının {kat} katı {diff} olduğuna göre, sayıyı bulmak için {kat}'e bölelim: {diff} ÷ {kat} = {val}."
            ]
            conc = f"bu sayı {val}'dir"
            return val, format_thought(steps, conc)
    return None


def _solve_two_variable_heads_and_legs(raw: str) -> tuple[int, str] | None:
    """'Toplam 12 tavuk ve koyun bulunuyor... 34 ayağı var' ya da otomobil ve motosiklet tekerlek problemi."""
    low = tr_lower(raw)
    if not (re.search(r"\b(?:ayak|ayağı|tekerlek|tekerleği)\b", low) and re.search(r"\bkaç\b", low)):
        return None
    is_animals = "tavuk" in low and "koyun" in low
    is_vehicles = "otomobil" in low and "motosiklet" in low
    if not (is_animals or is_vehicles):
        return None
    nums = [int(x) for x in re.findall(r"\b\d+\b", raw)]
    total_heads = None
    total_legs = None
    m_heads = re.search(r"toplam\s+(\d+)\s+(?:hayvan|araç|tavuk\s+ve\s+koyun|otomobil\s+ve\s+motosiklet)", low)
    if m_heads:
        total_heads = int(m_heads.group(1))
    m_legs = re.search(r"toplam\s+(\d+)\s+(?:ayağı|ayak|tekerlek|tekerleği)|(\d+)\s+tekerlek\s+bulunmaktadır|(\d+)\s+ayağı\s+var", low)
    if m_legs:
        total_legs = int(next(g for g in m_legs.groups() if g))
    if total_heads is None or total_legs is None:
        if len(nums) == 2:
            total_heads, total_legs = sorted(nums)
        else:
            return None
    if is_animals:
        t1, l1 = "tavuk", 2
        t2, l2 = "koyun", 4
        unit_name = "ayak"
    else:
        t1, l1 = "motosiklet", 2
        t2, l2 = "otomobil", 4
        unit_name = "tekerlek"
    base_legs = total_heads * l1
    diff = total_legs - base_legs
    leg_diff = l2 - l1
    if diff < 0 or diff % leg_diff != 0:
        return None
    count2 = diff // leg_diff
    count1 = total_heads - count2
    if count1 < 0:
        return None
    steps = [
        f"Tüm {total_heads} varlığın {l1} {unit_name}li ({t1}) olduğunu varsayarsak: {total_heads} × {l1} = {base_legs} {unit_name} olurdu.",
        f"Mevcut {unit_name} sayısı ile fark: {total_legs} - {base_legs} = {diff} {unit_name}.",
        f"Her bir {t2}, {t1}'den {l2} - {l1} = {leg_diff} {unit_name} fazla olduğuna göre: {diff} ÷ {leg_diff} = {count2} {t2} vardır.",
        f"{t1.capitalize()} sayısı ise: {total_heads} - {count2} = {count1} {t1} olarak bulunur."
    ]
    conc = f"{count1} {t1} ve {count2} {t2} vardır"
    return count2, format_thought(steps, conc)


def _solve_age_problem(raw: str) -> tuple[int, str] | None:
    """'Ali 14 yaşındadır. Babasının yaşı Ali'nin yaşının 3 katıdır. 5 yıl sonra ikisinin yaşları toplamı kaç olur?'"""
    low = tr_lower(raw)
    if "yaş" not in low or "kat" not in low:
        return None
    m_base = re.search(r"(\d+)\s+yaşındadır", low)
    m_kat = re.search(r"(\d+|[a-zçğıöşü]+)\s+katıdır", low)
    m_year = re.search(r"(\d+)\s+yıl\s+(sonra|önce)", low)
    if not (m_base and m_kat and m_year):
        return None
    a1 = int(m_base.group(1))
    kat = _word_to_num(m_kat.group(1))
    y = int(m_year.group(1))
    direction = m_year.group(2)
    delta = y if direction == "sonra" else -y
    a2 = a1 * kat
    a1_fut = a1 + delta
    a2_fut = a2 + delta
    tot_fut = a1_fut + a2_fut
    steps = [
        f"Ali'nin şimdiki yaşı {a1}'tür.",
        f"Babasının şimdiki yaşı: {a1} × {kat} = {a2}.",
        f"{y} yıl {direction} Ali: {a1} + {delta} = {a1_fut} yaşında olur.",
        f"{y} yıl {direction} Babası: {a2} + {delta} = {a2_fut} yaşında olur.",
        f"İkisinin yaşları toplamı: {a1_fut} + {a2_fut} = {tot_fut} olur."
    ]
    conc = f"{y} yıl sonra ikisinin yaşları toplamı {tot_fut} olur"
    return tot_fut, format_thought(steps, conc)


def _solve_per_unit_sequence(raw: str) -> tuple[int, str] | None:
    """'Bir sınıfta 6 sıra var. Her sırada 4 öğrenci oturuyor. 7 öğrenci çıkarsa...' veya çikolata kutuları."""
    low = tr_lower(raw)
    m1 = re.search(r"(?:bir|her)\s+[\wçğıöşü]+\s+(\d+)\s+([^\d]+?)\s+(?:var|vardır)", low)
    m2 = re.search(r"her\s+[\wçğıöşü]+(?:da|de|ta|te)?\s+(\d+)\s+([^\d]+?)\s+(?:oturuyor|bulunur|var|vardır)", low)
    if not (m1 and m2):
        return None
    g1 = int(m1.group(1))
    g2, item = int(m2.group(1)), m2.group(2).strip()
    curr_val = g1 * g2
    steps = [
        f"Başlangıçtaki toplam {item} sayısı: {g1} × {g2} = {curr_val} {item}."
    ]
    sents = [s.strip() for s in re.split(r"[.!?]+", raw) if s.strip()]
    for s in sents:
        sub_low = tr_lower(s)
        if ("her kutuda" in sub_low or "her sırada" in sub_low or
            "bir mağazada" in sub_low or "bir sınıfta" in sub_low):
            continue
        parts = [p.strip() for p in re.split(r",\s*(?:ardından|sonra)\s*|\b(?:ardından|daha sonra)\b|\s+(?=kaç\b)", sub_low) if p.strip()]
        for p in parts:
            p_nums = [int(x) for x in re.findall(r"\b\d+\b", p)]
            if not p_nums:
                continue
            val = p_nums[0]
            if any(w in p for w in ("çıkarsa", "çıktı", "çıkıyor", "satıyor", "sattı", "harcadı", "verdi", "veriyor", "eksildi")):
                prev = curr_val
                curr_val -= val
                steps.append(f"{val} {item} eksilince/satılınca: {prev} - {val} = {curr_val} {item} kalır.")
            elif any(w in p for w in ("getiriyor", "getirdi", "alıyor", "aldı", "ekledi", "ekleniyor")):
                prev = curr_val
                curr_val += val
                steps.append(f"{val} {item} eklenince/getirilince: {prev} + {val} = {curr_val} {item} olur.")
    if len(steps) < 2:
        return None
    conc = f"son durumda geriye {curr_val} {item} kalır"
    return curr_val, format_thought(steps, conc)


_UNSUPPORTED = re.compile(
    r"yarı|çeyrek|üçte|dörtte|yüzde|%|/|saat|dakika|saniye|hafta|ons|mil|fit|inç|litre|kilo|pound|dolar|sent|santim|metre|gram|oranı|ortalama")


def _has_unsupported(raw: str) -> bool:
    return bool(_UNSUPPORTED.search(tr_lower(raw or "")))


_EQ_MODEL = None
_EQ_ARITY = {
    "ADD": 2, "SUB": 2, "MUL": 2, "DIV": 2,
    "ADD3": 3, "SUB_2_3": 3, "n1-n2+n3": 3, "n1+n2-n3": 3, "SUMMUL": 3,
}


def _equation_model_answer(raw: str, min_prob: float = 0.55) -> tuple[int, str] | None:
    """Öğrenilmiş şablon modeli: yalnız kurallar susarsa, güven yüksekse ve sayı sayısı etiketle uyuşuyorsa."""
    global _EQ_MODEL
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "data" / "equation_model.pt"
    if _EQ_MODEL is None:
        try:
            import torch
            _EQ_MODEL = torch.load(path, weights_only=False) if path.is_file() else False
        except Exception:
            _EQ_MODEL = False
    if not _EQ_MODEL:
        return None
    try:
        from tools.train_equation_model import answer, numbers, predict
    except Exception:
        return None
    s_clean = re.sub(r"^[\s>]*\d+\s*[.)]\s+", "", (raw or "").strip())
    nums = numbers(s_clean)
    label, prob = predict(_EQ_MODEL, s_clean)
    if prob < min_prob or _EQ_ARITY.get(label) != len(nums) or len(set(nums)) != len(nums):
        return None
    got = answer(_EQ_MODEL, s_clean, min_prob)
    if got is None or got[0] < 0 or abs(got[0] - round(got[0])) > 1e-9:
        return None
    value = int(round(got[0]))
    shown = " ".join(str(int(n)) if n == int(n) else str(n) for n in nums)
    return value, f"model etiketi {label} ({shown}) → {value}"


def solve_story(raw: str) -> tuple[int, str] | None:
    """Kurallar önce; hiçbiri çözmezse öğrenilmiş şablon modeli (yüksek güven, sayı sayısı eşleşmesi şartıyla)."""
    found = _solve_story(raw)
    if found is not None:
        return found
    if len(_sentences(raw)) < 2:
        return None
    return _equation_model_answer(raw)


def _covered(raw: str, result: tuple[int, str] | None) -> tuple[int, str] | None:
    """Eski kalıp/zincir çözücüleri için: problemdeki her sayı gerekçede geçmiyorsa tahmin yerine sus."""
    if result is None:
        return None
    s_clean = re.sub(r"^[\s>]*\d+\s*[.)]\s+", "", (raw or "").strip())
    used = set(re.findall(r"\d+", result[1])) | {str(result[0])}
    return result if set(re.findall(r"\d+", s_clean)) <= used else None


def _solve_part_subtract(sentences: list[str]) -> tuple[int, str] | None:
    """'Zeynep 36 sayfalık kitabın ilk gün 12 sayfasını, ikinci gün 9 sayfasını okuyor. Geriye kaç sayfa kalır?'"""
    question = tr_lower(sentences[-1])
    if not (set(_words(question)) & LEFT_WORDS):
        return None
    body = sentences[:-1]
    if len(body) != 1:
        return None
    s = tr_lower(body[0])
    nums = _digits(s)
    if len(nums) < 2:
        return None
    # İlk sayı bir bütün ('36 sayfalık', '50 metrelik') ya da toplam miktar, sonrakiler eksilen parçalar:
    if not re.search(r"\b\d+\s+\w+(?:lık|lik|luk|lük)\b", s):
        return None
    if _classify(_words(s)) != "remove" and not any(w in s for w in ("okuyor", "okudu", "yaptı", "tamamladı", "kullandı", "harcadı", "yedi", "sattı")):
        return None
    total = nums[0]
    rest = nums[1:]
    rem = total - sum(rest)
    if rem < 0:
        return None
    expr = f"{total} - " + " - ".join(str(x) for x in rest) + f" = {rem}"
    return rem, expr


def _solve_story(raw: str) -> tuple[int, str] | None:
    """(sonuç, gerekçe) ya da None. Önce anlamsal çözücü, sonra eski kalıp çözücüleri."""
    for solver in (_solve_per_unit_sequence, _solve_two_variable_heads_and_legs,
                   _solve_algebraic_number_puzzle, _solve_age_problem,
                   _solve_division_remainder):
        res = solver(raw)
        if res is not None:
            return res

    from nora.cognition.semantic_math import solve_semantic
    sentences = _sentences(raw)
    if len(sentences) < 2:
        return None
    # Oran, kesir, yüzde ve birim içeren problemler (saat, litre, fit...) bu çözücülerin kapsamı dışında:
    # yanlış cevap vermektense 'bilmiyorum' yolunda kalırlar.
    if _has_unsupported(raw):
        return None
    # Belirli kalıplar önce (hayvan ayağı, parça çıkarma, bölme, çarpan, kişi başı): anlamsal çözücü bunları yanlış okuyabilir.
    for detector in (_solve_animal_parts, _solve_part_subtract, _solve_divide, _solve_multiplier, _solve_per_unit):
        found = _covered(raw, detector(sentences))
        if found is not None:
            return found
    from nora.cognition.semantic_math import solve_inverse, solve_relation
    relation = solve_relation(raw)
    if relation is not None:
        return relation
    inverse = solve_inverse(raw)
    if inverse is not None:
        return inverse
    semantic = solve_semantic(raw)
    if semantic is not None:
        return semantic
    for detector in (_solve_compare, _solve_divide, _solve_multiplier, _solve_per_unit, _solve_counts_sum):
        found = _covered(raw, detector(sentences))
        if found is not None:
            return found
    question = sentences[-1]
    qwords = _words(question)
    if set(qwords) & {"fazla", "fazlaydı", "eksik", "az", "her", "eşit", "ve"}:
        return None
    if re.search(r"\bbazı\b|\bbirkaç\b|\bbiraz\b|\beşit\b|\bher\b|'\s*(?:erli|arlı|er|ar)\b", tr_lower(raw or "")):
        return None
    asks_quantity = "kaç" in qwords or any(w.startswith("kaç") for w in qwords) or "ne" in qwords and "kadar" in qwords
    if not asks_quantity:
        return None
    asks_left = bool(set(qwords) & LEFT_WORDS)
    asks_total = bool(set(qwords) & TOTAL_WORDS)
    if asks_left and asks_total:          # ikisi birden: belirsiz
        return None
    if not asks_left and not asks_total:  # 'kaç X var?' → olaylardan sonraki güncel durum
        asks_total = True

    body = sentences[:-1]
    # Soru cümlesi de olay taşıyabilir ("3 tanesini yerse kaç kalır?"): onu da olay olarak işle.
    body = body + [question] if _digits(question) and _classify(qwords) != "neutral" else body
    start_words = _words(body[0])
    start_nums = _digits(body[0])
    if _classify(start_words) != "neutral":
        return None
    if len(start_nums) == 1:
        total = start_nums[0]
        steps = [f"{total} ile başladım"]
    elif len(start_nums) == 2 and " ve " in tr_lower(body[0]):
        # İki özne aynı başlangıç cümlesinde: "A 5 elma ve B 7 elma" → toplam 12.
        total = sum(start_nums)
        steps = [f"{' + '.join(str(n) for n in start_nums)} = {total} ile başladım"]
    else:
        return None
    gained = total                          # başlangıç + eklenenler (toplam sorusu için)

    for sentence in body[1:]:
        words = _words(sentence)
        nums = _digits(sentence)
        kind = _classify(words)
        if not nums:
            continue                       # olay içermeyen (bağlam) cümle
        if len(nums) != 1 or kind == "neutral":
            return None                    # belirsiz cümle: tahmin yürütmem
        value = nums[0]
        if kind == "remove":
            total -= value
            steps.append(f"{value} eksildi, kalan {total}")
        else:
            total += value
            gained += value
            steps.append(f"{value} eklendi, toplam {total}")

    if total < 0:
        return None
    if asks_total:
        steps.append(f"şu anki toplam {total}")
        return _covered(raw, (total, "; ".join(steps)))
    steps.append(f"sonuç {total}")
    return _covered(raw, (total, "; ".join(steps)))


def answer_text(result: tuple[int, str]) -> tuple[str, str]:
    """(yanıt, gerekçe) çifti: düşünce süreci ve çeşitlilikle yanıt üretir."""
    value, why = result
    # Eğer why zaten format_thought tarafından üretilmiş tam bir düşünce metniyse
    if "\n" in why and any(intro.split()[0] in why for intro in INTRO_CHOICES):
        return why, f"Kelime problemi çözümü: {value}"
    # Eğer why adımlar içeriyorsa (örn: 'A; B; C')
    if ";" in why:
        parts = [p.strip() for p in why.split(";") if p.strip()]
        if len(parts) >= 2:
            thought = format_thought(parts, f"sonuç {value}")
            return thought, f"Kelime problemi çözümü: {value}"
    if "=" in why and len(why) > 5 and not why.startswith("model etiketi"):
        reply = random.choice([
            f"{why}; geriye {value} kalır.",
            f"Hesapladığımızda: {why}; sonuç {value}.",
            f"{why}; böylece {value} olur.",
            f"Adım adım hesaplarsak: {why}."
        ])
        return reply, f"Kelime problemi çözümü: {why}"
    return f"{value}.", f"Kelime problemi çözümü: {why}"
